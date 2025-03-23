import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
from utils import Backbone


class YoloV1(L.LightningModule):
    def __init__(
            self, 
            num_classes: int, 
            num_boxes: int,
            grid_size: int,
            backbone: Backbone, 
            lambda_coord: float, 
            lambda_noobj: float, 
            *args, 
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.S = grid_size
        self.B = num_boxes
        self.C = num_classes

        self.backbone = backbone

        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj


    def _convert_yolo_to_absolute(self, inputs):
        # Assuming that the inputs have already been converted to shape 
        # (N, S, S, 5B + C)
        bbox_coords = inputs[..., :self.B * 5]
        bbox_coords = bbox_coords.view(bbox_coords.shape[0], self.S, self.S, self.B, 5)
        # (N, S, S, B, 5)

        # Shape = (N, S, S, B)
        x_rel = bbox_coords[..., 0]
        y_rel = bbox_coords[..., 1]
        w_rel = bbox_coords[..., 2]
        h_rel = bbox_coords[..., 3]


        grid_x = torch.arange(self.S).repeat(self.S, 1)
        grid_y = torch.arange(self.S).repeat(self.S, 1).T    
        grid_x = grid_x.unsqueeze(0).unsqueeze(-1).expand_as(x_rel)  # Shape (N, S, S, B)
        grid_y = grid_y.unsqueeze(0).unsqueeze(-1).expand_as(y_rel)
        
        # Get center coordinates
        x_abs = (grid_x + x_rel) / self.S
        y_abs = (grid_y + y_rel) / self.S

        xmin = x_abs - w_rel / 2
        xmax = x_abs + w_rel / 2
        ymin = y_abs - h_rel / 2
        ymax = y_abs + h_rel / 2

        bboxes = torch.stack([xmin, xmax, ymin, ymax], dim=-1)
        # Shape = (N, S, S, B, 4)

        return bboxes
    

    def _get_iou(self, pred, true):
        # Assuming that the input is of shape (N, S, S, B, 4)

        # Shape = (N, S, S, B)
        pred_xmin = pred[..., 0]
        pred_xmax = pred[..., 1]
        pred_ymin = pred[..., 2]
        pred_ymax = pred[..., 3]

        true_xmin = true[..., 0]
        true_xmax = true[..., 1]
        true_ymin = true[..., 2]
        true_ymax = true[..., 3]

        intersection_xmin = torch.max(pred_xmin, true_xmin)
        intersection_ymin = torch.max(pred_ymin, true_ymin)
        intersection_xmax = torch.min(pred_xmax, true_xmax)
        intersection_ymax = torch.min(pred_ymax, true_ymax)
        # Shape = (N, S, S, B)

        intersection_area = (intersection_xmax - intersection_xmin) * (intersection_ymax - intersection_ymin)
        # Shape = (N, S, S, B)

        pred_area = (pred_xmax - pred_xmin) * (pred_ymax - pred_ymin)
        true_area = (true_xmax - true_xmin) * (true_ymax - true_ymin)

        # Shape = (N, S, S, B)
        union_area = pred_area + true_area - intersection_area

        return intersection_area / (union_area + 1e-6)
    

    def _localization_loss(self, predictions: torch.Tensor, ground_truth: torch.Tensor, obj_mask: torch.Tensor, best_bbox_mask: torch.Tensor):
        """
        Calculate the localization loss for the predicted 
        bounding box. 
        """
        pred_x = predictions[..., 0]
        pred_y = predictions[..., 1]
        true_x = ground_truth[..., 0]
        true_y = ground_truth[..., 1]
        # (N, S, S, B)
        
        loc_loss_x = true_x - pred_x
        loc_loss_y = true_y - pred_y
        # (N, S, S, B)

        # Only select grid cells that are responsible for detecting the object, by 
        # only keeping those that have the highest IoU with the ground truth. Also,
        # intersect with the mask for ground truth locations that actually have objs
        center_loss = torch.sum(
            torch.where(
                obj_mask & best_bbox_mask, 
                loc_loss_x ** 2 + loc_loss_y ** 2, 
                0
            )
        )

        ## Calculate loss for box dimensions
        pred_w = torch.sqrt(predictions[..., 2])
        pred_h = torch.sqrt(predictions[..., 3])
        true_w = torch.sqrt(ground_truth[..., 2])
        true_h = torch.sqrt(ground_truth[..., 3])
        # (N, S, S, B)

        loc_loss_w = true_w - pred_w
        loc_loss_h = true_h - pred_h
        # (N, S, S, B)

        dimension_loss = torch.sum(
            torch.where(
                obj_mask & best_bbox_mask,
                loc_loss_w ** 2 + loc_loss_h ** 2, 
                0
            )
        )

        return self.lambda_coord * (center_loss + dimension_loss)
    

    def _confidence_loss(self, predictions: torch.Tensor, ground_truth: torch.Tensor, obj_mask: torch.Tensor, best_bbox_mask: torch.Tensor):
        # Confidence loss. We only consider cells that are responsible for 
        # detecting the object, and actually have an object. 
        obj_conf_loss = torch.sum(
            torch.where(
                obj_mask & best_bbox_mask, 
                (ground_truth - predictions) ** 2, 
                0
            )
        )

        # Consider cells that do not have objects, or those that have 
        # boxes but are not responsible for detection because they are not 
        # the best boxes.
        noobj_conf_loss = torch.sum(
            torch.where(
                (~obj_mask) | (obj_mask & ~best_bbox_mask), 
                (ground_truth - predictions) ** 2, 
                0
            )
        )

        return obj_conf_loss + self.lambda_noobj * noobj_conf_loss
    

    def _classification_loss(self, predictions: torch.Tensor, ground_truth: torch.Tensor, obj_mask: torch.Tensor):

        # For cells containing objects, we calculate the classification loss 
        return torch.sum(
            torch.where(
                obj_mask, 
                (ground_truth - predictions) ** 2, 
                0
            )
        )
        

    def forward(self, x, *args, **kwargs):
        pass


    def training_step(self, batch, batch_idx, *args, **kwargs):
        x, y = batch
        # TODO: ensure that y repeats the bbox locations
        outputs = self(x)

        num_samples = outputs.shape[0]
        outputs = torch.reshape(outputs, (num_samples, self.S, self.S, (self.B * 5 + self.C)))
        # Shape = (N, S, S, 5B + C)

        # Convert (x, y) to relative to image from relative to grid cell
        pred_bboxes = self._convert_yolo_to_absolute(outputs)
        true_bboxes = self._convert_yolo_to_absolute(y)
        # (N, S, S, B, 4)

        ious = self._get_iou(pred_bboxes, true_bboxes)
        # (N, S, S, B)

        grid = torch.arange(self.B).expand((num_samples, self.S, self.S, self.B))
        best_bbox_index = torch.argmax(ious, dim=-1, keepdim=True)
        best_bbox_mask = grid == best_bbox_index
        # (N, S, S, B)

        # Where confidence of ground truth == 1. Here, we follow Yolo V1's design
        # and assume that each grid cell is responsible for detecting at most 1 object
        obj_mask = (y[..., 4] == 1).unsqueeze(-1)
        # (N, S, S, 1)        
        # Not expanding here, so that torch can dynamically broadcast to whichever length, 
        # as needed

        localization_loss = self._localization_loss(
            predictions=pred_bboxes, 
            ground_truth=true_bboxes, 
            obj_mask=obj_mask,
            best_bbox_mask=best_bbox_mask
        )


        pred_conf = outputs[..., 4:self.B * 5:5]
        true_conf = y[..., 4:self.B * 5:5]
        # (N, S, S, B)
        confidence_loss = self._confidence_loss(
            predictions=pred_conf, 
            ground_truth=true_conf, 
            obj_mask=obj_mask, 
            best_bbox_mask=best_bbox_mask
        )

        pred_cls = outputs[..., self.B * 5:]
        true_cls = y[..., self.B * 5:]
        # (N, S, S, C)
        classification_loss = self._classification_loss(
            predictions=pred_cls, 
            ground_truth=true_cls, 
            obj_mask=obj_mask
        )

        train_loss = localization_loss + confidence_loss + classification_loss

        self.log("train_loc_loss", localization_loss)
        self.log("train_conf_loss", confidence_loss)
        self.log("train_cls_loss", classification_loss)
        # self.log("train_loss", train_loss)

        return train_loss
    

    # def validation_step(self, batch, batch_idx, *args, **kwargs):
    #     x, y = batch
    #     outputs = self(x)


    # def test_step(self, batch, batch_idx):
    #     images, bboxes = batch

    #     preds = self(images)
        
    #     # TODO: Build the rest of the test logic

    
    def configure_optimizers(self):
        return optim.SGD(
            self.parameters(), 
            lr=0.01, 
            momentum=0.9, 
            weight_decay=0.0005
        )
    