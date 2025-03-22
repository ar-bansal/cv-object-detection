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
        # (batch_size, S, S, 5B + C)
        bbox_coords = inputs[..., :self.B * 5]
        bbox_coords = bbox_coords.view(bbox_coords.shape[0], self.S, self.S, self.B, 5)

        # Shape = (batch_size, S, S, B, 1)
        x_rel = bbox_coords[..., 0]
        y_rel = bbox_coords[..., 1]
        w_rel = bbox_coords[..., 2]
        h_rel = bbox_coords[..., 3]


        grid_x = torch.arange(self.S).repeat(self.S, 1)
        grid_y = torch.arange(self.S).repeat(self.S, 1).T    
        grid_x = grid_x.unsqueeze(0).unsqueeze(-1).expand_as(x_rel)  # Shape (batch, S, S, B)
        grid_y = grid_y.unsqueeze(0).unsqueeze(-1).expand_as(y_rel)
        
        # Get center coordinates
        x_abs = (grid_x + x_rel) / self.S
        y_abs = (grid_y + y_rel) / self.S

        xmin = x_abs - w_rel / 2
        xmax = x_abs + w_rel / 2
        ymin = y_abs - h_rel / 2
        ymax = y_abs + h_rel / 2

        bboxes = torch.stack([xmin, xmax, ymin, ymax], dim=-1)
        # Shape = (batch_size, S, S, B, 4)

        return bboxes
    

    def _get_iou(self, pred, true):
        # Assuming that the input is of shape (batch_size, S, S, B, 4)

        # Shape = (batch_size, S, S, B)
        pred_xmin = pred[..., 0]
        pred_xmax = pred[..., 1]
        pred_ymin = pred[..., 2]
        pred_ymax = pred[..., 3]

        true_xmin = true[..., 0]
        true_xmax = true[..., 1]
        true_ymin = true[..., 2]
        true_ymax = true[..., 3]

        # Shape = (batch_size, S, S, B)
        intersection_xmin = torch.max(pred_xmin, true_xmin)
        intersection_ymin = torch.max(pred_ymin, true_ymin)
        intersection_xmax = torch.min(pred_xmax, true_xmax)
        intersection_ymax = torch.min(pred_ymax, true_ymax)

        # Shape = (batch_size, S, S, B)
        intersection_area = (intersection_xmax - intersection_xmin) * (intersection_ymax - intersection_ymin)

        pred_area = (pred_xmax - pred_xmin) * (pred_ymax - pred_ymin)
        true_area = (true_xmax - true_xmin) * (true_ymax - true_ymin)

        # Shape = (batch_size, S, S, B)
        union_area = pred_area + true_area - intersection_area

        return intersection_area / union_area
    

    def localization_loss(self, predictions: torch.Tensor, ground_truth: torch.Tensor, object_present: torch.Tensor):
        """
        Calculate the localization loss for the predicted 
        bounding box. 
        """

        # self(x) shape is (batch_size, S * S * (B * 5 + C))
        # num_samples = predictions.shape[0]
        # outputs = torch.reshape(predictions, (num_samples, self.S, self.S, (self.B * 5 + self.C)))
        # shape = (batch_size, S, S, 5B + C)
    
        # Convert (x, y) to relative to image from relative to grid cell
        # pred_bboxes = self._convert_yolo_to_absolute(outputs)
        # true_bboxes = self._convert_yolo_to_absolute(ground_truth)

        # ious = self._get_iou(pred_bboxes, true_bboxes)
        # Shape = (batch_size, S, S, B)

        # Get the box with the highest IoU with the ground truth box. 
        # Here, class does not matter
        # mask = torch.argmax(ious, dim=-1).unsqueeze(-1)
        # Shape = (batch_size, S, S, 1)

        # pred_x = torch.gather(predictions[..., 0], dim=-1, index=object_present)
        # pred_y = torch.gather(predictions[..., 1], dim=-1, index=object_present)
        # true_x = torch.gather(ground_truth[..., 0], dim=-1, index=object_present)
        # true_y = torch.gather(ground_truth[..., 1], dim=-1, index=object_present)

        pred_x = predictions[..., 0]
        pred_y = predictions[..., 1]
        true_x = ground_truth[..., 0]
        true_y = ground_truth[..., 1]

        # loc_loss_x = torch.gather(true_x - pred_x, dim=-1, index=object_present) 
        # loc_loss_y = torch.gather(true_y - pred_y, dim=-1, index=object_present)
        
        loc_loss_x = true_x - pred_x
        loc_loss_y = true_y - pred_y

        # Only select grid locations which are responsible for detecting the 
        # object.
        center_loss = torch.sum(
            torch.gather(
                loc_loss_x ** 2 + loc_loss_y ** 2, 
                dim=-1, 
                index=object_present
            )
        )


        ## Calculate loss for box dimensions
        pred_w = torch.sqrt(predictions[..., 2])
        pred_h = torch.sqrt(predictions[..., 3])
        true_w = torch.sqrt(ground_truth[..., 2])
        true_h = torch.sqrt(ground_truth[..., 3])

        loc_loss_w = true_w - pred_w
        loc_loss_h = true_h - pred_h

        dimension_loss = torch.sum(
            torch.gather(
                loc_loss_w ** 2 + loc_loss_h ** 2, 
                dim=-1, 
                mask=object_present
            )
        )

        return self.lambda_coord * (center_loss + dimension_loss)
    

    def confidence_loss(self, predictions: torch.Tensor, ground_truth: torch.Tensor, object_present: torch.Tensor):
        obj_conf_loss = torch.sum(
            torch.gather(
                (ground_truth - predictions) ** 2, 
                dim=-1, 
                index=object_present
            )
        )

        # Create boolean mask for no object present in grid cell
        grid = torch.arange(self.B).reshape((predictions.shape[0], self.S, self.S, self.B))
        no_object_present = grid != object_present

        noobj_conf_loss = torch.sum(
            torch.where(
                no_object_present == True, 
                ground_truth - predictions, 
                0
            )
        )

        return obj_conf_loss + self.lambda_noobj * noobj_conf_loss
        

    def forward(self, x, *args, **kwargs):
        pass


    def training_step(self, batch, batch_idx, *args, **kwargs):
        x, y = batch
        # TODO: ensure that y repeats the bbox locations
        outputs = self(x)

        num_samples = outputs.shape[0]
        outputs = torch.reshape(outputs, (num_samples, self.S, self.S, (self.B * 5 + self.C)))

        # Convert (x, y) to relative to image from relative to grid cell
        pred_bboxes = self._convert_yolo_to_absolute(outputs)
        true_bboxes = self._convert_yolo_to_absolute(y)

        ious = self._get_iou(pred_bboxes, true_bboxes)

        obj_present_mask = torch.argmax(ious, dim=-1).unsqueeze(-1)


        localization_loss = self.localization_loss(
            predictions=pred_bboxes, 
            ground_truth=true_bboxes, 
            object_present=obj_present_mask
        )


        pred_conf = outputs[..., 4:self.B * 5:5]
        true_conf = y[..., 4:self.B * 5:5]
        confidence_loss = self.confidence_loss(
            predictions=pred_conf, 
            ground_truth=true_conf, 
            object_present=obj_present_mask
        )

        classification_loss = self.classification_loss()
    

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        x, y = batch
        outputs = self(x)

        loc_loss = self.localization_loss(outputs, y)
        conf_loss = self.confidence_loss(outputs, y)
        prob_loss = self.class_probability_loss(outputs, y)
        noobj_loss = self.no_object_loss(outputs, y)

        # val_loss = (
        #     self.loc_loss_weight * loc_loss + 
        #     self.confidence_loss_weight * conf_loss + 
        #     self.cls_prob_loss_weight * prob_loss + 
        #     self.noobj_loss_weight * noobj_loss
        # )


    def test_step(self, batch, batch_idx):
        images, bboxes = batch

        preds = self(images)
        
        # TODO: Build the rest of the test logic

    
    def configure_optimizers(self):
        return optim.SGD(
            self.parameters(), 
            lr=0.01, 
            momentum=0.9, 
            weight_decay=0.0005
        )
    