import lightning as L
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict


class Backbone(nn.Module):
    def __init__(self, model, target_layer, trainable_layers, in_features, out_features):
        super().__init__()

        backbone = self._extract_backbone(model, target_layer)
        frozen_backbone = self._freeze_layers(backbone, trainable_layers)
        
        self.backbone = frozen_backbone
        self.flatten = nn.Flatten(1)
        self.fc1 = nn.Linear(in_features=in_features, out_features=4096)
        self.fc2 = nn.Linear(4096, out_features=out_features)


    def _extract_backbone(self, model, target_layer):
        layers = OrderedDict()

        for name, layer in model.named_children():
            layers[name] = layer
            if name == target_layer:
                break

        return nn.Sequential(layers)
    

    def _freeze_layers(self, model, trainable_layers):
        for param in model.parameters():
            param.requires_grad = False

        for name, param in model.named_parameters():
            for trainable_l in trainable_layers:
                if trainable_l in name:
                    param.requires_grad = True
                    print(f"Unfrozen: {name}")  

        return model


    def forward(self, x):
        x = self.backbone(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return self.fc2(x)