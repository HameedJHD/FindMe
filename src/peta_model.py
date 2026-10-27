import torch
import torch.nn as nn
from torchvision import models

class PetaAttrModel(nn.Module):
    def __init__(self, num_binary=61, num_color_classes=11):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # pool -> (B,2048,1,1)
        self.bn = nn.BatchNorm1d(2048)
        self.act = nn.ReLU(inplace=True)
        self.bin_head = nn.Linear(2048, num_binary)
        self.color_heads = nn.ModuleList([nn.Linear(2048, num_color_classes) for _ in range(4)])

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.bn(x)
        x = self.act(x)
        bin_logits = self.bin_head(x)
        color_logits = [head(x) for head in self.color_heads]  # list of 4 tensors
        return bin_logits, color_logits