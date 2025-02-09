import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.layers import ShapeSpec

from detectron2.modeling.backbone.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool

import fvcore.nn.weight_init as weight_init

"""
class SimpleConv2DModel(nn.Module):
    def __init__(self, width, height, n_classes):
        super(SimpleConv2DModel, self).__init__()
        # Define the layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)  # Output: 16xHxW
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1) # Output: 32xHxW
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1) # Output: 64xHxW
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1) # Output: 128xHxW
        self.fc1 = nn.Linear(128 * width* height, 128)  # Fully connected layer (adjust input size based on input image size)
        self.fc2 = nn.Linear(128, n_classes)          # Output layer for 10 classes
        
    def forward(self, x):
        # Apply convolutional layers
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2)  # Downsample (H/2 x W/2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2)  # Downsample again (H/4 x W/4)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2)  # Downsample again (H/8 x W/8)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, kernel_size=2)  # Downsample again (H/16 x W/16)
        
        # Flatten the tensor for fully connected layers
        x = x.view(x.size(0), -1)   # Flatten (batch_size, channels * height * width)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)            # No activation here; softmax is applied during loss computation
        
        return x

@BACKBONE_REGISTRY.register()
class ToyBackbone(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__()
        # create your own backbone
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=16, padding=3)

    def forward(self, image):
        return {"conv1": self.conv1(image)}

    def output_shape(self):
        return {"conv1": ShapeSpec(channels=64, stride=16)}
"""


class simple_backbone(Backbone):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)  
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1) 
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1) 
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1) 
        
    def forward(self, x):
        
        x1 = F.relu(self.conv1(x))
        x1 = F.max_pool2d(x1, kernel_size=2)
        
        x2 = F.relu(self.conv2(x1))
        x2 = F.max_pool2d(x2, kernel_size=2)
        
        x3 = F.relu(self.conv3(x2))
        x3 = F.max_pool2d(x3, kernel_size=2)
        
        x4 = F.relu(self.conv4(x3))
        x4 = F.max_pool2d(x4, kernel_size=2)
        
        return {
            "res2": x1,
            "res3": x2,
            "res4": x3,
            "res5": x4
        }
    
    def output_shape(self):
        return { #stride is the downsampling factor of the feature map w.r.t. the input image!!!
                #need to be power of 2
            "res2": ShapeSpec(channels=16, stride=2),
            "res3": ShapeSpec(channels=32, stride=4),
            "res4": ShapeSpec(channels=64, stride=8),
            "res5": ShapeSpec(channels=128, stride=16)
        }
    
    def init_weights(self):
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4]:
            weight_init.c2_msra_fill(layer)
    

@BACKBONE_REGISTRY.register()
def build_simple_backbone(conf, input_shape: ShapeSpec):
    """
    """
    model = simple_backbone()
    model.init_weights()
    
    return model


@BACKBONE_REGISTRY.register()
def build_simple_backbone_fpn(cfg, input_shape: ShapeSpec):
    """
    """
    bottom_up = build_simple_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone
