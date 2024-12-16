import torch
import torch.nn.functional as F
from detectron2.modeling import Backbone
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import FPN
from transformers import ASTModel, ASTConfig

class ASTBackboneMultiScale(Backbone):
    def __init__(self, ast_model, out_features):
        """
        Args:
            ast_model: Pretrained AST model.
            out_channels: Number of output channels for feature maps.
        """
        super().__init__()
        self.ast_model = ast_model
        self.out_channels = ast_model.config.hidden_size  # AST layers
        self.out_features = out_features
        self.num_layers = ast_model.config.num_hidden_layers  # AST layers
        
        self.norm = torch.nn.LayerNorm(self.out_channels)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, C, H, W).
        Returns:
            A dictionary of multi-scale feature maps.
        """
        print(x.shape)
        
        B, H, W = x.shape
        
        # Forward pass through the AST model
        outputs = self.ast_model(x, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # All transformer block outputs
        
        # Define multi-scale outputs (res2, res3, res4, res5)
        feature_maps = {}
        layers = [3, 6, 9, 12]
        scale = 1
        for layer_index in layers:
            
            # padding to make H and W even
            pad_input = (H % 2 == 1) or (W % 2 == 1)
            if pad_input:
                x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

            x0 = x[:, 0::scale, 0::scale]  # B H/2 W/2
            x1 = x[:, 1::scale, 0::scale]  # B H/2 W/2
            x2 = x[:, 0::scale, 1::scale]  # B H/2 W/2
            x3 = x[:, 1::scale, 1::scale]  # B H/2 W/2
            x = torch.cat([x0, x1, x2, x3], -1)  
            
            scale *= 2

        return feature_maps

    def output_shape(self):
        return {
            "AST_2": ShapeSpec(channels=self.out_channels, stride=8),
            "AST_3": ShapeSpec(channels=self.out_channels, stride=16),
            "AST_4": ShapeSpec(channels=self.out_channels, stride=32),
            "AST_5": ShapeSpec(channels=self.out_channels, stride=64),
        }
        
@BACKBONE_REGISTRY.register()
def build_ASTModel_backbone(cfg, input_shape: ShapeSpec):
    """
    """
    model = ASTBackboneMultiScale(
        ast_model = ASTModel(ASTConfig(
            max_length=cfg.INPUT.SAMPLING_RATE * cfg.INPUT.SECONDS_PER_SEGMENT,
        )),
        out_features = cfg.MODEL.AST.OUT_FEATURES
    )
    
    return model


@BACKBONE_REGISTRY.register()
def build_ASTModel_backbone_fpn(cfg, input_shape: ShapeSpec):
    """
    """
    bottom_up = build_ASTModel_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    print(in_features, out_channels)
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone

"""  
# AST feature extractor
import torch
from transformers import ASTFeatureExtractor, ASTModel
from detectron2.modeling import Backbone
from detectron2.modeling.backbone import ShapeSpec

class ASTBackboneWithFeatureExtractor(Backbone):
    def __init__(self, feature_extractor, ast_model, out_channels):
        " ""
        Args:
            feature_extractor: Pretrained ASTFeatureExtractor for preprocessing.
            ast_model: Pretrained ASTModel for feature extraction.
            out_channels: Number of output channels for feature maps.
        " ""
        super().__init__()
        self.feature_extractor = feature_extractor
        self.ast_model = ast_model
        self.out_channels = out_channels

    def forward(self, audio_inputs, sampling_rate=16000):
        " ""
        Args:
            audio_inputs: Batch of raw audio waveforms (list of 1D tensors or NumPy arrays).
            sampling_rate: Sampling rate of the audio inputs.
        
        Returns:
            A dictionary of multi-scale feature maps.
        " ""
        # Preprocess audio inputs using ASTFeatureExtractor
        inputs = self.feature_extractor(
            audio_inputs, sampling_rate=sampling_rate, return_tensors="pt", padding=True
        )
        spectrograms = inputs["input_values"]  # Processed log-mel spectrograms

        # Pass the spectrograms through AST model
        outputs = self.ast_model(spectrograms, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # Intermediate outputs from AST layers

        # Define multi-scale outputs (res2, res3, res4, res5)
        feature_maps = {}
        scales = ["res2", "res3", "res4", "res5"]
        for i, scale in enumerate(scales):
            features = hidden_states[i + 1]  # Transformer block output
            num_patches = int(features.size(1) ** 0.5)  # Assuming square patches
            feature_map = features.view(
                spectrograms.size(0), num_patches, num_patches, -1
            ).permute(0, 3, 1, 2)  # Reshape to (B, C, H, W)
            feature_maps[scale] = feature_map

        return feature_maps

    def output_shape(self):
        return {
            "res2": ShapeSpec(channels=self.out_channels, stride=8),
            "res3": ShapeSpec(channels=self.out_channels, stride=16),
            "res4": ShapeSpec(channels=self.out_channels, stride=32),
            "res5": ShapeSpec(channels=self.out_channels, stride=64),
        }
"""