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
        self.out_channels = 1  # AST layers
        self.out_features = out_features
        self.num_layers = ast_model.config.num_hidden_layers  # AST layers
        
        #self.norm = torch.nn.LayerNorm(self.out_channels)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, W, H).
        Returns:
            A dictionary of multi-scale feature maps.
        """
        print(x.shape)
        
        B, W, H = x.shape
        
        # Forward pass through the AST model
        outputs = self.ast_model(x, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # All transformer block outputs
        
        # Define multi-scale outputs (res2, res3, res4, res5)
        feature_maps = {}
        layers = [3, 6, 9, 12]
        scale = 1
        for layer_index in layers:
            
            feature_maps[f"AST_{layer_index}"] = hidden_states[layer_index]#.permute(0, 2, 1)
            B, L, C = feature_maps[f"AST_{layer_index}"].shape
            
            # padding to make H and W even
            if L % scale != 0 or C % scale != 0:
                feature_maps[f"AST_{layer_index}"] = F.pad(feature_maps[f"AST_{layer_index}"], (0, C % scale, 0, L % scale))

            # average pooling
            feature_maps[f"AST_{layer_index}"] = F.avg_pool2d(
                feature_maps[f"AST_{layer_index}"], kernel_size=scale, stride=scale,
            )#.permute(0, 2, 1)
            
            # Reshape to (B, C, H, W)
            feature_maps[f"AST_{layer_index}"] = feature_maps[f"AST_{layer_index}"].unsqueeze_(1).permute(0, 1, 3, 2)
            
            scale *= 2
            
            #self.__out_channels = feature_maps[f"AST_{layer_index}"].shape[1]
            
            """
                # padding to make H and W even
                pad_input = (H % 2 == 1) or (W % 2 == 1)
                if pad_input:
                    x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

                x0 = x[:, 0::scale, 0::scale]  # B H/2 W/2
                x1 = x[:, 1::scale, 0::scale]  # B H/2 W/2
                x2 = x[:, 0::scale, 1::scale]  # B H/2 W/2
                x3 = x[:, 1::scale, 1::scale]  # B H/2 W/2
                x = torch.cat([x0, x1, x2, x3], -1)  
                
                
            """
        
        return feature_maps

    def output_shape(self):
        return {
            "AST_3": ShapeSpec(channels=self.out_channels, stride=1),
            "AST_6": ShapeSpec(channels=self.out_channels, stride=2),
            "AST_9": ShapeSpec(channels=self.out_channels, stride=4),
            "AST_12": ShapeSpec(channels=self.out_channels, stride=8),
        }
    
@BACKBONE_REGISTRY.register()
def build_ASTModel_backbone(cfg, input_shape: ShapeSpec):
    """
    """
    
    config = ASTConfig.from_pretrained(cfg.MODEL.AST.PRETRAINED_MODEL)
    
    print(config.hidden_size)
        
    # Modifica la lunghezza massima
    config.max_length = cfg.INPUT.SECONDS_PER_SEGMENT * cfg.INPUT.SAMPLING_RATE
    
    model = ASTModel.from_pretrained(
        cfg.MODEL.AST.PRETRAINED_MODEL, 
        config=config,
        ignore_mismatched_sizes=True
    )
    
    return ASTBackboneMultiScale(
        ast_model = model,
        out_features = cfg.MODEL.AST.OUT_FEATURES
    )


@BACKBONE_REGISTRY.register()
def build_ASTModel_backbone_fpn(cfg, input_shape: ShapeSpec):
    """
    """
    bottom_up = build_ASTModel_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = 1 #cfg.MODEL.FPN.OUT_CHANNELS
    print(in_features, out_channels)
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone
