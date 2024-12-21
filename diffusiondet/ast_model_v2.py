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
        self.out_channels = 768  # AST layers
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
        #print(x.shape)
        
        B, W, H = x.shape
        
        W_hidden = ((W - 16) // 10) + 1
        H_hidden = 12
        
        # Forward pass through the AST model
        outputs = self.ast_model(x, output_hidden_states=True, )
        hidden_states = outputs.hidden_states  # All transformer block outputs
        
        # Define multi-scale outputs (res2, res3, res4, res5)
        feature_maps = {}
        layers = [3, 6, 9, 12]
        scale = 1
        divisibility = 8
        for layer_index in layers:
            
            layer_output =  hidden_states[layer_index]
            B, L, C = layer_output.shape # B, 100*time, 768
            
            #get first two tokens
            tokens = layer_output[:, :2, :]
            
            #remove first two tokens
            layer_output = layer_output[:, 2:, :]
            layer_output = layer_output.transpose(1, 2)
            layer_output = layer_output.view(B, C, H_hidden, W_hidden)
            
            
            #padding layer output such that H and W are divisible by scale
            if (H_hidden % divisibility != 0 or W_hidden % divisibility != 0):
                layer_output = F.pad(layer_output, (0, (divisibility - W_hidden % divisibility), 0, (divisibility - H_hidden % divisibility)))

            #pad to the next power of 2
            #if scale == 1:
            #    new_H_hidden = 2**int(H_hidden.bit_length())
            #    new_W_hidden = 2**int(W_hidden.bit_length())
            #    layer_output = F.pad(layer_output, (0, new_W_hidden - layer_output.shape[3], 0, new_H_hidden - layer_output.shape[2]))
            
            # average pooling
            layer_output = F.avg_pool2d(
                layer_output, kernel_size=1, stride=scale,
            )
            
            feature_maps[f"AST_{layer_index}"] = layer_output
            
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
        
        #for key in feature_maps:
            #print(key, feature_maps[key].shape)
        
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
    config.max_length = cfg.INPUT.SECONDS_PER_SEGMENT * 100 #cfg.INPUT.SAMPLING_RATE
    
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
