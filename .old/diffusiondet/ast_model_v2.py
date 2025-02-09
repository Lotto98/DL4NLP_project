import torch
import torch.nn.functional as F
from detectron2.modeling import Backbone
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool
from torch.nn import ReLU
from transformers import ASTModel, ASTConfig

class ASTBackboneMultiScale(Backbone):
    def __init__(self, ast_model: ASTModel, out_features):
        """
        Args:
            ast_model: Pretrained AST model.
            out_channels: Number of output channels for feature maps.
        """
        super().__init__()
        self.ast_model = ast_model
        self.ast_model.layernorm.requires_grad_(False)
        self.out_channels = 256  # AST layers
        self.out_features = out_features
        self.num_layers = ast_model.config.num_hidden_layers  # AST layers
        
        self.layers = [2, 5, 9, 12]
        self.scales = [1, 2, 4, 8]
        self.divisibility = self.scales[-1]
        
        W = self.ast_model.config.num_mel_bins
        H = self.ast_model.config.max_length
        
        self.W_hidden = ((W - 16) // 10) + 1
        self.H_hidden = ((H - 16) // 10) + 1
        
        self.proj = torch.nn.ModuleList([
                            torch.nn.Sequential(*[
                                #torch.nn.AvgPool2d(kernel_size=1, stride=scale, padding=0),
                                torch.nn.Conv2d(self.out_channels, self.out_channels*scale, kernel_size=scale, stride=scale, padding=0),
                                torch.nn.LayerNorm([self.out_channels*scale, 
                                                    (self.H_hidden + (self.divisibility - self.H_hidden % self.divisibility) ) // scale if (self.H_hidden % self.divisibility != 0) else (self.H_hidden // scale), 
                                                    (self.W_hidden + (self.divisibility - self.W_hidden % self.divisibility) ) // scale if (self.W_hidden % self.divisibility != 0) else (self.W_hidden // scale)]),
                                ]) for scale in self.scales
                            ])
        
        self.activations = torch.nn.ModuleList([
                            ReLU() for _ in self.scales])
        
        self.iterations = 0

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, T*100, num bin).
        Returns:
            A dictionary of multi-scale feature maps.
        """
        #assert x.shape == (x.shape[0], 3000, 128), x.shape
        
        # Forward pass through the AST model
        outputs = self.ast_model(x, output_hidden_states=True, )
        hidden_states = outputs.hidden_states  # All transformer block outputs
        
        feature_maps = {}
    
        for i, layer_index in enumerate(self.layers):
            
            layer_output =  hidden_states[layer_index]
            B, L, C = layer_output.shape # B, L, 768
            
            #remove first two tokens
            layer_output = layer_output[:, 2:, :]
            
            #reshape layer output to 2D tensor (B, C, H_hidden, W_hidden)
            layer_output = layer_output.transpose(1, 2)
            layer_output = layer_output.view(B, C, self.H_hidden, self.W_hidden)
            
            #residual connection
            #x_down = F.interpolate(x.unsqueeze(1), size=(self.H_hidden, self.W_hidden) , mode='bicubic', align_corners=False)
            #layer_output = self.activation(layer_output + x_down)
            
            #padding layer output such that H and W are divisible by scale
            if (self.H_hidden % self.divisibility != 0):
                new_H = (self.H_hidden + (self.divisibility - self.H_hidden % self.divisibility))
            else:
                new_H = self.H_hidden
                
            if self.W_hidden % self.divisibility != 0:
                new_W = (self.W_hidden + (self.divisibility - self.W_hidden % self.divisibility))
            else:
                new_W = self.W_hidden
            
            if new_H != self.H_hidden or new_W != self.W_hidden:     
                layer_output = F.interpolate(layer_output, size=(new_H, new_W), mode='bicubic', align_corners=False)

            #apply projection
            #layer_output = F.interpolate(layer_output, size=(new_H//self.scales[i], new_W//self.scales[i]) , mode='bicubic', align_corners=False)
            layer_output = self.proj[i](layer_output)
            
            #plot the layer output
            #if self.iterations >= 500:
            #    import matplotlib.pyplot as plt
            #    plt.imshow(layer_output[0, 0].detach().cpu().numpy())
            #    plt.show()
            
            
            # residual connection
            #x_down = F.interpolate(x.unsqueeze(1), size=(new_H//self.scales[i], new_W//self.scales[i]) , mode='bicubic', align_corners=False)
            #layer_output = self.activations[i](layer_output + x_down)
            
            #if self.iterations >= 500:
            #    plt.imshow(x_down[0, 0].detach().cpu().numpy())
            #    plt.show()
                
            #    plt.imshow(layer_output[0, 0].detach().cpu().numpy())
            #    plt.show()
            
            feature_maps[f"AST_{layer_index}"] = layer_output
            self.iterations += 1
        
        return feature_maps

    def output_shape(self):
        return {
            f"AST_{layer_id}": ShapeSpec(channels=self.out_channels*scale, stride=scale) for scale, layer_id  in zip(self.scales, self.layers)
        }
    
@BACKBONE_REGISTRY.register()
def build_ASTModel_backbone(cfg, input_shape: ShapeSpec):
    """
    """
    
    config = ASTConfig()#.from_pretrained(cfg.MODEL.AST.PRETRAINED_MODEL)
    
    #print(config.hidden_size)
        
    # Modifica la lunghezza massima
    config.max_length = 1126
    config.num_mel_bins = 166
    config.hidden_size = 256
    config.num_attention_heads = 8
    
    model = ASTModel(config)
    
    #.from_pretrained(
    #    cfg.MODEL.AST.PRETRAINED_MODEL, 
    #    config=config,
    #    ignore_mismatched_sizes=True
    #)
    
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
        #top_block=LastLevelMaxPool(),
    )
    return backbone
