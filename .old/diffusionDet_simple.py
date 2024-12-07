from math import sqrt, pi
import math
from random import randint
import random
from numpy import extract, linspace
import torch
from torch import nn
from detectron2.modeling.poolers import ROIPooler

"""
HOW TO READ THIS FILE:

I copied the pseudocode from the paper and pasted it here. It is not a working code. 
I only enriched it with some comments and some functions that are not completely okey yet.

The purpose of this file is to understand the training/evaluation algorithm.

THE FUNCTIONS THAT I COPIED FROM THE PAPER ARE:
- train_loss
- infer

THE FUNCTIONS THAT I DEFINED ARE:
- image_encoder: It is a placeholder for the image encoder. Not further explanations are needed.
- pad_boxes: It pads the ground truth boxes to N boxes. (should work)
- cosine_beta_schedule: It generates cosine beta schedule. (should work)
- alpha_cumprod: It generates alpha cumprod. (should work)
- SinusoidalPositionEmbeddings: It generates sinusoidal position embeddings. (should work)
- Added main classes with line comments and explanations... hope that everything is correct...
- set_prediction_loss: It computes the loss for DiffusionDet. NOT DEFINED YET.
- ddim_step: It is the ddim step. (should work)
- predict_noise_from_start: It predicts noise from start. (should work)
- box_renewal: It renews the boxes. NOT DEFINED YET.
"""


def image_encoder(images):
    """
    images: [B, H, W, 3]
    """
    # Encode image features
    pass


def pad_boxes(gt_boxes, num_proposals=300):
    """
    Code taken from: detector.py in prepare_diffusion_concat(...) function.
    
    Pads the ground truth boxes to a specified number of proposals.

    Args:
        gt_boxes (Tensor): Ground truth boxes with shape [B, *, 4].
        num_proposals (int): Number of proposals to pad to. Default is 300.

    Returns:
        Tensor: Padded boxes with shape [B, num_proposals, 4].
    """
    # Pad gt_boxes to N
    num_gt = gt_boxes.shape[0]
    if not num_gt:  # generate fake gt boxes if empty gt boxes
        gt_boxes = torch.as_tensor([[0.5, 0.5, 1., 1.]], dtype=torch.float, device="cuda")
        num_gt = 1

    if num_gt < num_proposals:
        box_placeholder = torch.randn(num_proposals - num_gt, 4,
                                      device="cuda") / 6. + 0.5  # 3sigma = 1/2 --> sigma: 1/6
        box_placeholder[:, 2:] = torch.clip(box_placeholder[:, 2:], min=1e-4)
        x_start = torch.cat((gt_boxes, box_placeholder), dim=0)
    elif num_gt > num_proposals:
        select_mask = [True] * num_proposals + [False] * (num_gt - num_proposals)
        random.shuffle(select_mask)
        x_start = gt_boxes[select_mask]
    else:
        x_start = gt_boxes

    return x_start


def cosine_beta_schedule(timesteps, s=0.008):
    """
    code taken from: detector.py in cosine_beta_schedule(...) function.
    
    Generates a cosine beta schedule for denoising diffusion probabilistic models as proposed in https://openreview.net/forum?id=-NEXDKk8gZ.
    The authors propose a new noise schedule based on a cosine function. This schedule changes the noise level more
    gradually, especially at the beginning and end of the process, which helps in preserving information better.
    The new schedule improves both the log-likelihood and the Fréchet Inception Distance (FID), indicating better
    sample quality and more accurate modeling of the data distribution.

    from the paper:
    Our cosine schedule is designed to have a linear drop-off of t in the middle of the
    process, while changing very little near the extreme soft=0 and t=T to prevent abrupt changes in noise level.
    The small offsets in our schedule prevent t from being too small near t = 0, since we found that
    having tiny amounts of noise at the beginning of the process made it hard for the network to predict accurately enough.
    In particular, we selected s such that \sqrt(\beta_0) was slightly smaller than the pixel bin size, 1/1275.
    We chose to use \cos^2 in particular because it is a common mathematical function with the shape we were looking for.
    This choice was arbitrary, and we expect that many other functions with similar shapes would work as well.

    Args:
        timesteps (int): Number of timesteps.
        s (float): Small constant to prevent division by zero. Default is 0.008.

    Returns:
        Tensor: Beta values for each timestep.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * pi * 0.5)  2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def alpha_cumprod(timesteps):
    """
    code taken from: detector.py in __init__(...) function under #build diffusion comment.

    Generates the cumulative product of alphas for a given number of timesteps.

    Args:
        timesteps (int): Number of timesteps.

    Returns:
        Tensor: Cumulative product of alphas.
    """
    # alpha_cumprod
    betas = cosine_beta_schedule(timesteps)
    alphas = 1. - betas
    return torch.cumprod(alphas, dim=0)


"""
code taken from: head.py in SinusoidalPositionEmbeddings(...) class.
The `SinusoidalPositionEmbeddings` class is a PyTorch module that generates sinusoidal position embeddings for a
given input tensor. These embeddings are used to encode positional information, which is crucial for models like
transformers that do not inherently capture the order of input sequences.

Initialization (`__init__` method):
    - The constructor takes a single argument `dim`, which specifies the dimension of the embeddings.
    - It initializes the `dim` attribute with the provided value.

Forward Pass (`forward` method):
    - The `forward` method takes a tensor `time` as input, which represents the time steps or positions.
    - It calculates the `half_dim` as half of the embedding dimension.
    - It computes a scaling factor using the logarithm of 10000 divided by `half_dim - 1`.
    - It generates a range of values from 0 to `half_dim - 1` and scales them using the computed factor.
    - It multiplies the `time` tensor with the scaled values to create the embeddings.
    - It applies the sine and cosine functions to the embeddings and concatenates the results along the last dimension.
"""


class SinusoidalPositionEmbeddings(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


"""
Detection Decoder part of the model.
"""


class DynamicHead(nn.Module):
    # cfg: Contains model configuration details.
    # roi_input_shape: Specifies the input feature map properties (channels, stride, etc.).
    def __init__(self, cfg, roi_input_shape):
        """
        RoI Pooler Initialization:
        Creates a region of interest (RoI) pooling layer using _init_box_pooler, which extracts fixed-size features for each proposal from multi-scale feature maps.

        RCNN Heads Initialization:
        Defines the number of iterative refinement stages (num_heads).
        Initializes multiple RCNN heads (self.head_series) using the RCNNHead class, each responsible for one refinement step.

        Time Embedding Layer:
        Creates a Gaussian random feature embedding for time steps.
        Encodes the denoising step (t) into a higher-dimensional representation using sinusoidal embeddings and MLP layers.

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(d_model),
            nn.Linear(d_model, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        Loss Configuration:
        Configures the model to use focal loss or federated loss, adjusting bias initialization based on the prior probability (PRIOR_PROB).

        Parameter Initialization:
        Initializes model weights and biases using Xavier uniform initialization.
        Specifically adjusts biases for classification layers if focal or federated loss is used.

        self.box_pooler: RoI pooling layer for extracting proposal features.
        self.head_series: A stack of RCNN heads for iterative proposal refinement.
        self.time_mlp: Embedding layer for encoding time steps into feature space.
        self.use_focal: Flag for using focal loss for classification.
        self.use_fed_loss: Flag for using federated loss.
        self.bias_value: Bias initialization value for focal/federated loss.
        self.num_heads: Number of iterative refinement stages.
        self.return_intermediate: Whether to return intermediate outputs for deep supervision.
        """

        pass

    def _reset_parameters(self):
        # init all parameters.
        pass

    @staticmethod
    def _init_box_pooler(cfg, input_shape):
        # Extracts configuration for the RoI pooler:

        # in_features: Feature maps to use for pooling.
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        # pooler_resolution: Size of pooled feature maps.
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        # pooler_scales: Rescaling factor for each feature map.
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        # sampling_ratio: Number of samples per RoI bin.
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        # pooler_type: Type of pooling (ROIAlignV2).
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE

        # Ensures all input feature maps have the same number of channels.
        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels

        # Creates and returns an RoI pooler.
        # ROIPooler is a class in the Detectron2 library used for Region of Interest (RoI) pooling.
        # It extracts fixed-size feature maps from variable-sized regions of interest in the input feature maps.
        # This is commonly used in object detection models to pool features from regions proposed by the Region Proposal
        # Network (RPN) or other proposal mechanisms.
        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        return box_pooler

    def forward(self, features, init_bboxes, t, init_features):
        # assert t shape (batch_size)
        # time_mlp(t): Embeds the diffusion timestep into a feature vector.
        time = self.time_mlp(t)

        # Initializes variables for intermediate predictions, batch size (bs), and bounding boxes.
        inter_class_logits = []
        inter_pred_bboxes = []
        bs = len(features[0])
        bboxes = init_bboxes
        num_boxes = bboxes.shape[1]

        # Handles initial proposal features if provided, replicating them for each batch.
        if init_features is not None:
            init_features = init_features[None].repeat(1, bs, 1)
            proposal_features = init_features.clone()
        else:
            proposal_features = None

        # Iterates over the detection heads
        for head_idx, rcnn_head in enumerate(self.head_series):
            """
            class_logits, pred_bboxes: Predictions from the current head.
            proposal_features: Updated features for proposals.
            
            The outputs of the RCNNHead include:

            class_logits: Class scores for each proposal, shape (N, nr_boxes, num_classes), where:
            - N is the batch size.
            - nr_boxes is the number of proposals per image.
            - num_classes is the number of object categories.
            
            pred_bboxes: Refined bounding box coordinates, shape (N, nr_boxes, 4):
            - Each bounding box is represented as [x1, y1, x2, y2].
            
            obj_features: Enhanced object features after proposal refinement, shape (N, nr_boxes, d_model):
            - Used for further interactions or intermediate outputs if needed.
            """
            class_logits, pred_bboxes, proposal_features = rcnn_head(features, bboxes, proposal_features,
                                                                     self.box_pooler, time)

            # Stores intermediate predictions if return_intermediate is enabled.
            if self.return_intermediate:
                inter_class_logits.append(class_logits)
                inter_pred_bboxes.append(pred_bboxes)
            # Updates bounding boxes for the next iteration.
            bboxes = pred_bboxes.detach()

        # Returns predictions
        if self.return_intermediate:
            return torch.stack(inter_class_logits), torch.stack(inter_pred_bboxes)

        return class_logits[None], pred_bboxes[None]


class RCNNHead(nn.Module):
    # Arguments:
    # cfg: Configuration dictionary.
    # d_model: Dimensionality of features.
    # num_classes: Number of object classes.
    # dim_feedforward: Size of the feedforward network in transformer layers.
    # nhead: Number of attention heads.
    # dropout: Dropout rate.
    # activation: Activation function (e.g., ReLU).
    # scale_clamp: Maximum allowed scaling factor for bounding boxes.
    # bbox_weights: Weights for bounding box regression.
    def __init__(self, cfg, d_model, num_classes, dim_feedforward=2048, nhead=8, dropout=0.1, activation="relu",
                 scale_clamp: float = _DEFAULT_SCALE_CLAMP, bbox_weights=(2.0, 2.0, 1.0, 1.0)):
        """
        Model Dimensions and Hyperparameters:
        Initializes key model parameters like d_model (hidden dimension), dim_feedforward, number of attention heads (nhead), dropout rate, activation function, and others based on the configuration (cfg).

        Self-Attention Layer:
        Creates a multi-head self-attention layer (self.self_attn) to model relationships between object proposals, helping refine their features iteratively.

        Dynamic Convolution Layer:
        Initializes a dynamic convolution layer (self.inst_interact) to further refine features by interacting with the feature maps, providing more detailed object representation.

        Feedforward Neural Network:
        Defines a two-layer feedforward network (linear1 and linear2) to further process features after attention and dynamic convolution steps, with dropout and layer normalization applied.

        Normalization and Dropout Layers:
        Initializes Layer Normalization (norm1, norm2, norm3) and Dropout layers for regularization, applied at different stages of feature refinement to prevent overfitting and stabilize learning.

        Activation Function:
        Selects the activation function (ReLU, SiLU, etc.) using the helper function _get_activation_fn, which is applied after the linear layers.

        Block Time MLP:
        Defines a block time multi-layer perceptron (MLP) (block_time_mlp) that processes temporal embeddings (e.g., time steps for denoising) and adjusts feature scaling and shifting at each iteration.

        Classification Module:
        Initializes the classification head (cls_module) consisting of a series of linear layers, normalization, and activation functions, designed to predict the object class labels for each proposal.

        Regression Module:
        Initializes the regression head (reg_module) that predicts bounding box deltas (adjustments to the initial bounding boxes) after feature refinement.

        Final Prediction Layers:
        Defines the final classification logits (class_logits) and bounding box deltas (bboxes_delta) layers for output predictions.

        self.d_model: The hidden dimension size of the model's feature representation.
        self.self_attn: Multi-head attention layer for refining features by modeling relationships between proposals.
        self.inst_interact:	Dynamic convolution layer to further refine features based on object-specific interactions.
        self.linear1, self.linear2:	Feedforward layers for processing refined features.
        self.norm1, self.norm2, self.norm3:	Layer normalization layers to stabilize the training process.
        self.dropout1, self.dropout2, self.dropout3: Dropout layers for regularization, applied at different points in the model.
        self.activation: Activation function applied after the linear transformations.
        self.block_time_mlp: MLP that adjusts features based on temporal (denoising) embeddings.
        self.cls_module: Classification module consisting of layers for predicting object classes.
        self.reg_module: Regression module consisting of layers for predicting bounding box deltas.
        self.class_logits: Final layer for computing class logits (classification scores) for each proposal.
        self.bboxes_delta: Final layer for computing bounding box deltas (adjustments to initial bounding boxes).
        """

        pass

    # Inputs:
    # features: Input feature maps.
    # bboxes: Initial bounding boxes.
    # pro_features: Proposal features (may be None initially).
    # pooler: RoI pooling module.
    # time_emb: Temporal embeddings from the diffusion process.
    def forward(self, features, bboxes, pro_features, pooler, time_emb):
        # Converts bounding boxes into Boxes objects for RoI pooling.
        N, nr_boxes = bboxes.shape[:2]
        # roi_feature.
        proposal_boxes = list()
        for b in range(N):
            proposal_boxes.append(Boxes(bboxes[b]))
        # roi_features: Extracted features from RoI pooling.
        roi_features = pooler(features, proposal_boxes)

        # Initializes pro_features using RoI features if not provided.
        if pro_features is None:
            pro_features = roi_features.view(N, nr_boxes, self.d_model, -1).mean(-1)

        # Applies self-attention to the proposal features
        # Normalizes and updates with residual connection. (ResNet)
        roi_features = roi_features.view(N * nr_boxes, self.d_model, -1).permute(2, 0, 1)
        pro_features = pro_features.view(N, nr_boxes, self.d_model).permute(1, 0, 2)

        """
        Iterative Refinement Through Attention and Convolution
        
        Each forward pass of the RCNNHead involves:
        - Self-Attention (self_attn): Models relationships between proposals, which helps separate noisy proposals from valid ones.
        - Dynamic Convolution (inst_interact): Refines proposal features by interacting with localized regions in the feature map.
        
        This iterative mechanism ensures that the model gradually denoises the proposals by incorporating contextual and localized information.
        """
        pro_features2 = self.self_attn(pro_features, pro_features, value=pro_features)[0]
        pro_features = pro_features + self.dropout1(pro_features2)
        pro_features = self.norm1(pro_features)

        # inst_interact: Applies dynamic convolution for instance interaction.
        pro_features = pro_features.view(nr_boxes, N, self.d_model).permute(1, 0, 2).reshape(1, N * nr_boxes,
                                                                                             self.d_model)
        pro_features2 = self.inst_interact(pro_features, roi_features)
        pro_features = pro_features + self.dropout2(pro_features2)
        # Normalizes the updated proposal features.
        obj_features = self.norm2(pro_features)

        # Refines object features using a feedforward network with residual connections.
        obj_features2 = self.linear2(self.dropout(self.activation(self.linear1(obj_features))))
        obj_features = obj_features + self.dropout3(obj_features2)
        obj_features = self.norm3(obj_features)

        fc_feature = obj_features.transpose(0, 1).reshape(N * nr_boxes, -1)

        # Scales and shifts features using the time embedding.
        scale_shift = self.block_time_mlp(time_emb)
        scale_shift = torch.repeat_interleave(scale_shift, nr_boxes, dim=0)
        scale, shift = scale_shift.chunk(2, dim=1)
        fc_feature = fc_feature * (scale + 1) + shift

        # Refines features separately for classification (cls_feature) and regression (reg_feature).
        cls_feature = fc_feature.clone()
        reg_feature = fc_feature.clone()
        for cls_layer in self.cls_module:
            cls_feature = cls_layer(cls_feature)
        for reg_layer in self.reg_module:
            reg_feature = reg_layer(reg_feature)

        # class_logits: Class scores.
        class_logits = self.class_logits(cls_feature)
        # bboxes_deltas: Bounding box deltas.
        bboxes_deltas = self.bboxes_delta(reg_feature)

        # Applies bounding box deltas to transform initial boxes.
        pred_bboxes = self.apply_deltas(bboxes_deltas, bboxes.view(-1, 4))
        # Returns: Class logits, Predicted bounding boxes, Object features.
        return class_logits.view(N, nr_boxes, -1), pred_bboxes.view(N, nr_boxes, -1), obj_features

    def apply_deltas(self, deltas, boxes):
        """
        Handles bounding box transformation using predicted deltas (dx, dy, dw, dh).

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """

        pass


class DiffusionDet(nn.Module):
    # Arguments:
    # cfg: Configuration dictionary.
    def __init__(self, cfg):
        """
        Device and Configuration:
        Sets the device and retrieves model configurations (e.g., features, classes, proposals, hidden dimensions).

        Backbone:
        Builds the backbone network and sets its size_divisibility for processing input images.

        Diffusion Process:
        Initializes parameters (e.g., betas, alphas) for diffusion modeling.
        Precomputes and buffers constants for the forward diffusion process.
        Configures self-conditioning, scaling, box renewal, and ensemble options.
        Dynamic Head: Creates a DynamicHead module for object detection using backbone outputs.

        Loss Functions:
        Defines weights for classification, GIoU, L1 losses, and no-object penalty.
        Sets up SetCriterionDynamicK with a HungarianMatcherDynamicK for matching and loss computation.
        Supports deep supervision and optional focal or federated losses.
        Image Normalization: Prepares normalization parameters for input images (mean and standard deviation).

        Device Transfer:
        Transfers all components to the specified device for computation.

        Attributes:

        Model Configuration:
        num_classes: Number of object classes.
        num_proposals: Number of proposals (queries) for detection.
        hidden_dim: Dimension of hidden states in the dynamic head.
        num_heads: Number of attention heads.

        Backbone:
        backbone: Backbone network for feature extraction.
        size_divisibility: Input size divisibility constraint.

        Diffusion Process:
        num_timesteps: Number of timesteps in the diffusion process.
        betas, alphas: Diffusion process parameters.
        alphas_cumprod: Precomputed cumulative product of alphas.
        sqrt_alphas_cumprod: Square root of cumulative product of alphas for forward diffusion.
        sqrt_one_minus_alphas_cumprod: Square root of one minus cumulative product of alphas for forward diffusion.
        posterior_mean_coef1, posterior_mean_coef2: Coefficients for reverse diffusion computation.
        posterior_variance: Variance for reverse diffusion.
        posterior_log_variance_clipped: Clipped logarithm of posterior variance.

        Diffusion Settings:
        self_condition: Enables self-conditioning in diffusion.
        scale: Scaling factor for diffusion noise.
        renew_box: Flag to renew box states during training.
        ensemble_score: Enables ensemble scoring during inference.

        Dynamic Head:
        head: Instance of DynamicHead for object detection.
        deep_supervision: Whether to compute intermediate losses.

        Loss Components:
        matcher: Instance of HungarianMatcherDynamicK for matching predictions to ground truth.
        criterion: Instance of SetCriterionDynamicK for loss computation.
        class_weight, giou_weight: Weights for classification and generalized IoU loss.
        l1_weight, no_object_weight: Weights for L1 loss and no-object penalty.

        Image Normalization:
        pixel_mean: Mean values for input image normalization.
        pixel_std: Standard deviation values for normalization.

        Device Management:
        device: Device (CPU/GPU) for model execution.
        """

        pass

    def predict_noise_from_start(self, x_t, t, x0):
        """
        Input: Accepts the noisy input x_t, timestep t, and optionally x_0 (original image state).
        Self-Conditioning:.
         - If `self_condition` is enabled:\hat{x}_0 = x_0\) if provided, otherwise, predicts x_0 first.
         - If disabled:\hat{x}_0 = None\).
        Forward Pass: Uses x_t, \hat{x}_0, and t to compute the predicted noise through the model.
        Output: Returns the predicted noise for timestep t.

        This method predicts the noise present at a given diffusion step t, enabling the model to iteratively reconstruct the clean data x_0 from the noisy states during reverse diffusion.
        """
        pass

    def model_predictions(self, backbone_feats, images_whwh, x, t, x_self_cond=None, clip_x_start=False):
        """
        Input: Takes noisy input x_t, timestep t, and optional clip_x_start flag.                
        Predictions:
        Runs the model to compute:                                                                          
         1. Predicted noise \epsilon_\theta: noise at timestep t.                           
         2. Predicted clean data x_0: estimated clean image.                                       
        Clipping: If `clip_x_start` is `True`, clips x_0 to the range[-1, 1]\).                              
        Outputs:                                                                                           
         - \epsilon_\theta (predicted noise).                                                         
         - x_0 (predicted clean data).                                                                

        This method provides both the predicted noise and clean data, which are essential for calculating loss functions and refining the reconstruction during reverse diffusion.
        """
        pass

    # @torch.no_grad(): This decorator ensures that no gradients are computed during the execution of this function.
    # It is used because the function is part of inference (prediction) and does not need to update model parameters.
    @torch.no_grad()
    def ddim_sample(self, batched_inputs, backbone_feats, images_whwh, images, clip_denoised=True, do_postprocess=True):
        """
        Arguments:
        batched_inputs: Inputs for a batch of images.
        backbone_feats: Features extracted by the model’s backbone (likely CNN).
        images_whwh: A tensor representing the width and height of each image.
        images: The input images.
        clip_denoised: A flag to control whether to clip the denoised images (i.e., refined bounding boxes).
        do_postprocess: A flag to control whether post-processing should be performed on the outputs.
        """

        # batch: The batch size (number of images in the batch).
        batch = images_whwh.shape[0]
        # shape: The shape of the generated tensor for bounding boxes, (batch_size, num_proposals, 4) (the 4 corresponds to the (x1, y1, x2, y2) bounding box coordinates).
        shape = (batch, self.num_proposals, 4)
        # total_timesteps: Total number of diffusion steps.
        # sampling_timesteps: Number of steps used during sampling (could be fewer than total_timesteps for faster inference).
        # eta: A parameter controlling the amount of noise added at each step.
        # objective: The objective function for the diffusion process (in this case, it’s likely predicting x0).
        total_timesteps, sampling_timesteps, eta, objective = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        #  Creates the time steps for the diffusion process. It generates a linearly spaced tensor (times) from -1 to
        #  total_timesteps - 1 and then reverses it to go from the last time step to the first.
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        # time_pairs creates a list of adjacent time step pairs, like (T-1, T-2) and (T-2, T-3), which are used to update the generated image at each step.
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        # Initializes a tensor img filled with random values (representing random bounding boxes) that will be iteratively refined during the diffusion process.
        img = torch.randn(shape, device=self.device)

        # Initializes lists to store the ensemble scores, labels, and coordinates if the model is using an ensemble approach (multiple predictions to combine).
        ensemble_score, ensemble_label, ensemble_coord = [], [], []
        x_start = None
        # Iterates through each time step pair
        for time, time_next in time_pairs:
            # Creates a tensor time_cond representing the current time step for each image in the batch.
            time_cond = torch.full((batch,), time, device=self.device, dtype=torch.long)
            # Sets self_cond to x_start if self-conditioning is enabled, meaning the model refines its previous predictions. Otherwise, self_cond is None.
            self_cond = x_start if self.self_condition else None

            # Calls model_predictions to get the model’s predictions for the current time step, which include:
            # pred_noise: The predicted noise at the current step.
            # x_start: The predicted bounding boxes (coordinates) for the start of the diffusion process.
            preds, outputs_class, outputs_coord = self.model_predictions(backbone_feats, images_whwh, img, time_cond,
                                                                         self_cond, clip_x_start=clip_denoised)
            pred_noise, x_start = preds.pred_noise, preds.pred_x_start

            # If box renewal is enabled:
            # The model filters out boxes with low confidence scores by applying a threshold (0.5).
            # The model keeps only the boxes that exceed the threshold, and the noise and bounding box predictions are updated accordingly.
            if self.box_renewal:  # filter
                score_per_image, box_per_image = outputs_class[-1][0], outputs_coord[-1][0]
                threshold = 0.5
                score_per_image = torch.sigmoid(score_per_image)
                value, _ = torch.max(score_per_image, -1, keepdim=False)
                keep_idx = value > threshold
                num_remain = torch.sum(keep_idx)

                pred_noise = pred_noise[:, keep_idx, :]
                x_start = x_start[:, keep_idx, :]
                img = img[:, keep_idx, :]

            # If the next time step is less than zero (i.e., we are at the final step), it sets the generated image (img) to the final bounding box predictions (x_start) and continues.
            if time_next < 0:
                img = x_start
                continue

            # Computes values for the diffusion process based on the current and next time steps
            # alpha and alpha_next are values that control the scaling of the image at each step.
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            # sigma controls the amount of noise added at each step.
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            # c is another scaling factor that controls the contribution of the noise.
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            # Refines the generated image (img) by adding noise (pred_noise and noise) scaled by alpha_next and sigma.
            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            # If box renewal is enabled, it replenishes the proposals with new random boxes to ensure the number of proposals remains constant.
            if self.box_renewal:  # filter
                # replenish with randn boxes
                img = torch.cat((img, torch.randn(1, self.num_proposals - num_remain, 4, device=img.device)), dim=1)

            # If using an ensemble approach, the model performs inference at the current time step and stores the predicted boxes, scores, and labels in the ensemble lists.
            if self.use_ensemble and self.sampling_timesteps > 1:
                box_pred_per_image, scores_per_image, labels_per_image = self.inference(outputs_class[-1],
                                                                                        outputs_coord[-1],
                                                                                        images.image_sizes)
                ensemble_score.append(scores_per_image)
                ensemble_label.append(labels_per_image)
                ensemble_coord.append(box_pred_per_image)

        # If using an ensemble and multiple sampling timesteps, it combines the results from all ensemble members,
        # applies non-maximum suppression (NMS) to filter out redundant boxes, and stores the final predictions.
        if self.use_ensemble and self.sampling_timesteps > 1:
            box_pred_per_image = torch.cat(ensemble_coord, dim=0)
            scores_per_image = torch.cat(ensemble_score, dim=0)
            labels_per_image = torch.cat(ensemble_label, dim=0)
            if self.use_nms:
                keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.5)
                box_pred_per_image = box_pred_per_image[keep]
                scores_per_image = scores_per_image[keep]
                labels_per_image = labels_per_image[keep]

            result = Instances(images.image_sizes[0])
            result.pred_boxes = Boxes(box_pred_per_image)
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results = [result]
        else:
            # If not using an ensemble, the final predictions are made from the last outputs of the model.
            # The inference function is called to get the final bounding box predictions.
            output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            results = self.inference(box_cls, box_pred, images.image_sizes)
        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    # forward diffusion
    def q_sample(self, x_start, t, noise=None):
        """
        The `q_sample` method in DiffusionDet adds noise to a given sample x_0 to simulate its state at a specific diffusion timestep t. Here's a summary:

        Input: Takes the clean sample x_0, timestep t, and optional noise \epsilon. 
        Noise Initialization: If \epsilon (noise) is not provided, it is randomly sampled from a Gaussian distribution. 
        Diffusion Process: Computes x_t, the noisy version of x_0, using the formula:            
        x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon. 
        Output: x_t, the noisy sample at timestep t.                        

        The `q_sample` method simulates the forward diffusion process, gradually adding noise to x_0 according to a predefined noise schedule. This is essential for training diffusion models, as it provides noisy data for the model to learn the denoising process.
        """
        pass

    def forward(self, batched_inputs, do_postprocess=True):
        """
        Inputs:
        batched_inputs: A list of dictionaries, each representing an image and its associated data (e.g., instances, height, width).
        do_postprocess: A flag to indicate if post-processing should be applied to the outputs.
        """
        # Prepares the input images for the backbone by normalizing them and handling their spatial dimensions.
        # images contains the processed image tensor, and images_whwh contains their width/height information.
        images, images_whwh = self.preprocess_image(batched_inputs)
        if isinstance(images, (list, torch.Tensor)):
            images = nested_tensor_from_tensor_list(images)

        # Feature Extraction.
        # Passes the preprocessed images through the backbone network to extract feature maps.
        # self.in_features defines which feature maps to use from the backbone.
        src = self.backbone(images.tensor)
        features = list()
        for f in self.in_features:
            feature = src[f]
            features.append(feature)

        # Prepare Proposals.
        # If the model is in inference mode (not self.training), it performs sampling using the DDIM (Denoising Diffusion Implicit Models) approach,
        # which generates predictions from the features.
        if not self.training:
            results = self.ddim_sample(batched_inputs, features, images_whwh, images)
            return results

        if self.training:
            # Ground Truth (GT) Preparation: Converts ground-truth annotations to the device and normalizes bounding boxes relative to image dimensions.
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            # Calls self.prepare_targets to generate diffusion targets, bounding boxes, noise, and diffusion time steps t.
            targets, x_boxes, noises, t = self.prepare_targets(gt_instances)
            t = t.squeeze(-1)
            x_boxes = x_boxes * images_whwh[:, None, :]

            # Uses the DynamicHead module to predict:
            # outputs_class: Classification scores for each proposal.
            # outputs_coord: Coordinates for each bounding box.
            outputs_class, outputs_coord = self.head(features, x_boxes, t, None)
            # Stores the final predictions in output.
            output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

            if self.deep_supervision:
                output['aux_outputs'] = [{'pred_logits': a, 'pred_boxes': b}
                                         for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

            # Computes loss using the SetCriterionDynamicK object, which handles classification, bounding box regression, and other losses.
            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            # Adjusts the losses with the respective weights and returns the loss dictionary.
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict

    def prepare_diffusion_concat(self, gt_boxes):
        """
        The prepare_diffusion_concat function generates diffusion-based augmented bounding boxes by combining the ground
        truth boxes (gt_boxes) with randomly generated placeholders or selecting a subset if there are too many ground truth boxes.

        Inputs:
        gt_boxes: A tensor containing normalized ground truth bounding boxes in the format (cx, cy, w, h).

        Initialize Noise and Time:
        - Generates a random diffusion timestep t.
        - Creates random noise for diffusion with the same dimensions as the target number of proposals.

        Handle Ground Truth Box Cases:
        - If there are no ground truth boxes, create a single default box at the image center with dimensions (0.5, 0.5, 1.0, 1.0).
        - If the number of ground truth boxes is less than the target number of proposals:
        Adds placeholder boxes sampled from a Gaussian distribution for the remaining proposals.
        Ensures minimum box dimensions to prevent zero-area boxes.
        - If there are more ground truth boxes than the target proposals:
        - Randomly selects a subset of boxes to fit the target number.

        Scale and Diffusion:
        - Normalizes all boxes to fit a range of [-scale, scale] using the self.scale parameter.
        - Applies forward diffusion using the q_sample method, which adds Gaussian noise to the normalized boxes based on the diffusion timestep.

        Post-Diffusion Adjustments:
        - Clamps the diffused box values to be within the valid range of [-scale, scale].
        - Rescales the boxes back to the range [0, 1] for compatibility with subsequent operations.
        - Converts the boxes from (cx, cy, w, h) format to (x1, y1, x2, y2) format.

        Outputs:
        - diff_boxes: Diffused bounding boxes in (x1, y1, x2, y2) format.
        - noise: The noise applied during the diffusion process.
        - t: The randomly chosen diffusion timestep.

        Conclusion:
        The function generates a set of proposals for training by combining ground truth data with randomized noise.
        It ensures a consistent number of proposals and introduces variability through the diffusion process,
        aiding robust training for object detection.
        """
        pass

    def prepare_targets(self, targets):
        """
        This function takes a list of ground-truth targets (targets), each corresponding to an image, and prepares them
        for the model's loss computation.

        new_targets: Stores the processed ground-truth information for each image.
        diffused_boxes: Stores boxes generated by adding noise to ground-truth boxes.
        noises: Stores the noise applied to the ground-truth boxes.
        ts: Stores the diffusion timesteps used for each image.

        Iterates over the list of ground-truth targets, each corresponding to one image.
        Extracts the height (h) and width (w) of the image.
        Converts the image size to a tensor image_size_xyxy for normalization purposes.
        Retrieves ground-truth class labels (gt_classes) and bounding boxes (gt_boxes).
        Normalizes bounding box coordinates by dividing by the image size, then converts them from [x1, y1, x2, y2] format to [cx, cy, w, h] (center-based representation).
        Calls prepare_diffusion_concat, which:
            Adds noise to the ground-truth boxes.
            Chooses a random diffusion timestep t.
            Ensures the result aligns with the number of proposals.
            d_boxes: Diffused boxes (with noise added).
            d_noise: Noise applied to the boxes.
            d_t: Chosen diffusion timestep.

        For each image:
            Creates a dictionary target containing:
                labels: Ground-truth class labels.
                boxes: Normalized ground-truth boxes in [cx, cy, w, h] format.
                boxes_xyxy: Ground-truth boxes in original [x1, y1, x2, y2] format.
                image_size_xyxy: Image dimensions as a tensor.
                image_size_xyxy_tgt: Repeated dimensions for compatibility with proposals.
                area: Area of each ground-truth box.
            Adds the processed target to new_targets.

        Outputs:
            new_targets: List of dictionaries with processed ground-truth information.
            diffused_boxes: A tensor stack of diffused boxes for all images.
            noises: A tensor stack of the applied noise.
            ts: A tensor stack of the diffusion timesteps.
        """
        pass

    def inference(self, box_cls, box_pred, image_sizes):
        """
        Arguments:
        box_cls: This tensor contains the classification scores for each bounding box proposal. Its shape is (batch_size, num_proposals, K), where K is the number of classes (including background).
        box_pred: This tensor contains the predicted bounding box coordinates (in cx, cy, w, h format). Its shape is (batch_size, num_proposals, 4).
        image_sizes: A list containing the original sizes of the images in the batch.

        The function aims to perform inference (i.e., make predictions for bounding boxes and their corresponding class labels) based on the classification scores (box_cls) and predicted bounding boxes (box_pred).
        """

        # This ensures that the number of images in box_cls matches the number of images in image_sizes. This is a sanity check to avoid mismatched batch sizes.
        assert len(box_cls) == len(image_sizes)
        results = []

        # This condition checks whether focal loss or FED loss is being used. These losses require special handling for the predicted scores.
        if self.use_focal or self.use_fed_loss:
            # Applies the sigmoid function to the box_cls tensor. This transforms the raw class logits into probabilities (in the range [0, 1]).
            scores = torch.sigmoid(box_cls)

            # Creates a tensor labels with shape (num_proposals * num_classes) that contains the class labels for each bounding box proposal.
            # torch.arange(self.num_classes) generates a tensor of class indices, which is then repeated to match the number of proposals (num_proposals).
            labels = torch.arange(self.num_classes, device=self.device). \
                unsqueeze(0).repeat(self.num_proposals, 1).flatten(0, 1)

            # Loops through each image in the batch. scores_per_image, box_pred_per_image, and image_size are the individual class scores, predicted boxes, and image sizes for the i-th image, respectively.
            for i, (scores_per_image, box_pred_per_image, image_size) in enumerate(zip(
                    scores, box_pred, image_sizes
            )):
                # Initializes an empty Instances object for the i-th image. This will store the predicted bounding boxes, scores, and class labels.
                result = Instances(image_size)
                # Flattens the scores_per_image tensor and selects the top num_proposals highest scoring boxes using the topk function. The topk_indices gives the indices of the top scores.
                scores_per_image, topk_indices = scores_per_image.flatten(0, 1).topk(self.num_proposals, sorted=False)
                # Selects the corresponding class labels for the top num_proposals highest scoring boxes.
                labels_per_image = labels[topk_indices]

                # This adjusts the predicted bounding boxes (box_pred_per_image) to match the top-k selected boxes.
                # It reshapes box_pred_per_image to have a shape of (batch_size * num_proposals, 1, 4) and repeats it for each class, resulting in a shape of (batch_size * num_proposals * num_classes, 4).
                # This allows each bounding box proposal to be repeated for each class and then selects the topk boxes accordingly.
                box_pred_per_image = box_pred_per_image.view(-1, 1, 4).repeat(1, self.num_classes, 1).view(-1, 4)
                box_pred_per_image = box_pred_per_image[topk_indices]

                # If ensemble learning is used and the sampling timesteps are greater than 1, the function returns the predicted boxes, scores, and labels directly without performing Non-Maximum Suppression (NMS).
                if self.use_ensemble and self.sampling_timesteps > 1:
                    return box_pred_per_image, scores_per_image, labels_per_image

                # If NMS is enabled (self.use_nms), it applies the batched_nms function to filter out overlapping boxes based on their Intersection over Union (IoU). The 0.5 threshold indicates that boxes with IoU > 0.5 are considered overlapping.
                # After NMS, only the kept boxes (keep) are retained for further processing.
                if self.use_nms:
                    keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.5)
                    box_pred_per_image = box_pred_per_image[keep]
                    scores_per_image = scores_per_image[keep]
                    labels_per_image = labels_per_image[keep]

                # Assigns the filtered bounding boxes (box_pred_per_image), scores (scores_per_image), and class labels (labels_per_image) to the result object.
                result.pred_boxes = Boxes(box_pred_per_image)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                # Appends the result object (which contains the final predictions for the i-th image) to the results list.
                results.append(result)

        else:
            # For each box we assign the best class or the second best if the best on is `no_object`.
            scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

            for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(
                    scores, labels, box_pred, image_sizes
            )):
                if self.use_ensemble and self.sampling_timesteps > 1:
                    return box_pred_per_image, scores_per_image, labels_per_image

                if self.use_nms:
                    keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.5)
                    box_pred_per_image = box_pred_per_image[keep]
                    scores_per_image = scores_per_image[keep]
                    labels_per_image = labels_per_image[keep]

                result = Instances(image_size)
                result.pred_boxes = Boxes(box_pred_per_image)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                results.append(result)

        # Returns the results list, which contains the inference results for all images in the batch.
        return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        pass


def set_prediction_loss(pb_pred, gt_boxes):
    """
    
    code taken from: loss.py from class SetCriterionDynamicK and HungarianMatcherDynamicK.
    
    pb_pred: [B, N, 4]
    gt_boxes: [B, *, 4]
    """

    """ This class computes the loss for DiffusionDet.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    pass


def train_loss(images, gt_boxes):
    """
    images: [B, H, W, 3]
    gt_boxes: [B, *, 4]
    # B: batch
    # N: number of proposal boxes
    """
    scale = 2.0  # signal scaling
    T = 1000  # number of time steps
    N = 300  # number of proposal boxes
    alphas = alpha_cumprod(T)

    B = images.shape[0]  # batch size

    # Encode image features
    feats = image_encoder(images)

    # Pad gt_boxes to N
    pb = pad_boxes(gt_boxes)  # padded boxes: [B, N, 4]

    # Signal scaling
    pb = (pb * 2 - 1) * scale

    # Corrupt gt_boxes
    t = randint(0, T)  # time step

    eps = torch.randn(B, N, 4)  # noise: [B, N, 4]
    pb_crpt = sqrt(alphas[t]) * pb + sqrt(1 - alphas[t]) * eps  # corrupted boxes: [B, N, 4]

    pb_crpt = torch.clamp(pb_crpt, min=-1 * scale, max=scale)  # clip corrupted boxes to [-scale, scale]
    pb_crpt = ((pb_crpt / scale) + 1) / 2.  # normalize corrupted boxes to [0, 1]

    # Predict
    pb_pred = detection_decoder(pb_crpt, feats, t)  # predicted boxes: [B, N, 4]

    # Set prediction loss
    loss = set_prediction_loss(pb_pred, gt_boxes)  # prediction loss

    return loss


def ddim_step(pb_t, pb_pred, t_now, t_next, timesteps=1000):
    """
    code taken from: detector.py in ddim_sample(...) function.
    """

    alphas = alpha_cumprod(timesteps)

    alpha = alphas[t_now]
    alpha_next = alphas[t_next]

    sigma = 1 * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
    c = (1 - alpha_next - sigma  2).sqrt()

    noise = torch.randn_like(pb_t)

    pred_noise = predict_noise_from_start(pb_pred, t_next, pb_t)

    img = pb_pred * alpha_next.sqrt() + \
          c * pred_noise + \
          sigma * noise

    return img


def predict_noise_from_start(x_t, t, x0, timesteps=1000):
    """
    code taken from: detector.py in predict_noise_from_start(...) function.
    """
    alphas = alpha_cumprod(timesteps)

    return (
            (sqrt(1 / alphas[t]) * x_t - x0) / sqrt(1 / alphas[t] - 1)
    )


def box_renewal(pb_t):
    """
    code taken from: detector.py in ddim_sample(...) function.
    
    pb_t: [B, N, 4]
    """
    return pb_t


def infer(images, steps, T):
    """
    images: [B, H, W, 3]
    # steps: number of sample steps
    # T: number of time steps
    """

    T = 1000  # number of time steps
    N = 300  # number of proposal boxes

    B = images.shape[0]  # batch size

    # Encode image features
    feats = image_encoder(images)

    # noisy boxes: [B, N, 4]
    pb_t = torch.randn(B, N, 4)

    # uniform sample step size
    times = reversed(linspace(-1, T, steps))

    # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
    time_pairs = list(zip(times[:-1], times[1:]))

    for t_now, t_next in zip(time_pairs):
        # Predict pb_0 from pb_t
        pb_pred = detection_decoder(pb_t, feats, t_now)

        # Estimate pb_t at t_next
        pb_t = ddim_step(pb_t, pb_pred, t_now, t_next)

        # Box renewal
        pb_t = box_renewal(pb_t)

    return pb_pred
