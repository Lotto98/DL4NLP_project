
from math import sqrt, pi
import math
from random import randint
import random
from numpy import extract, linspace
import torch


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
- detection_decoder: this is the actual forward function of the model "head". NEED TO ENRICH IT.
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
    
    gt_boxes: [B, *, 4]
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
    
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def alpha_cumprod(timesteps):
    """
    code taken from: detector.py in __init__(...) function under #build diffusion comment.
    
    t: time step
    """
    # alpha_cumprod
    betas = cosine_beta_schedule(timesteps)
    alphas = 1. - betas
    return torch.cumprod(alphas, dim=0)

class SinusoidalPositionEmbeddings(torch.nn.Module):
    
    """
    code taken from: head.py in SinusoidalPositionEmbeddings(...) class.
    """
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

def detection_decoder(pb_crpt, feats, t):
    """
    THIS THING SHOULD BE A CLASS WITH FORWARD METHOD. THIS IS JUST A FORWARD FUNCTION FOR NOW.
    
    code taken from: head.py in forward(...) and __init__ functions.
    
    pb_crpt: [B, N, 4]
    feats: [B, H, W, C]
    t: time step
    """
    
    d_model = d_model #hidden dimension
    time_dim = d_model * 4
    time_mlp = torch.nn.Sequential(
        SinusoidalPositionEmbeddings(d_model),
        torch.nn.Linear(d_model, time_dim),
        torch.nn.GELU(),
        torch.nn.Linear(time_dim, time_dim),
    )
    
    # assert t shape (batch_size)
    time = time_mlp(t)

    bs = len(feats[0])
    bboxes = pb_crpt
    num_boxes = bboxes.shape[1]

    
    proposal_features = None
    box_pooler = None #????????
    head_series = RCNNHead(".....") # list of head modules ???? Need to understand it.
    
    for head_idx, rcnn_head in enumerate(head_series):
        class_logits, pred_bboxes, proposal_features = rcnn_head(feats, bboxes, proposal_features, box_pooler, time)
        bboxes = pred_bboxes.detach()

    return pred_bboxes[None]

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
    scale = 2.0 # signal scaling
    T=1000 # number of time steps
    N = 300 # number of proposal boxes
    alphas = alpha_cumprod(T)
    
    
    B = images.shape[0] # batch size
    
    # Encode image features
    feats = image_encoder(images)
    
    # Pad gt_boxes to N
    pb = pad_boxes(gt_boxes) # padded boxes: [B, N, 4]
    
    # Signal scaling
    pb = (pb * 2 - 1) * scale
    
    # Corrupt gt_boxes
    t = randint(0, T)# time step
    
    eps = torch.randn(B, N, 4) # noise: [B, N, 4]
    pb_crpt = sqrt( alphas[t] ) * pb + sqrt(1 - alphas[t] ) * eps # corrupted boxes: [B, N, 4]
    
    pb_crpt = torch.clamp(pb_crpt, min=-1 * scale, max=scale) # clip corrupted boxes to [-scale, scale]
    pb_crpt = ((pb_crpt / scale) + 1) / 2. # normalize corrupted boxes to [0, 1]
    
    # Predict
    pb_pred = detection_decoder(pb_crpt, feats, t) # predicted boxes: [B, N, 4]
    
    # Set prediction loss
    loss = set_prediction_loss(pb_pred, gt_boxes) # prediction loss
    
    return loss


def ddim_step(pb_t, pb_pred, t_now, t_next, timesteps=1000):
    
    """
    code taken from: detector.py in ddim_sample(...) function.
    """
    
    alphas = alpha_cumprod(timesteps)
    
    alpha = alphas[t_now]
    alpha_next = alphas[t_next]

    sigma = 1 * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
    c = (1 - alpha_next - sigma ** 2).sqrt()
    
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
            (sqrt(1/ alphas[t]) * x_t - x0) / sqrt( 1/ alphas[t]-1)
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
    
    T=1000 # number of time steps
    N = 300 # number of proposal boxes
    
    B = images.shape[0] # batch size
    
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