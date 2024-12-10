import torch
import random
import torch.nn.functional as F
import torchvision.ops.boxes as box_ops

from math import sqrt
import math

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

def batched_nms(
    boxes: torch.Tensor, scores: torch.Tensor, idxs: torch.Tensor, iou_threshold: float
):
    """
    TO MODIFY: need to pass a dummy x2, y2 in the boxes tensor
    """
    assert boxes.shape[-1] == 4
    # Note: Torchvision already has a strategy (https://github.com/pytorch/vision/issues/1311)
    # to decide whether to use coordinate trick or for loop to implement batched_nms. So we
    # just call it directly.
    # Fp16 does not have enough range for batched NMS, so adding float().
    return box_ops.batched_nms(boxes.float(), scores, idxs, iou_threshold)

class DiffusionDet_audio(torch.nn.Module):
    
    def pad_boxes(self, gt_boxes):
        """
        Code taken from: detector.py in prepare_diffusion_concat(...) function.
        
        Pads the ground truth boxes to a specified number of proposals.

        Args:
            gt_boxes (Tensor): Ground truth boxes with shape [B, *, 2].
            num_proposals (int): Number of proposals to pad to. Default is 300.

        Returns:
            Tensor: Padded boxes with shape [B, num_proposals, 2].
        """
        # Pad gt_boxes to N
        num_gt = gt_boxes.shape[0]
        if not num_gt:  # generate fake gt boxes if empty gt boxes
            gt_boxes = torch.as_tensor([[0.5, 1.]], dtype=torch.float, device="cuda")
            num_gt = 1

        if num_gt < self.num_proposals:
            box_placeholder = torch.randn(self.num_proposals - num_gt, 2,
                                        device="cuda") / 6. + 0.5  # 3sigma = 1/2 --> sigma: 1/6
            box_placeholder[:, 1:] = torch.clip(box_placeholder[:, 1:], min=1e-4)
            x_start = torch.cat((gt_boxes, box_placeholder), dim=0)
        elif num_gt > self.num_proposals:
            select_mask = [True] * self.num_proposals + [False] * (num_gt - self.num_proposals)
            random.shuffle(select_mask)
            x_start = gt_boxes[select_mask]
        else:
            x_start = gt_boxes
            
        x_start = (x_start * 2. - 1.) * self.scale

        return x_start
    
    def corrupted_boxes_generator(self, gt_boxes ):
        
        t = torch.randint(0, self.num_timesteps, (1,), device=self.device).long() # sample of timesteps
        noise = torch.randn(self.num_proposals, 4, device=self.device) # gaussian noise
        
        # noise sample
        pb_crpt = extract(self.sqrt_alphas_cumprod, t, gt_boxes.shape) * gt_boxes \
                    + extract(self.sqrt_one_minus_alphas_cumprod, t, gt_boxes.shape) * noise

        pb_crpt = torch.clamp(pb_crpt, min=-1 * self.scale, max=self.scale)  # clip corrupted boxes to [-scale, scale]
        pb_crpt = ((pb_crpt / self.scale) + 1) / 2.  # normalize corrupted boxes
        
        return pb_crpt, t
    
    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    def ddim_step_with_box_renewal(self, boxes_t, output_detection_decoder, t_now, t_next):
        
        predicted_class, predicted_boxes = output_detection_decoder
        
        boxes_start = predicted_boxes[-1]
        
        predicted_noise = self.predict_noise_from_start(boxes_t, t_now, boxes_start)
        
        if self.box_renewal:
            score_per_audio = predicted_class[-1][0]
            threshold = 0.5
            score_per_audio = torch.sigmoid(score_per_audio)
            value, _ = torch.max(score_per_audio, -1, keepdim=False)
            keep_idx = value > threshold
            num_remain = torch.sum(keep_idx)

            predicted_noise = predicted_noise[:, keep_idx, :]
            boxes_start = boxes_start[:, keep_idx, :]
            boxes_t = boxes_t[:, keep_idx, :]
        
        alpha = self.alphas_cumprod[t_now]
        alpha_next = self.alphas_cumprod[t_next]

        sigma = self.eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        c = (1 - alpha_next - sigma ** 2).sqrt()

        noise = torch.randn_like(boxes_t)

        boxes_t = boxes_start * alpha_next.sqrt() + \
                c * predicted_noise + \
                sigma * noise
        
        if self.box_renewal:  # filter
            # replenish with randn boxes
            boxes_t = torch.cat((boxes_t, torch.randn(1, self.num_proposals - num_remain, 
                                                2, device=boxes_t.device)), dim=1)
            
        return boxes_t
    
    def postprocessing(self, predicted_class, predicted_boxes):
        
        batched_audio_results = []
        
        # For each box we assign the best class.
        scores, labels = F.softmax(predicted_class, dim=-1)[:, :, :-1].max(-1)

        for i, (scores_per_audio, labels_per_audio, box_pred_per_audio) in enumerate(zip(
                scores, labels, predicted_boxes
        )):
            if self.use_ensemble and self.sampling_timesteps > 1:
                return box_pred_per_audio, scores_per_audio, labels_per_audio

            if self.use_nms:
                keep = batched_nms(box_pred_per_audio, scores_per_audio, labels_per_audio, 0.5)
                box_pred_per_audio = box_pred_per_audio[keep]
                scores_per_audio = scores_per_audio[keep]
                labels_per_audio = labels_per_audio[keep]
                
            singles_audio_result = {}
            singles_audio_result["pred_boxes"] = box_pred_per_audio
            singles_audio_result["scores"] = scores_per_audio
            singles_audio_result["pred_classes"] = labels_per_audio
            
            batched_audio_results.append(singles_audio_result)
    
    def __init__(self, scale=1.0, timesteps=1000, sampling_timesteps=1, num_proposals=300):
        super(DiffusionDet_audio, self).__init__()
        
        self.encoder = torch.nn.Sequential() # Define the encoder -> OpenL3
        
        self.detection_decoder = torch.nn.Sequential()
        
        self.loss = torch.nn.CrossEntropyLoss() #Set prediction loss
        
        self.num_proposals = num_proposals
        
        # build diffusion
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        self.sampling_timesteps = sampling_timesteps
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 1.
        self.self_condition = False
        self.scale = scale
        self.box_renewal = True
        self.use_ensemble = True

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        
        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    def forward(self, audio: torch.Tensor, true_boxes: torch.Tensor):
        
        audio_features = self.encoder(audio) #Extract audio features. TODO: implement OpenL3
        
        if not self.training:
            
            batch_size = audio_features.shape[0]
            
            # noisy boxes: [B, N, 2]
            boxes_t = torch.randn(batch_size, self.num_proposals, 2, device=self.device)
            
            # uniform sample step size
            times = torch.linspace(-1, self.num_timesteps-1, steps = self.sampling_timesteps + 1)
            times = list(reversed(times.int().tolist()))
            time_pairs = list(zip(times[:-1], times[1:]))   # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

            for t_now, t_next in zip(time_pairs):
                
                # Predict pb_0 from pb_t
                output_detection_decoder = self.detection_decoder(boxes_t, audio_features, t_now) #TODO: implement detection decoder: this is self.head(...)

                if t_next >= 0:
                    # Estimate pb_t at t_next only if t_next >= 0
                    boxes_t = self.ddim_step_with_box_renewal(boxes_t, output_detection_decoder, t_now, t_next)
            
            predicted_class, predicted_boxes = output_detection_decoder
            
            #supposing we don't use fed_loss or focal!!!
            self.postprocessing(predicted_class, predicted_boxes)

            return predicted_class, predicted_boxes
        
        if self.training:
            
            true_boxes = self.pad_boxes(true_boxes)
            
            corrupted_boxes, t = self.corrupted_boxes_generator(true_boxes)
            
            output_detection_decoder = self.detection_decoder(corrupted_boxes, audio_features, t) #TODO: implement detection decoder: this is self.head(...)
            
            return self.loss(output_detection_decoder, true_boxes) #TODO: implement loss function