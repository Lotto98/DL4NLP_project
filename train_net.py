# ==========================================
# Modified by Shoufa Chen
# ===========================================
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DiffusionDet Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import os
import itertools
import weakref
from typing import Any, Dict, List, Set
import logging
from collections import OrderedDict
import pandas as pd

import torch
from fvcore.nn.precise_bn import get_bn_modules

import detectron2.utils.comm as comm
from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_batch_data_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch, create_ddp_model, \
    AMPTrainer, SimpleTrainer, hooks, HookBase, default_writers
from detectron2.utils.events import get_event_storage
from detectron2.evaluation import COCOEvaluator, LVISEvaluator, verify_results
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.modeling import build_model

from diffusiondet import DiffusionDetDatasetMapper, add_diffusiondet_config, DiffusionDetWithTTA
from diffusiondet.dataset_audio import DiffusionDetAudioDataset, AudioEvaluator
from diffusiondet.util.model_ema import add_model_ema_configs, may_build_model_ema, may_get_ema_checkpointer, EMAHook, \
    apply_model_ema_and_restore, EMADetectionCheckpointer
    
from tqdm import tqdm


class Trainer(DefaultTrainer):
    """ Extension of the Trainer class adapted to DiffusionDet. """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super(DefaultTrainer, self).__init__()  # call grandfather's `__init__` while avoid father's `__init()`
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        
        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        model = create_ddp_model(model, broadcast_buffers=False)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer, zero_grad_before_forward=True
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        ########## EMA ############
        kwargs = {
            'trainer': weakref.proxy(self),
        }
        kwargs.update(may_get_ema_checkpointer(cfg, model))
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            **kwargs,
            # trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())
        self._trainer._hooks = self._hooks

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:

        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = build_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        # setup EMA
        may_build_model_ema(cfg, model)
        return model

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        
        return AudioEvaluator(dataset_name, cfg, output_folder)
        
        
        if 'lvis' in dataset_name:
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        else:
            return COCOEvaluator(dataset_name, cfg, True, output_folder)
        

    @classmethod
    def build_train_loader(cls, cfg):
        
        dataset = DiffusionDetAudioDataset(name="ami", split="train", cfg=cfg)
        
        assert len(dataset) == cfg.INPUT.TRAINING_DATASET_LENGTH, f"dataset len is{len(dataset)} but {cfg.INPUT.TRAINING_DATASET_LENGTH} was provided in config"
        
        return build_batch_data_loader(dataset,
                total_batch_size=cfg.INPUT.TOT_BATCH_SIZE,
                num_workers=cfg.INPUT.NUM_WORKERS,
                pin_memory=True,
                drop_last=False,
                sampler=None
            )
        
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        
        dataset = DiffusionDetAudioDataset(name="ami", split="validation", cfg=cfg)
        
        #from tqdm import tqdm 
        
        #for i, a in tqdm(enumerate(dataset), total=len(dataset)):
        #    assert a["image"].shape == (3000, 128), f"{a["image"].shape}, {i}"
        
        return build_batch_data_loader(dataset,
                total_batch_size=1,
                num_workers=1,
                pin_memory=True,
                drop_last=False,
                sampler=None
            )
        

    @classmethod
    def build_optimizer(cls, cfg, model):
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for key, value in model.named_parameters(recurse=True):
            #print(key, value.requires_grad)
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "backbone" in key:
                if "fpn" in key:
                    lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER_FPN
                elif "bottom_up" in key:
                    lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER_AST
                    #print("lr bottom_up", lr)
                else:
                    lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER_AST
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                    cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                    and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                    and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def ema_test(cls, cfg, model, evaluators=None):
        # model with ema weights
        logger = logging.getLogger("detectron2.trainer")
        if cfg.MODEL_EMA.ENABLED:
            logger.info("Run evaluation with EMA.")
            with apply_model_ema_and_restore(model):
                results = cls.test(cfg, model, evaluators=evaluators)
        else:
            results = cls.test(cfg, model, evaluators=evaluators)
            
        return results

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        logger.info("Running inference with test-time augmentation ...")
        model = DiffusionDetWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        if cfg.MODEL_EMA.ENABLED:
            cls.ema_test(cfg, model, evaluators)
        else:
            res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res
    
    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            EMAHook(self.cfg, self.model) if cfg.MODEL_EMA.ENABLED else None,  # EMA hook
            hooks.LRScheduler(),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))
        
        ret.append(GradientWriter(self.cfg.OUTPUT_DIR, self.model))
        
        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            results_df = {x: [y] for x, y in self._last_eval_results.items()}
            results_df = pd.DataFrame.from_dict(results_df)
            
            # append to csv
            csv_path = os.path.join(self.cfg.OUTPUT_DIR, "evaluator_metrics.csv")
            if os.path.exists(csv_path):
                results_df.to_csv(csv_path, mode='a', header=False)
            else:
                results_df.to_csv(csv_path)
            
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))
        
        ret.append(hooks.BestCheckpointer(cfg.TEST.EVAL_PERIOD, self.checkpointer, "AP@0.75", "max"))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=1))
        return ret
    
    def build_writers(self):
        """
        Build a list of writers to be used using :func:`default_writers()`.
        If you'd like a different list of writers, you can overwrite it in
        your trainer.

        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.
        """
        return default_writers(self.cfg.OUTPUT_DIR, self.max_iter)


class GradientWriter(hooks.HookBase):
    
    
    def __init__(self, output_dir, model):
        from torch.utils.tensorboard import SummaryWriter
        
        self.logger = logging.getLogger("detectron2.utils.events")
        self.model = model
        self.writer = SummaryWriter(log_dir=f'{output_dir}')
        self.prev_weights={}
    
    def __do_gradient_norm(self, before=False):
        
        threshold=0
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            self.writer.add_scalar(f"weights/{name}", param.mean().item(), get_event_storage().iter)
            
            if name in self.prev_weights:
                if abs(param.mean().item()-self.prev_weights[name]) <= threshold:
                    self.logger.info(f"{name} has constant weights.")
                    self.prev_weights[name] = param.mean().item()
            else:
                self.prev_weights[name] = param.mean().item()
            
            
            """
            if param.grad is None:
                #self.logger.info(f"{name} has no gradient.")
                continue
            
            self.writer.add_histogram(f"gradients/{name}", param.grad, get_event_storage().iter)
            
            param_norm = param.data.norm(2).item()
            grad_norm = param.grad.norm(2).item()
            if param_norm > 0:  # Avoid division by zero
                ratio = grad_norm / param_norm
                self.writer.add_scalar(f"gradients_norm/{name}", ratio, get_event_storage().iter)
            """
    def after_backward(self):
        self.__do_gradient_norm(before=False)
    
    def close(self):
        pass

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    cfg.defrost()
    cfg.TEST.EVAL_PERIOD = 1000 #cfg.INPUT.TRAINING_DATASET_LENGTH // cfg.INPUT.TOT_BATCH_SIZE #test every epoch
    cfg.SOLVER.MAX_ITER = (cfg.INPUT.TRAINING_DATASET_LENGTH  // cfg.INPUT.TOT_BATCH_SIZE) * cfg.SOLVER.NUM_EPOCHS #stop training after num_epochs
    
    cfg.SOLVER.WARMUP_ITERS = 10000 #0 * (cfg.INPUT.TRAINING_DATASET_LENGTH // cfg.INPUT.TOT_BATCH_SIZE) #warmup for one complete epoch
    
    #TODO: add SOLVER.STEPS here to be parametrized by MAX_ITER
    cfg.freeze()
    
    default_setup(cfg, args)
    return cfg


def main(args):
    
    print(args.resume)
    cfg = setup(args)
    
    #print(cfg)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        kwargs = may_get_ema_checkpointer(cfg, model)
        
        model_path = os.path.join(cfg.OUTPUT_DIR, "model_0000899.pth")
        
        if cfg.MODEL_EMA.ENABLED:
            EMADetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR, **kwargs).resume_or_load(model_path, #cfg.MODEL.WEIGHTS,
                                                                                              resume=args.resume)
        else:
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR, **kwargs).resume_or_load(model_path, #cfg.MODEL.WEIGHTS,
                                                                                           resume=args.resume)
        res = Trainer.ema_test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    #args.resume = False ########################################################## MODIFIED
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
