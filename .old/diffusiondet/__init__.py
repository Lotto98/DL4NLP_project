# ========================================
# Modified by Shoufa Chen
# ========================================
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .config import add_diffusiondet_config
from .detector import DiffusionDet
from .dataset_mapper import DiffusionDetDatasetMapper
from .test_time_augmentation import DiffusionDetWithTTA
from .swintransformer import build_swintransformer_fpn_backbone
#from .simple_audio_encoder import build_simple_backbone_fpn
from .ast_model_v2 import build_ASTModel_backbone_fpn