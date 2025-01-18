from diffusiondet.dataset_audio import DiffusionDetAudioDataset
from argparse import Namespace


INPUT = Namespace(SAMPLING_RATE=16000, SECONDS_PER_SEGMENT=11)
cfg = Namespace(INPUT=INPUT)

DiffusionDetAudioDataset( cfg=cfg, name="ami", split="validation").to_yolo_format()