import copy
import logging
import numpy as np
import torch
import torchaudio

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data import DatasetMapper
from transformers import ASTFeatureExtractor
from xml.etree import ElementTree as ET
from detectron2.structures import Instances, Boxes

class DiffusionDetAudioDatasetMapper(DatasetMapper):
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by DiffusionDet.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    def __init__(self, cfg, is_train=True):
        self.feature_extractor = ASTFeatureExtractor(
            sampling_rate=cfg.INPUT.SAMPLING_RATE,
            #max_length=cfg.INPUT.MAX_LENGTH
        )
        self.sample_rate = cfg.INPUT.SAMPLING_RATE

    def parse_ami_annotation(self, annotation_path, audio_duration, feature_length):
        """
        Parse AMI Corpus XML annotation to extract segments and labels.

        Args:
            annotation_path (str): Path to the XML annotation file.
            audio_duration (float): Duration of the audio in seconds.
            feature_length (int): Length of the extracted features (time dimension).

        Returns:
            boxes (torch.Tensor): A tensor of shape (N, 2) where each row is [start_idx, end_idx].
            labels (torch.Tensor): A tensor of shape (N,) with the label for each box.
        """
        # Parse the XML file
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        # Store boxes and labels
        boxes = []
        labels = []

        # Iterate over segment annotations
        for segment in root.findall(".//segment"):
            start_time = float(segment.attrib["starttime"])
            end_time = float(segment.attrib["endtime"])
            label = int(segment.attrib["speaker_id"])  # Use speaker ID or map to a class

            # Convert times to feature indices
            start_idx = int((start_time / audio_duration) * feature_length)
            end_idx = int((end_time / audio_duration) * feature_length)

            # Store the box and label
            boxes.append([start_idx, end_idx]) # add als
            labels.append(label)

        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

    def __call__(self, dataset_dict):
        # Copy dataset_dict to avoid modifying the original one
        dataset_dict = dataset_dict.copy()

        # Load the audio file
        audio_path = dataset_dict["file_name"]
        annotation_path = dataset_dict["annotation_file"]  # Path to the XML annotation
        waveform, sr = torchaudio.load(audio_path)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Extract features using ASTFeatureExtractor
        features = self.feature_extractor(waveform.squeeze(0).numpy(), return_tensors="pt")
        input_features = features["input_values"]
        feature_length = input_features.size(1)

        # Compute audio duration
        audio_duration = waveform.size(1) / self.sample_rate

        # Parse the AMI annotations and get GT boxes and labels
        gt_boxes, gt_labels = self.parse_ami_annotation(annotation_path, audio_duration, feature_length)

        # Detectron2 expects "image" to be a tensor
        dataset_dict["image"] = input_features.squeeze(0)  # [seq_len, feature_dim]
        dataset_dict["instances"] = Instances(
            image_size=input_features.shape[-2:],  # Match the extracted feature dimensions
            gt_boxes=Boxes(gt_boxes),  # Ground truth boxes
            gt_classes=gt_labels       # Labels for the boxes
        )

        return dataset_dict
    
from detectron2.data import DatasetCatalog, MetadataCatalog

def register_ami_segmentation(name, audio_root="datasets/ami/audio", annotation_root="datasets/ami/annotations"):
    def load_ami_data():
        import os
        dataset_dicts = []
        for audio_file in os.listdir(audio_root):
            if audio_file.endswith(".wav"):
                file_name = os.path.join(audio_root, audio_file)
                annotation_file = os.path.join(annotation_root, audio_file.replace(".wav", ".xml"))
                dataset_dicts.append({
                    "file_name": file_name,
                    "annotation_file": annotation_file
                })
        return dataset_dicts

    DatasetCatalog.register(name, load_ami_data)
    MetadataCatalog.get(name).set(thing_classes=["Speaker1", "Speaker2", "Speaker3"])

# Register the dataset
register_ami_segmentation("ami_segmentation", "/path/to/audio", "/path/to/annotations")