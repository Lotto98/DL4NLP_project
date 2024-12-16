import copy
import json
import logging
import numpy as np
import torch
import torchaudio

from detectron2.data import transforms as T
from transformers import ASTFeatureExtractor
from xml.etree import ElementTree as ET
from detectron2.structures import Instances, Boxes
from detectron2.data import DatasetCatalog, MetadataCatalog
import os
from torch.utils.data import IterableDataset

class DiffusionDetAudioDataset(IterableDataset):

    def load_ami(self, split = "train"):
        
        audio_root="datasets/ami/audio"
        annotation_root="datasets/ami/annotations"
    
        if not os.path.exists(annotation_root):
            raise ValueError(f"Annotation root {annotation_root} does not exist.")
        
        with open(os.path.join(annotation_root, f"{split}.json"), "r") as f:
            annotations = json.load(f)
        
        for annotation in annotations:
            file_name = annotation["meeting_id"]
            annotation["file_name"] = os.path.join(audio_root, f"{file_name}.wav")    
    
        return annotations
    
    def segments_generator(self, idx_audio):
        # Copy dataset_dict to avoid modifying the original one
        audio_dict = self.annotations_per_audio[idx_audio].copy()

        # split also the time pair if it falls between two or more segments
        new_annotations = []
        new_labels = []
        new_segments = []
        for (start_time, end_time), speaker_id in zip(audio_dict["time_pairs"], audio_dict["speaker_id"]):
            starting_segment = int(start_time // self.seconds_per_segment)
            ending_segment = int(end_time // self.seconds_per_segment)
            for segment_index in range(starting_segment, ending_segment + 1):
                if segment_index == starting_segment:
                    new_start_time = start_time
                else:
                    new_start_time = segment_index * self.seconds_per_segment
                    
                if segment_index == ending_segment:
                    new_end_time = end_time
                else:
                    new_end_time = (segment_index + 1) * self.seconds_per_segment
                
                new_annotations.append([new_start_time, new_end_time])
                new_labels.append(speaker_id)
                new_segments.append(segment_index)
        
        new_items = {}
        for new_annotation, new_label, new_segment in zip(new_annotations, new_labels, new_segments):
            if new_segment not in new_items:  
                new_items[new_segment] = {
                    "segment_id": new_segment, # self.feature_extractor(waveform_segments[new_segment].squeeze(0).numpy(), return_tensors="pt", sampling_rate=self.sample_rate)["input_values"],
                    "segment": None,
                    "time_pairs": [],
                    "speaker_ids": [],
                    "audio_file": audio_dict["file_name"]
                }

            new_items[new_segment]["time_pairs"].append(new_annotation)
            new_items[new_segment]["speaker_ids"].append(new_label)
        
        return new_items
    
    def audio_generator(self):
        new_id = 0
        self.all_segments = {}
        for idx_audio in range(len(self.annotations_per_audio)):
            segments = self.segments_generator(idx_audio)
            for _, v in segments.items():
                self.all_segments[new_id] = v
                new_id += 1
    
    def __init__(self, cfg, name="ami", split="train"):
        
        if name == "ami":
            self.annotations_per_audio = self.load_ami(split)
            
            self.feature_extractor = ASTFeatureExtractor(
                sampling_rate=cfg.INPUT.SAMPLING_RATE,
                max_length=cfg.INPUT.SECONDS_PER_SEGMENT * cfg.INPUT.SAMPLING_RATE,
            )
            
            self.sample_rate = self.feature_extractor.sampling_rate
            self.seconds_per_segment = cfg.INPUT.SECONDS_PER_SEGMENT
            
            self.audio_generator()

    def __getitem__(self, idx_segment: int):
        # Load the audio file
        audio_path = self.all_segments[idx_segment]["audio_file"]
        waveform, sr = torchaudio.load(audio_path)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # split the audio in segments of x-seconds
        segment_length = self.seconds_per_segment * self.sample_rate
        waveform_segments = waveform.split(segment_length, dim=1)
        
        # Update the segment with the audio feature
        self.all_segments[idx_segment].update({"segment": self.feature_extractor(
            waveform_segments[self.all_segments[idx_segment]["segment_id"]].squeeze(0).numpy(), 
            return_tensors="pt", 
            sampling_rate=self.sample_rate
        )["input_values"]})
        
        return {
            "image": self.all_segments[idx_segment]["segment"].squeeze(0),
            "instances": Instances(
                image_size = self.all_segments[idx_segment]["segment"].shape[-2:],
                gt_boxes=Boxes(torch.tensor([[st, 0, et, self.all_segments[idx_segment]["segment"].shape[-1]] for st, et in self.all_segments[idx_segment]["time_pairs"]])),
                gt_classes=torch.zeros(len(self.all_segments[idx_segment]["speaker_ids"])) #self.all_segments[idx_segment]["speaker_ids"]
            )
        }
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            iter_start = 0
            iter_end = len(self.all_segments)
        else:
            per_worker = int(np.ceil(len(self.all_segments) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.all_segments))
    
        for idx in range(iter_start, iter_end):
            yield self.__getitem__(idx)