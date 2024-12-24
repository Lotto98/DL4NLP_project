import copy
import json
import logging
import pickle
import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
import librosa

from detectron2.data import transforms as T
from transformers import ASTFeatureExtractor
from xml.etree import ElementTree as ET
from detectron2.structures import Instances, Boxes
from detectron2.evaluation import DatasetEvaluator
import os
from torch.utils.data import IterableDataset

from sklearn.preprocessing import LabelEncoder

from tqdm import trange

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
            if abs(start_time - end_time) < 0.001:
                #print("skipping", start_time, end_time, speaker_id)
                continue
            
            # print("ORIGINAL",start_time, end_time, speaker_id)
            starting_segment = int(start_time // self.seconds_per_segment)
            ending_segment = int(end_time // self.seconds_per_segment)
            for segment_index in range(starting_segment, ending_segment + 1):
                segment_start_time = segment_index * self.seconds_per_segment

                if segment_index == starting_segment:
                    # rescale start time to the segment
                    new_start_time = start_time - segment_start_time
                else:
                    new_start_time = 0 # segment_index * self.seconds_per_segment
                    
                if segment_index == ending_segment:
                    new_end_time = end_time - segment_start_time
                else:
                    new_end_time = self.seconds_per_segment # (segment_index + 1) * self.seconds_per_segment
                
                if abs(new_start_time - new_end_time) < 0.001:
                    #print("skipping", new_start_time, new_end_time, speaker_id)
                    continue
                
                assert new_start_time < new_end_time, f"start time {new_start_time} must be less than end time {new_end_time}"
                
                new_annotations.append([new_start_time*100, new_end_time*100]) # * 100 to convert in correct scale for spectogram
                new_labels.append(speaker_id)
                new_segments.append(segment_index)
        
        new_items = {}
        unique_labels = set()
        max_boxes = 0
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
            
            unique_labels.add(new_label)
            max_boxes = max(max_boxes, len(new_items[new_segment]["time_pairs"]))
        
        # Load the audio file if it has not been loaded yet
        if audio_dict["file_name"] not in self.audio_waveform_segments:
            
            # Load the audio file
            audio_path = audio_dict["file_name"]
            f = open(audio_path, 'rb')
            waveform, sr = torchaudio.load(f)
            f.close()

            # Check if the audio is stereo (2 channels)
            if waveform.shape[0] > 1:
                # Convert to mono by averaging the two channels
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # split the audio in segments of x-seconds
            segment_length = self.seconds_per_segment * self.sample_rate #check this
            waveform_segments = waveform.split(segment_length, dim=1)
            
            self.audio_waveform_segments[audio_dict["file_name"]] = waveform_segments
        
        return new_items, unique_labels, max_boxes
    
    def audio_generator(self):
        new_id = 0
        self.all_segments = {}
        all_labels = set()
        self.max_boxes = 0
        
        for idx_audio in trange(len(self.annotations_per_audio), leave=False):
            
            segments, unique_labels, max_boxes_segment = self.segments_generator(idx_audio)
            
            all_labels.update(unique_labels)
            self.max_boxes = max(self.max_boxes, max_boxes_segment)
            
            for _, v in segments.items():
                self.all_segments[new_id] = v
                new_id += 1
        
        """
        if self.fit_label_encoder:
            self.label_encoder = self.label_encoder.fit(list(all_labels))
            with open("datasets/ami/classes.pkl", "wb") as f:
                pickle.dump(self.label_encoder, f)
        else:
            with open("datasets/ami/classes.pkl", "rb") as f:
                self.label_encoder = pickle.load(f)
        """
    
    def __init__(self, cfg, name:str="ami", split:str="train", label_encoder: LabelEncoder | None = None):
        
        if name == "ami":
            self.annotations_per_audio = self.load_ami(split)
            
            self.feature_extractor = ASTFeatureExtractor.from_pretrained(cfg.MODEL.AST.PRETRAINED_MODEL)
            
            # Modifica il tasso di campionamento e la lunghezza massima
            self.feature_extractor.sampling_rate = cfg.INPUT.SAMPLING_RATE
            self.feature_extractor.max_length = cfg.INPUT.SECONDS_PER_SEGMENT * 100 #* cfg.INPUT.SAMPLING_RATE
            
            self.sample_rate = self.feature_extractor.sampling_rate
            self.seconds_per_segment = cfg.INPUT.SECONDS_PER_SEGMENT
            
            #self.label_encoder = LabelEncoder()
            #self.fit_label_encoder = (split == "train")
                
            self.audio_waveform_segments = {}
            
            print(f"####################### LOADING DATASET {name} WITH SPLIT {split} #######################")
            
            self.audio_generator()
            
            print("- sampling rate:", self.feature_extractor.sampling_rate)
            print("- max length:", self.feature_extractor.max_length)
            #print("- number of classes", len(self.label_encoder.classes_))
            print("- max boxes per segment", self.max_boxes)
            
            
    @staticmethod
    def plot_spectrogram(spectrogram, 
                        time_pairs, speaker_ids,
                        sample_rate,
                         title="Mel-Spectrogram"):
        
        # Define the time and frequency axes
        num_time_frames = spectrogram.shape[1]
        num_frequency_bins = spectrogram.shape[0]

        # Time axis (seconds)
        time = np.linspace(0, num_time_frames / sample_rate, num_time_frames)

        # Frequency axis (Hz)
        frequency = np.linspace(0, sample_rate / 2, num_frequency_bins)

        #print(spectrogram.shape, spectrogram.max(), spectrogram.min())
        
        # Plot the spectrogram
        plt.figure(figsize=(10, 6))
        plt.imshow(spectrogram.squeeze(0).transpose(1,0), aspect='auto', origin='lower')
                    #extent=[time.min(), time.max(), frequency.min(), frequency.max()])
        plt.colorbar(label='Intensity')
        plt.title(title)
        #plt.xlabel("Time (s)")
        #plt.ylabel("Frequency (Hz)")
        # Define a color map for speaker IDs
        unique_speaker_ids = list(set(speaker_ids))
        #print(len(unique_speaker_ids))
        color_palette = plt.cm.get_cmap("spring", len(unique_speaker_ids))
        color_map = {speaker_id: color_palette(i) for i, speaker_id in enumerate(unique_speaker_ids)}
        speakers_already=[]
        for (start_time, end_time), speaker_id in zip(time_pairs, speaker_ids):
            print(start_time, end_time, speaker_id)
            color = color_map.get(speaker_id, 'black')  # Default color is black if speaker_id not in color_map
            
            if speaker_id not in speakers_already:
                speakers_already.append(speaker_id)
                
                # add label on legend
                plt.plot([], [], color=color, label=speaker_id)
                
            y_pos = 5 + (speakers_already.index(speaker_id) * 5) # add 5 to separate the speakers
            
            # add double arrow to indicate the speaker
            plt.annotate("", xy=(start_time, y_pos), xytext=(end_time, y_pos),
                         arrowprops=dict(arrowstyle="<->", color=color))
            
            # add speaker id on the top of the arrow
            # plt.text((start_time + end_time) / 2, y_pos + 0.1, speaker_id, ha='center', va='bottom', color=color)
        
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
        plt.xlabel("Time (cs)")
        plt.ylabel("Mel-spectrogram bins")
        plt.tight_layout()
        plt.show() #.savefig("figure.png") #.show()
        
    def __getitem__(self, idx_segment: int):
        
        waveform_segments = self.audio_waveform_segments[self.all_segments[idx_segment]["audio_file"]]
        
        # Update the segment with the audio feature
        features = self.feature_extractor(
            waveform_segments[self.all_segments[idx_segment]["segment_id"]].squeeze(0).numpy(), 
            return_tensors="pt", 
            sampling_rate=self.sample_rate
        )["input_values"]
        
        #self.plot_spectrogram(self.all_segments[idx_segment]["segment"].squeeze(0), 
        #                      self.all_segments[idx_segment]["time_pairs"], self.all_segments[idx_segment]["speaker_ids"],
        #                      self.sample_rate)
        
        #print(idx_segment)
        
        return {
            "image": features.squeeze(0),
            "instances": Instances(
                image_size = features.shape[-2:],
                gt_boxes = Boxes(torch.tensor([[st, 0, et, features.shape[-1]] for st, et in self.all_segments[idx_segment]["time_pairs"]])),
                gt_classes = torch.as_tensor(
                                    [
                                        0 if speaker_id[0] == 'M' else 1
                                        for speaker_id in self.all_segments[idx_segment]["speaker_ids"]
                                    ]
                                    #self.label_encoder.transform(self.all_segments[idx_segment]["speaker_ids"])
                                    )
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

        while True:
            for idx in range(iter_start, iter_end):
                yield self.__getitem__(idx)
                
    def __len__(self):
        return len(self.all_segments)

class AudioEvaluator(DatasetEvaluator):
    def __init__(self, name, cfg, output_dir):
        super().__init__()
    
    def reset(self):
        pass
    
    def process(self, inputs, outputs):
        
        for input, output in zip(inputs, outputs):
            print(input)
            print(output)
        pass
    
    def evaluate(self):
        pass