import copy
import json
import logging
import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
import librosa

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
                    # rescale start time to the segment
                    segment_start_time = segment_index * self.seconds_per_segment
                    new_start_time = start_time - segment_start_time
                else:
                    new_start_time = 0 # segment_index * self.seconds_per_segment
                    
                if segment_index == ending_segment:
                    new_end_time = end_time - segment_start_time
                else:
                    new_end_time = self.seconds_per_segment - 1 # (segment_index + 1) * self.seconds_per_segment
                
                new_annotations.append([new_start_time*100, new_end_time*100])
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
        
        #self.plot_spectrogram(audio_dict)
        
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
            
            """
            self.feature_extractor = ASTFeatureExtractor(
                sampling_rate=cfg.INPUT.SAMPLING_RATE,
                max_length=cfg.INPUT.SECONDS_PER_SEGMENT * cfg.INPUT.SAMPLING_RATE,
            )
            """
            
            self.feature_extractor = ASTFeatureExtractor.from_pretrained(cfg.MODEL.AST.PRETRAINED_MODEL)
            
            # Modifica il tasso di campionamento e la lunghezza massima
            self.feature_extractor.sampling_rate = cfg.INPUT.SAMPLING_RATE
            self.feature_extractor.max_length = cfg.INPUT.SECONDS_PER_SEGMENT * 100 #* cfg.INPUT.SAMPLING_RATE
            self.feature_extractor.num_mel_bins = 5
            
            print(self.feature_extractor.sampling_rate)
            print(self.feature_extractor.max_length)
            
            self.sample_rate = self.feature_extractor.sampling_rate
            self.seconds_per_segment = cfg.INPUT.SECONDS_PER_SEGMENT
            
            self.audio_generator()

    @staticmethod
    def plot(spectrogram: np.ndarray, time_pairs: list, speaker_ids: list):
        
        print(spectrogram.shape)
        
        plt.matshow(spectrogram)
        
        for (start_time, end_time), speaker_id in zip(time_pairs, speaker_ids):
            #plt.axvline(x=start_time, color="red", linestyle="--", label=speaker_id)
            #plt.axvline(x=end_time, color="red", linestyle="--", label=speaker_id)
            pass
        #plt.legend()
        plt.show()  
    
    def plot_spectrogram(self, spectrogram, 
                        time_pairs, speaker_ids,
                         title="Mel-Spectrogram"):
        """
        Plots a precomputed spectrogram.

        Args:
            spectrogram (ndarray): 2D array of the spectrogram (frequency x time).
            sample_rate (int): Sampling rate of the original audio.
            title (str): Title of the plot.
        """
        """
        
        print(audio_dict["file_name"])
        
        # Load the audio file
        audio_path = audio_dict["file_name"]
        waveform, sr = torchaudio.load(audio_path)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        self.feature_extractor.max_length = 7200
        
        # Update the segment with the audio feature
        spectrogram =  self.feature_extractor(
            waveform.squeeze(0).numpy(), 
            return_tensors="pt", 
            sampling_rate=self.sample_rate
            
        )["input_values"]
        """
        # Define the time and frequency axes
        num_time_frames = spectrogram.shape[1]
        num_frequency_bins = spectrogram.shape[0]

        # Time axis (seconds)
        time = np.linspace(0, num_time_frames / self.sample_rate, num_time_frames)

        # Frequency axis (Hz)
        frequency = np.linspace(0, self.sample_rate / 2, num_frequency_bins)

        print(spectrogram.shape, spectrogram.max(), spectrogram.min())
        
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
        print(len(unique_speaker_ids))
        color_palette = plt.cm.get_cmap("spring", len(unique_speaker_ids))
        color_map = {speaker_id: color_palette(i) for i, speaker_id in enumerate(unique_speaker_ids)}
        speakers_already=[]
        for (start_time, end_time), speaker_id in zip(time_pairs, speaker_ids):
            print(start_time, end_time, speaker_id)
            color = color_map.get(speaker_id, 'black')  # Default color is black if speaker_id not in color_map
            if speaker_id not in speakers_already:
                plt.axvline(x=start_time, color=color, linestyle="-", label=speaker_id)
                speakers_already.append(speaker_id)
            else:
                plt.axvline(x=start_time, color=color, linestyle="-")
                
            plt.axvline(x=end_time, color=color, linestyle="-")
            #plt.axvspan(start_time, end_time, color=color, alpha=0.3)
        
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
        plt.tight_layout()
        plt.show()

    # Load the audio file
    @staticmethod
    def get_min_max_frequencies(spectrogram, sample_rate, threshold=0):
        """
        Extract the minimum and maximum frequencies from a spectrogram.

        Args:
            spectrogram (ndarray): The spectrogram (magnitude or power) of the audio.
            sample_rate (int): The sampling rate of the original audio.
            threshold (float): Energy threshold to identify active frequencies.
        
        Returns:
            min_freq (float): Minimum frequency with significant energy.
            max_freq (float): Maximum frequency with significant energy.
        """
        # Compute the frequency axis for the spectrogram
        num_freq_bins = spectrogram.shape[0]
        frequencies = np.linspace(0, sample_rate / 2, num_freq_bins)

        # Find frequency bins with energy above the threshold
        active_bins = np.any(spectrogram > threshold, axis=1)

        # Extract minimum and maximum frequencies
        min_freq = frequencies[active_bins].min() if np.any(active_bins) else 0
        max_freq = frequencies[active_bins].max() if np.any(active_bins) else sample_rate / 2

        return min_freq, max_freq
        
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
        
        #torchaudio.save("datasets/ami/prova.wav", waveform_segments[self.all_segments[idx_segment]["segment_id"]], self.sample_rate)
        
        # Update the segment with the audio feature
        self.all_segments[idx_segment].update({"segment": self.feature_extractor(
            waveform_segments[self.all_segments[idx_segment]["segment_id"]].squeeze(0).numpy(), 
            return_tensors="pt", 
            sampling_rate=self.sample_rate
        )["input_values"]})
        
        self.plot_spectrogram(self.all_segments[idx_segment]["segment"].squeeze(0), 
                              self.all_segments[idx_segment]["time_pairs"], self.all_segments[idx_segment]["speaker_ids"])
        
                  #self.all_segments[idx_segment]["time_pairs"],
                 # self.all_segments[idx_segment]["speaker_ids"])
        #torch.save(waveform_segments[self.all_segments[idx_segment]["segment_id"]].squeeze(0), f"datasets/ami/{idx_segment}.wav")
        
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
            
    
if __name__ == "__main__":
    dataset = DiffusionDetAudioDataset()
    
    for i in dataset:
        pass
    """
    dataset = DiffusionDetAudioDataset()
    dataset.__iter__()
    """