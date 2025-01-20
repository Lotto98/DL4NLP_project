import copy
import json
import logging
import pickle
import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
from .loss import SetCriterionDynamicK, HungarianMatcherDynamicK

from detectron2.data import transforms as T
from transformers import ASTFeatureExtractor
from xml.etree import ElementTree as ET
from detectron2.structures import Instances, Boxes
from detectron2.evaluation import DatasetEvaluator
import os
from torch.utils.data import IterableDataset
import torch.nn.functional as F

from .util.box_ops import box_xyxy_to_cxcywh

#from sklearn.preprocessing import LabelEncoder

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
    
    def to_yolo_format(self):
        from torchvision.utils import save_image
        
        path = "datasets/ami_yolo"
        
        path_labels = os.path.join(path, f"labels/{self.split}")
        path_images = os.path.join(path, f"images/{self.split}")
        
        os.makedirs(path_labels, exist_ok=True)
        os.makedirs(path_images, exist_ok=True)
        
        audio_dict = {}
        
        for idx in trange(len(self)):
            item = self.__getitem__(idx)
            
            txt_file = os.path.join(path_labels, f"{idx}.txt")
            image_file = os.path.join(path_images, f"{idx}.png")
            
            with open(txt_file, "w") as f:
                for box, label in zip(item["instances"].gt_boxes.tensor, item["instances"].gt_classes):
                    h,w = item["instances"].image_size
                    box = box/torch.tensor([w,h,w,h], dtype=torch.float32)
                    box = box_xyxy_to_cxcywh(box)
                    f.write(f"{0} {box[0]} {box[1]} {box[2]} {box[3]} \n")
            
            save_image(item["image"].unsqueeze(0), image_file)
            
            if item["audio_name"] not in audio_dict:
                audio_dict[item["audio_name"]] = [idx, idx]
            
            audio_dict[item["audio_name"]][1] = idx
                
        json_file = os.path.join(path, f"audio_dict_{self.split}.json")
        with open(json_file, "w") as f:
            json.dump(audio_dict, f)
    
    def segments_generator(self, idx_audio):
        # Copy dataset_dict to avoid modifying the original one
        audio_dict = self.annotations_per_audio[idx_audio].copy()       
        
        # Load the audio file
        waveform_segments = self.load_audio(audio_dict["file_name"])
        self.audio_waveform_segments[audio_dict["file_name"]] = waveform_segments
        
        # split also the time pair if it falls between two or more segments
        new_annotations = []
        new_labels = []
        new_segments = []
        new_texts = []
        for (start_time, end_time), speaker_id, text in zip(audio_dict["time_pairs"], audio_dict["speaker_id"], audio_dict["text"]):
            if abs(start_time - end_time) < 0.001:
                #print("skipping", start_time, end_time, speaker_id)
                continue
            
            # print("ORIGINAL",start_time, end_time, speaker_id)
            starting_segment = int(start_time // self.seconds_per_segment)
            ending_segment = int(end_time // self.seconds_per_segment)
            
            if starting_segment > len(waveform_segments) - 1:
                print("WARNING: start time is greater than the audio length for audio", audio_dict["file_name"])
                continue
            
            if (ending_segment > len(waveform_segments) - 1):
                ending_segment = len(waveform_segments) - 1
                end_time = len(waveform_segments) * self.seconds_per_segment
                print("WARNING: end time is greater than the audio length for audio", audio_dict["file_name"])
            
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
                
                new_annotations.append([new_start_time*99.818181, new_end_time*99.818181]) # * 100 to convert in correct scale for spectogram
                new_labels.append(speaker_id)
                new_segments.append(segment_index)
                new_texts.append(text)
                
        # sort arrays
        combined = list(zip(new_segments, new_annotations, new_labels, new_texts))
        combined.sort(key=lambda x: (x[0], x[1][0]))  # Ordina per segment_index

        new_segments, new_annotations, new_labels, new_texts = zip(*combined)
        new_segments = list(new_segments)
        new_annotations = list(new_annotations)
        new_labels = list(new_labels)
        new_texts = list(new_texts)
                
        current_times = {}  
        new_items = {}
        max_boxes = 0
        prev_segment = -1
        for new_annotation, new_label, new_segment, t in zip(new_annotations, new_labels, new_segments, new_texts):
            if new_segment not in new_items:  
                """ UNCOMMENT TO ADD NOISE
                # if there are missing segments, add noise
                if prev_segment != -1 and new_segment - prev_segment > 1:
                    for missing_segment in range(prev_segment + 1, new_segment):
                        new_items[missing_segment] = {
                            "segment_id": missing_segment, # self.feature_extractor(waveform_segments[missing_segment].squeeze(0).numpy(), return_tensors="pt", sampling_rate=self.sample_rate)["input_values"],
                            "time_pairs": [[0, self.seconds_per_segment * 100]],
                            "speaker_ids": [0],
                            "audio_file": audio_dict["file_name"]
                        }
                        
                        current_times[missing_segment] = self.seconds_per_segment * 100
                        
                # add noise at the end of prev segment if needed
                if prev_segment != -1 and current_times[prev_segment] < (self.seconds_per_segment * 100) - 0.001:
                    new_items[prev_segment]["time_pairs"].append([current_times[prev_segment], self.seconds_per_segment * 100])
                    new_items[prev_segment]["speaker_ids"].append(0)
                    current_times[prev_segment] = self.seconds_per_segment * 100
                """
                
                # if there are missing segments, add noise
                if prev_segment != -1 and new_segment - prev_segment > 1:
                    for missing_segment in range(prev_segment + 1, new_segment):
                        new_items[missing_segment] = {
                            "segment_id": missing_segment, 
                            "time_pairs": [],
                            "speaker_ids": [],
                            "audio_file": audio_dict["file_name"],
                            "text": []
                        }
                
                #current_times[new_segment] = 0
                new_items[new_segment] = {
                    "segment_id": new_segment, # self.feature_extractor(waveform_segments[new_segment].squeeze(0).numpy(), return_tensors="pt", sampling_rate=self.sample_rate)["input_values"],
                    "time_pairs": [],
                    "speaker_ids": [],
                    "audio_file": audio_dict["file_name"],
                    "text": []
                }
            
            """ UNCOMMENT TO ADD NOISE 
            # transform to two classes noise (0) and silence (1)
            st, et = new_annotation
            if st - current_times[new_segment] > 0.001:
                new_items[new_segment]["time_pairs"].append([current_times[new_segment], st])
                new_items[new_segment]["speaker_ids"].append(0)
            
            if et > current_times[new_segment]:
                current_times[new_segment] = et
            """

            new_items[new_segment]["time_pairs"].append(new_annotation)
            
            
            if new_label[0] == 'M':
                new_items[new_segment]["speaker_ids"].append(0)
            elif new_label[0] == 'F':
                new_items[new_segment]["speaker_ids"].append(1)
            else:
                raise ValueError(f"Speaker label {new_label[0]} not supported.")
            
            new_items[new_segment]["text"].append(t)
            
            #new_items[new_segment]["speaker_ids"].append(0) # new_label forced to 0 = speach
            
            max_boxes = max(max_boxes, len(new_items[new_segment]["time_pairs"]))
            
            prev_segment = new_segment
        
        """ UNCOMMENT TO ADD NOISE
        # add noise at the end of the last segment if needed
        if prev_segment != -1 and current_times[prev_segment] < (self.seconds_per_segment * 100) - 0.001:
            new_items[prev_segment]["time_pairs"].append([current_times[prev_segment], self.seconds_per_segment * 100])
            new_items[prev_segment]["speaker_ids"].append(0)
            current_times[prev_segment] = self.seconds_per_segment * 100
        
        if len(waveform_segments) > max(new_items.keys()) + 1:
            for missing_segment in range(max(new_items.keys()) + 1, len(waveform_segments)):
                new_items[missing_segment] = {
                    "segment_id": missing_segment,
                    "time_pairs": [[0, self.seconds_per_segment * 100]],
                    "speaker_ids": [0],
                    "audio_file": audio_dict["file_name"]
                }
        """

        # add empty segments if needed at the beginning and at the end
        if min(new_items.keys()) !=0:
            for missing_segment in range(0, min(new_items.keys())):
                new_items[missing_segment] = {
                    "segment_id": missing_segment,
                    "time_pairs": [],
                    "speaker_ids": [],
                    "audio_file": audio_dict["file_name"],
                    "text": []
                }
        
        if len(waveform_segments) > max(new_items.keys()) + 1:
            for missing_segment in range(max(new_items.keys()) + 1, len(waveform_segments)):
                new_items[missing_segment] = {
                    "segment_id": missing_segment,
                    "time_pairs": [],
                    "speaker_ids": [],
                    "audio_file": audio_dict["file_name"],
                    "text": []
                }
                
        assert len(waveform_segments) == len(new_items)
        
        return dict(sorted(new_items.items())), max_boxes
    
    def load_audio(self, audio_path):
        # Load the audio file
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
        
        return waveform_segments
    
    def audio_generator(self):
        new_id = 0
        self.all_segments = {}
        self.max_boxes = 0
        
        for idx_audio in trange(len(self.annotations_per_audio), leave=False, disable=False):
            
            segments, max_boxes_segment = self.segments_generator(idx_audio)
            
            self.max_boxes = max(self.max_boxes, max_boxes_segment)
            
            for _, v in segments.items():
                self.all_segments[new_id] = v
                new_id += 1
    
    def __init__(self, cfg, name:str="ami", split:str="train", start_ixd:int|None=None, end_idx:int|None=None):
        
        if name == "ami":
            self.annotations_per_audio = self.load_ami(split)
            
            self.feature_extractor = ASTFeatureExtractor(sampling_rate=cfg.INPUT.SAMPLING_RATE,
                                                        max_length=1126,#cfg.INPUT.SECONDS_PER_SEGMENT * 100 + 26,
                                                        num_mel_bins=166,
                                                        )
            
            self.sample_rate = self.feature_extractor.sampling_rate
            self.seconds_per_segment = cfg.INPUT.SECONDS_PER_SEGMENT
            
            #self.label_encoder = LabelEncoder()
            self.is_training = (split == "train")
            self.split = split
                
            self.audio_waveform_segments = {}
            
            print(f"####################### LOADING DATASET {name} WITH SPLIT {split} #######################")
            
            self.audio_generator()
            
            #self.plot_statistics()
            
            print("- sampling rate:", self.feature_extractor.sampling_rate)
            print("- max length:", self.feature_extractor.max_length)
            print("- max boxes per segment", self.max_boxes)
            print("- number of segments", len(self))
            
            #debug: set the start and end index for the dataset
            if start_ixd is not None or end_idx is not None:
                start_ixd = 0 if start_ixd is None else start_ixd
                end_idx = len(self) if end_idx is None else end_idx
                self.all_segments = {k: v for k, v in self.all_segments.items() if start_ixd <= k <= end_idx}
            
        else:
            raise ValueError(f"Dataset {name} not supported.")
            
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
        for (start_time, end_time), speaker_id, in zip(time_pairs, speaker_ids):
            #print(start_time, end_time, speaker_id, t)
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
        
        print()
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
        plt.xlabel("Time (cs)")
        plt.ylabel("Mel-spectrogram bins")
        plt.tight_layout()
        plt.show() #.savefig("figure.png") #.show()
    
    def plot_statistics(self):
        
        box_count_distribution = { n_boxes:0 for n_boxes in range(0, self.max_boxes+1) }
        labels_count = {0:0, 1:0}
        
        for _, segment in self.all_segments.items():
            l = len(segment["time_pairs"])
            box_count_distribution[l] += 1
            for label in segment["speaker_ids"]:
                labels_count[label] += 1
        
        print("Statistics for the number of boxes per segment")
        
        plt.bar(box_count_distribution.keys(), box_count_distribution.values())
        plt.show()
        
        print("Statistics for the number of labels per segment")
        
        plt.bar(labels_count.keys(), labels_count.values())
        plt.show()
        
        
        #plt.savefig("boxes_per_segment.png")
        
    def __getitem__(self, idx_segment: int):
        
        #idx_segment = 0 if idx_segment%2 == 0 else 1
        
        waveform_segments = self.audio_waveform_segments[self.all_segments[idx_segment]["audio_file"]]
        
        # Update the segment with the audio feature
        features = self.feature_extractor(
            waveform_segments[self.all_segments[idx_segment]["segment_id"]].squeeze(0).numpy(), 
            return_tensors="pt", 
            sampling_rate=self.sample_rate
        )["input_values"]
        
        #self.plot_spectrogram(
        #    features, 
        #    self.all_segments[idx_segment]["time_pairs"], 
        #    self.all_segments[idx_segment]["speaker_ids"],
        #    #self.all_segments[idx_segment]["text"], 
        #    self.sample_rate, 
        #    title=f"Mel-Spectrogram for segment {idx_segment}"
        #)
        #print(self.all_segments[idx_segment]["time_pairs"])
        
        # to make swin work
        # features = F.interpolate(features.unsqueeze(0), size=(1024, 160), mode='bicubic', align_corners=False)
        #features = F.pad(features, (0, 0, 0, 26), value=0)
        
        return {
            "image": features.squeeze(0), #torch.cat([features, features, features], dim=0)
            "instances": Instances(
                image_size = features.shape[-2:],
                gt_boxes = Boxes(torch.tensor([[0, st, features.shape[-1], et] for st, et in self.all_segments[idx_segment]["time_pairs"]])),
                gt_classes = torch.as_tensor(
                                    [
                                        speaker_id
                                        for speaker_id in self.all_segments[idx_segment]["speaker_ids"]
                                    ]
                                    #self.label_encoder.transform(self.all_segments[idx_segment]["speaker_ids"])
                ),
                texts = self.all_segments[idx_segment]["text"],
            ),
            "audio_name": self.all_segments[idx_segment]["audio_file"],
            "segment_id_for_waveform" : self.all_segments[idx_segment]["segment_id"]
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
            if not self.is_training:
                break
                
    def __len__(self):
        return len(self.all_segments)

class AudioEvaluator(DatasetEvaluator):
    def __init__(self, name, cfg, output_dir):
        super().__init__()
        
        self.iou_thresholds = [0.5, 0.75]
        self.results = []
    
    def reset(self):
        # resetta le metriche per un nuovo round di valutazione
        self.results = []
    
    def _bb_to_time_pairs(self, boxes):
        _, y0, _, y1 = boxes.unbind(-1)
        return torch.stack([y0, y1], dim=-1).tolist()
    
    def process(self, inputs, outputs):
        
        for i, o in zip(inputs, outputs):
            
            if False:
                times_1 = self._bb_to_time_pairs(i["instances"].gt_boxes.tensor)
                
                DiffusionDetAudioDataset.plot_spectrogram(
                    i["image"], 
                    times_1, 
                    i["instances"].gt_classes.tolist(),
                    16000, 
                    title=f"Mel-Spectrogram for gt segment {i["segment_id_for_waveform"]}"
                )
                
                times_2 = self._bb_to_time_pairs(o["instances"].pred_boxes.tensor)
                
                DiffusionDetAudioDataset.plot_spectrogram(
                    i["image"], 
                    times_2, 
                    o["instances"].pred_classes.tolist(), 
                    16000, 
                    title=f"Mel-Spectrogram for pred segment {i["segment_id_for_waveform"]}"
                )
                
                if len(times_1) != len(times_2):
                    print("WARNING: different number of boxes", len(times_1), len(times_2))
        
        for input, output in zip(inputs, outputs):
            gt_boxes = input["instances"].gt_boxes.tensor.numpy()
            pred_boxes = output["instances"].pred_boxes.tensor.cpu().numpy()
            scores = output["instances"].scores.cpu().numpy()

            # Match ground truth boxes to predicted boxes here
            self.results.append(self.evaluate_boxes(gt_boxes, pred_boxes, scores))
        
        #maybe not feasible because we dont have the class logits
        #self.criterion(formatted_outputs, formatted_inputs)
        
    def evaluate_boxes(self, gt_boxes, pred_boxes, scores):
        """
        Evaluate IoU and compute metrics across multiple thresholds.
        """
        
        if len(gt_boxes) == 0 and len(pred_boxes) == 0:
            # No GT and no predictions, perfect score
            base_result = {f"AP@{iou:.2f}": 1.0 for iou in self.iou_thresholds}
            base_result.update({f"F1@{iou:.2f}": 1.0 for iou in self.iou_thresholds})
            base_result["SpeechCoveragePrecentage"] = 1.0
            return base_result
        
        base_result = {f"AP@{iou:.2f}": 0.0 for iou in self.iou_thresholds}
        base_result.update({f"F1@{iou:.2f}": 0.0 for iou in self.iou_thresholds})
        base_result["SpeechCoveragePrecentage"] = 0.0
        
        if len(pred_boxes) == 0:
            # No predictions, AP is 0 for all thresholds if there are GT boxes
            return base_result
    
        if len(gt_boxes) == 0:
            # No ground truth, AP is 0 for all thresholds if there are predicted boxes
            return base_result

        # Calculate IoU for each pair of boxes
        iou_matrix = self.calculate_iou_matrix(gt_boxes, pred_boxes)

        # Compute precision for each IoU threshold
        metrics = {}
        for threshold in self.iou_thresholds:
            matches = iou_matrix > threshold  # Boolean matrix of matches
            true_positives = matches.any(axis=0).sum()  # Count matched predicted boxes
            false_positives = len(pred_boxes) - true_positives
            false_negatives = len(gt_boxes) - matches.any(axis=1).sum()  # Count unmatched GT boxes
            precision = true_positives / (true_positives + false_positives) if len(pred_boxes) > 0 else 0.0
            metrics[f"AP@{threshold:.2f}"] = precision
            
            # Compute precision and recall
            precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0.0
            recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0.0

            # Compute F1 score
            f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0

            metrics[f"F1@{threshold:.2f}"] = f1
            
        # compute percentage of speech time correctly covered by pred time pairs
        tot_time = 0
        tot_time_covered = 0
        for gt_box in gt_boxes:
            gt_start, gt_end = gt_box[1], gt_box[3]
            tot_time += gt_end - gt_start
            # sort pred boxes by start time
            pred_boxes_sorted = sorted(pred_boxes, key=lambda x: x[1])
            merged = []
            for pred_box in pred_boxes_sorted:
                pred_start, pred_end = pred_box[1], pred_box[3]
                if not merged or pred_start > merged[-1][1]:
                    merged.append([pred_start, pred_end])
                else:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], pred_end))

            # Check coverage and compute uncovered time
            uncovered_time = 0
            current_start = gt_start

            for seg_start, seg_end in merged:
                if seg_start > current_start:
                    uncovered_time += seg_start - current_start
                current_start = max(current_start, seg_end)

                if current_start >= gt_end:
                    break

            if current_start < gt_end:
                uncovered_time += gt_end - current_start   
                
            tot_time_covered += gt_end - gt_start - uncovered_time                 

        metrics["SpeechCoveragePrecentage"] = tot_time_covered / tot_time if tot_time > 0 else 0.0
        
        return metrics
        
    def calculate_iou_matrix(self, gt_boxes, pred_boxes):
        """
        Calculates the IoU matrix between ground truth and predicted boxes.
        """
        def iou(box1, box2):
            p1, p2 = np.maximum(box1[:2], box2[:2]), np.minimum(box1[2:], box2[2:])
            intersection = np.prod(np.maximum(0, p2 - p1 + 1))
            area1 = np.prod(box1[2:] - box1[:2] + 1)
            area2 = np.prod(box2[2:] - box2[:2] + 1)
            union = area1 + area2 - intersection
            return intersection / union if union > 0 else 0.0

        iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
        for i, gt_box in enumerate(gt_boxes):
            for j, pred_box in enumerate(pred_boxes):
                iou_matrix[i, j] = iou(gt_box, pred_box)
                
        return iou_matrix
    
    def evaluate(self):
        if len(self.results) == 0:
            results = {f"AP@{iou:.2f}": 0.0 for iou in self.iou_thresholds}
            results.update({f"F1@{iou:.2f}": 0.0 for iou in self.iou_thresholds})
            results["SpeechCoveragePrecentage"] = 0.0
            return results

        # Aggregate results across all IoU thresholds
        final_results = {f"AP@{iou:.2f}": 0.0 for iou in self.iou_thresholds}
        final_results.update({f"F1@{iou:.2f}": 0.0 for iou in self.iou_thresholds})
        final_results["SpeechCoveragePrecentage"] = 0.0

        for res in self.results:
            for key in final_results.keys():
                final_results[key] += res[key]

        # Average over all samples
        num_samples = len(self.results)
        for key in final_results.keys():
            final_results[key] /= num_samples

        return final_results
    