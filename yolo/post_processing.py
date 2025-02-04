import time
import json
import numpy as np
import torch
import argparse
import torchaudio
from ultralytics.engine.results import Results
from ultralytics import YOLO
import os
from tqdm import tqdm, trange
import editdistance
import pandas as pd
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer, cer

from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

import gc

import matplotlib.pyplot as plt

from typing import List

def plot_intervals(boxes_to_plot):
    
    fig, ax = plt.subplots(figsize=(10, len(boxes_to_plot) * 0.5))
    y_ticks = np.arange(len(boxes_to_plot))
    
    for i, (_, start, _, end) in enumerate(boxes_to_plot):
        ax.barh(y=i, width=end - start, left=start, height=0.4, color='skyblue', edgecolor='black')
    
    # Formatting
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{start:.1f}, {end:1f}" for _, start, _, end in boxes_to_plot])
    ax.set_xlabel("Time")
    ax.set_title("Intervals")
    ax.invert_yaxis()  # Invert to have the first interval at the top
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.show()
    
def load_audio(audio_path:str, sample_rate:int)->torch.Tensor:
    # Load the audio file
    f = open(audio_path, 'rb')
    waveform, sr = torchaudio.load(f)
    f.close()

    # Check if the audio is stereo (2 channels)
    if waveform.shape[0] > 1:
        # Convert to mono by averaging the two channels
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Resample if needed
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    
    return waveform

def post_process_without_yolo(audio_dict: dict = {}):
    
    to_whisper = []
    total_time = 0
    
    sample_rate = 16000
    seconds_per_segment = 30
    
    audios = {audio_file:load_audio(audio_file, sample_rate) for audio_file in audio_dict.keys()}
    
    for _, waveform in audios.items():
        segment_length = seconds_per_segment * sample_rate
        waveform_segments:tuple= waveform.split(segment_length, dim=1)
        
        for segment in waveform_segments:
            to_whisper.append(segment.squeeze().numpy())
            total_time += seconds_per_segment
    
    return to_whisper, total_time

def get_sorted_image_paths(folder_path: str) -> list:
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    image_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    return [os.path.join(folder_path, f) for f in image_files]

def get_YOLO_best_params(model_path: str, batch: int, n_calls: int = 20):
    model = YOLO(model_path, task="detect").eval()
    
    # Define the search space
    space = [
        Real(0.001, 0.3, name='conf'),  # Confidence threshold range
        Real(0.1, 0.6, name='iou')     # IoU threshold range
    ]

    @use_named_args(space)
    def objective(conf, iou):
        """Objective function to minimize (negative F1)."""
        metrics = model.val(batch=batch, conf=conf, iou=1, agnostic_nms=True).results_dict
        precision = metrics.get("metrics/precision(B)", 0)
        recall = metrics.get("metrics/recall(B)", 0)
        F1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        return -F1  # Minimize negative F1 to maximize F1

    # Run Bayesian optimization
    res = gp_minimize(objective, space, n_calls=n_calls, random_state=42)
    
    best_conf, best_iou = res.x
    best_F1 = -res.fun
    print(f"Best F1: {best_F1:.4f}, Best conf: {best_conf:.3f}, Best iou: {best_iou:.3f}")

    return best_conf, best_iou, best_F1

def get_YOLO_best_batch_size(model_path:str, dataset_name:str, imgsz:int, conf=0, iou=1):
    model = YOLO(model_path, task="detect").eval()
    file_names = get_sorted_image_paths(f"datasets/ami_yolo/images/{dataset_name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lower_bound = 0.85
    upper_bound = 0.95
    torch.cuda.empty_cache()
    total_memory, _ = torch.cuda.mem_get_info() # Total GPU memory
    print("Total memory",total_memory)
    batch_size = 100
    
    while True:
        try:
            # Create a dummy input with the current batch size
            batch_files = file_names[0:batch_size]
            
            # Forward pass to check memory usage
            model.predict(batch_files, batch=batch_size, 
                                    conf=conf, 
                                    iou=iou, 
                                    agnostic_nms=True,
                                    imgsz=imgsz)
            
            # Calculate current memory usage
            free_memory, _ = torch.cuda.mem_get_info()
            memory_usage = 1-(free_memory / total_memory)

            print(free_memory, total_memory, memory_usage, flush=True)
            
            # Check if memory usage is within the target range
            if lower_bound <= memory_usage <= upper_bound:
                print(f"Optimal batch size found: {batch_size}")
                print(f"Memory usage: {memory_usage * 100:.2f}%")
                return batch_size
            elif memory_usage < lower_bound:
                # Increase batch size if memory usage is too low
                batch_size += 5
            else:
                # Decrease batch size if memory usage is too high
                batch_size = max(1, batch_size - 5)
            
            # Clear memory to avoid OOM errors
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if 'out of memory' in str(e):
                # If OOM occurs, reduce batch size and clear cache
                torch.cuda.empty_cache()
                batch_size = max(1, batch_size - 5)
            else:
                raise e
            
def yolo_inference(model_path:str, dataset_name:str, imgsz:int, batch:int, conf:float, iou:int) -> Results:
    model = YOLO(model_path, task="detect").eval()
    file_names = get_sorted_image_paths(f"datasets/ami_yolo/images/{dataset_name}")
        
    total_results = []
    for i in trange(0, len(file_names), batch, desc="Yolo inference..."):
        batch_files = file_names[i:i+batch]
        results = model.predict(batch_files, batch=batch, 
                                conf=conf, 
                                iou=iou, 
                                agnostic_nms=True,
                                imgsz=imgsz)
        for r in results:
            r=r.to("cpu")
            total_results.append(r)
    
    return total_results

def post_process_yolo(resulted_objects: list[Results] = [], 
                        audio_dict: dict = {}   # {"audio_file1": [starting_Segment, ending_Segment], "audio_file2": [starting_Segment, ending_Segment],}
                    ):                          
    
    to_whisper = []
    total_time = 0
        
    sample_rate = 16000
    seconds_per_segment = 11
    
    def get_audio_file(segment_id: int, audio_dict: dict) -> str:
        for audio_file, (start, end) in audio_dict.items():
            if start <= segment_id <= end:
                return audio_file
        return None
    
    waveform = -1
    current_audio_name = ""

    for segment_id, r in enumerate(resulted_objects):
        
        boxes = r.boxes.xyxy 
        preds = r.boxes.cls
        
        if len(boxes) == 0:
            continue
        
        #sort boxes by from left to right (box1 y1 < box2 y1)
        boxes, indexes = torch.sort(boxes, dim=0)
        preds = preds[indexes]
        
        merged_boxes = [boxes[0]]
        
        #merge consecutive boxes in the same box
        # box = [x1, y1, x2, y2] with x constant
        for box in boxes[1:]:
            last_box = merged_boxes[-1]
            
            # if y1 (start) of the current box <= y2 (end) of the last box
            # then the boxes are overlapping and 
            # should be merged by updating the y2 of the last box to the max y2 of the two boxes
            # else, the boxes are not overlapping and current box should be added to the list
            if box[1] <= last_box[3]:
                merged_boxes[-1][3] = max(last_box[3], box[3])
            else:
                merged_boxes.append(box)
        
        #plot_intervals(boxes)
        #plot_intervals(merged_boxes)
        
        boxes = merged_boxes
        
        audio_name = get_audio_file(segment_id, audio_dict)
        starting_segment, ending_segment = audio_dict[audio_name]
        
        if audio_name != current_audio_name:
            waveform = load_audio(audio_name, sample_rate).squeeze(0)
            current_audio_name = audio_name
        
        for box, pred in tqdm(zip(boxes, preds), desc="Post processing..."):
            
            start = ((box[1] / 99.818181) * sample_rate) + ((segment_id - starting_segment) * seconds_per_segment * sample_rate)
            end = ((box[3] / 99.818181) * sample_rate) + ((segment_id - starting_segment) * seconds_per_segment * sample_rate)
            
            segment = waveform[int(start):int(end)]
            
            to_whisper.append(segment)
            
            total_time += segment.shape[0] / sample_rate
    
    del waveform
    gc.collect()
    
    to_whisper = torch.concatenate(to_whisper)
    
    to_whisper = to_whisper.split(30 * sample_rate, dim=0)
    
    to_whisper = [segment.numpy() for segment in to_whisper]
    
    if to_whisper[-1].shape[0] > 0:
        
        # Pad the last segment
        to_whisper[-1] = np.pad(to_whisper[-1], (0, 30 * sample_rate - to_whisper[-1].shape[0]), mode="constant")
    
    return to_whisper, total_time

def whisper_inference(yolo:bool, imgsz:int, max_size:int = -1, batch_size:int = 4, sample_rate:int = 16000, yolo_model_path:str = None):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open("datasets/ami_yolo/audio_dict_test.json", "r") as f:
        audio_dict=json.load(f)
    
    if yolo:
        # check if the best batch size and parameters are already calculated
        if os.path.exists(f"experiments/yolo_{model_name}_{image_size}.csv"):
            df = pd.read_csv(f"experiments/yolo_{model_name}_{image_size}.csv")
            batch = int(df["batch"].values[0])
            conf = float(df["conf"].values[0])
            iou = float(df["iou"].values[0])
        else:
            batch = get_YOLO_best_batch_size(yolo_model_path, "validation", imgsz=imgsz)
            conf, iou, f1 = get_YOLO_best_params(yolo_model_path, 64, 10)
            
            # save to csv
            df = pd.DataFrame({"batch": [batch], "conf": [conf], "iou":[iou] ,"f1": [f1]})
            df.to_csv(f"experiments/yolo_{model_name}_{image_size}.csv", index=False)
    
    print("Starting...")
    start_time = time.time()
    
    if yolo:
        results = yolo_inference(yolo_model_path, "test", imgsz, batch, conf, iou)
        dataset, total_time = post_process_yolo(results, audio_dict)
    else:
        dataset, total_time = post_process_without_yolo(audio_dict)
    
    print(f"Seconds of audio feed to whisper: {total_time}")
    if max_size != -1:
        dataset = dataset[:max_size]
        
    processor, model = load_whisper()
    results = []
    for i in tqdm(range(0, len(dataset), batch_size), desc="Processing Batches"):
        batch = dataset[i:i+batch_size]
        
        # Preprocess waveforms
        inputs = processor(
            batch, sampling_rate=sample_rate, return_tensors="pt", padding=True,
            return_attention_mask=True,  # Include attention mask
        )
        
        input_features = inputs.input_features.to(device)
        attention_mask = inputs.attention_mask.to(device)

        # Generate tokens
        generated_ids = model.generate(input_features, attention_mask=attention_mask)

        # Decode to text
        transcriptions = processor.batch_decode(generated_ids, skip_special_tokens=True)
        results.extend(transcriptions)

    end_time = time.time() - start_time
    
    print(f"Time taken: {end_time} seconds")
    
    return " ".join(results), end_time

def load_whisper():
    # Load the model and processor
    model_name = "openai/whisper-tiny.en"  # Replace with the desired Whisper model
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    return processor, model

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, choices=["nano", "medium"])
    parser.add_argument("--image-size", type=int, choices=[640, 1216])
    args = parser.parse_args()
    
    model_name = args.model_name
    image_size = args.image_size
    
    experiments_path = "experiments/"
    os.makedirs(experiments_path, exist_ok=True)
    
    if os.path.exists(f"{experiments_path}/whisper_no_yolo.json") == False:
        text_no_yolo, time_no_yolo = whisper_inference(False, batch_size=4)
        
        whisper_no_jolo = {"text": text_no_yolo, "time": time_no_yolo}
        
        with open(f"{experiments_path}/whisper_no_yolo.json", "w") as f:
            json.dump(whisper_no_jolo, f)
    else:
        with open(f"{experiments_path}/whisper_no_yolo.json", "r") as f:
            metrics = json.load(f)
            text_no_yolo = metrics["text"]
            time_no_yolo = metrics["time"]
    
    model_path = f"models/yolo_{model_name}_{image_size}.pt"
    text_yolo, time_yolo = whisper_inference(True, yolo_model_path=model_path, imgsz=image_size)
    
    #print(f"Text with YOLO: {text_yolo}")
    #print(f"Text without YOLO: {text_no_yolo}")
    
    #WER calculation
    start = time.time()
    wer_val = wer(text_no_yolo, text_yolo)
    end = time.time() - start
    print(f"Time taken to calculate WER: {end} seconds")
    print(f"WER: {wer_val}")
    
    #CER calculation
    start = time.time()
    cer_val = cer(text_no_yolo, text_yolo)
    end = time.time() - start
    print(f"Time taken to calculate CER: {end} seconds")
    print(f"CER: {cer_val}")    
    
    """
    #Edit distance calculation
    print(f"Edit distance complexity: O(n*m) = {len(text_yolo)} * {len(text_no_yolo)}")
    
    start = time.time()
    edit_distance = editdistance.eval(text_yolo, text_no_yolo)
    end = time.time() - start
    print(f"Time taken to calculate edit distance: {end} seconds")
    print(f"Edit distance: {edit_distance}")
    
    # normalizing the edit distance
    edit_distance = 1 - (edit_distance / max(len(text_yolo), len(text_no_yolo)))
    print(f"Normalized edit distance: {edit_distance}")
    """
    
    experiment = {}
    
    experiment["model"] = model_name
    experiment["image_size"] = image_size
    experiment["WER"] = wer_val
    experiment["CER"] = cer_val
    #experiment["Edit_distance"] = edit_distance
    experiment["Time_no_yolo"] = time_no_yolo
    experiment["Time_yolo"] = time_yolo
    
    # save to json
    experiment_name = os.path.join(experiments_path, f"experiment_{model_name}_{image_size}.json")
    with open(f"{experiments_path}/experiment_{model_name}_{image_size}.json", "w") as f:
        json.dump(experiment, f)