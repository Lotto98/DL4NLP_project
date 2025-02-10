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
import pandas as pd
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer, cer

from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

import gc

import matplotlib.pyplot as plt

from typing import List, Tuple

from ultralytics.utils.ops import xywh2xyxy

def plot_intervals(boxes_to_plot: torch.tensor, Title="Intervals"):
    """
    Plot the intervals.

    Args:
        boxes_to_plot (torch.tensor): tensor of boxes.
        Title (str, optional): Plot title. Defaults to "Intervals".
    """
    
    w = 1126 # original image size
    
    fig, ax = plt.subplots(figsize=(15, len(boxes_to_plot) + 5))
    y_ticks = np.arange(len(boxes_to_plot))
    x_ticks = np.arange(0, w, 100)
    
    for i, (_, start, _, end) in enumerate(boxes_to_plot):
        ax.barh(y=i, width=end - start, left=start, height=0.4, color='skyblue', edgecolor='black')
    
    # Formatting
    ax.set_yticks(y_ticks)
    ax.set_xticks(x_ticks)
    ax.set_yticklabels([f"{start:.1f}, {end:1f}" for _, start, _, end in boxes_to_plot])
    ax.set_xlabel("Time")
    ax.set_title(Title)
    ax.invert_yaxis()  # Invert to have the first interval at the top
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.show()
    
def load_audio(audio_path:str, sample_rate:int)->torch.Tensor:
    """
    Load the audio file and resample it if needed.

    Args:
        audio_path (str): path to the audio file.
        sample_rate (int): sample rate.

    Returns:
        torch.Tensor: audio waveform.
    """
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

def post_process_without_yolo(audio_dict: dict = {}, max_size:int = -1)->Tuple[List[np.ndarray], int]:
    """
    Post-process the audio files without YOLO segmentation.
    The audio files are split into 30-second segments.

    Args:
        audio_dict (dict, optional): starting and ending segments for each audio file.
        max_size (int, optional): maximum number of segments to process. Defaults to -1 (all).

    Returns:
        Tuple[List[np.ndarray], int]: list of audio segments (waveforms) and total time in seconds.
    """
    
    if max_size !=-1:
        max_size = (max_size * 11) / 30
    
    to_whisper = []
    total_time = 0
    
    sample_rate = 16000
    seconds_per_segment = 30
    
    audios = {audio_file:load_audio(audio_file, sample_rate) for audio_file in audio_dict.keys()}
    
    size=0
    for _, waveform in audios.items():
        segment_length = seconds_per_segment * sample_rate
        waveform_segments:tuple= waveform.split(segment_length, dim=1)
        
        for segment in waveform_segments:
            to_whisper.append(segment.squeeze().numpy())
            total_time += seconds_per_segment
            
            size += 1
            if max_size != -1 and size >= max_size:
                return to_whisper, total_time
    
    return to_whisper, total_time

def get_sorted_paths(folder_path: str, max_size:int=-1) -> list:
    """
    Get the sorted paths of the files in the folder.

    Args:
        folder_path (str): path to the folder.
        max_size (int, optional): max size. Defaults to -1 (all).

    Returns:
        list: sorted paths of the files.
    """
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    image_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    if max_size != -1:
        image_files = image_files[:max_size]
    return [os.path.join(folder_path, f) for f in image_files]

def get_YOLO_best_params(model_path: str, batch: int, max_size:int, n_calls: int = 20)->Tuple[float, float]:
    """
    Get the best confidence threshold for YOLO model using Bayesian optimization.

    Args:
        model_path (str): path to the YOLO model.
        batch (int): batch size.
        max_size (int): max number of images (segments) to process.
        n_calls (int, optional): Numbers of iteration for Bayesian optimization. Defaults to 20.

    Returns:
        Tuple[float, float]: best confidence threshold and best F1 score.
    """
    
    print("Starting Bayesian optimization to find the best confidence threshold...")
    
    # Define the search space
    space = [
        Real(0.001, 0.4, name='conf'),  # Confidence threshold range
    ]
    
    ground_truth = get_gt_boxes_per_image("validation", max_size)
    
    results = yolo_inference(model_path = model_path, 
                    dataset_name="validation", conf=0, 
                    imgsz=image_size, batch=batch, max_size=max_size)
    
    bar = tqdm(total=n_calls, desc="Bayesian optimization...")
    
    @use_named_args(space)
    def objective(conf):
        """Objective function to minimize (negative F1)."""
        
        new_results = filter_and_merge_boxes(results, conf)
        F1 = get_F1(new_results, ground_truth)
        
        bar.update(1)
        bar.set_description(f"Bayesian optimization... Conf: {conf:.3f}, F1: {F1:.4f}")
        
        return -F1  # Minimize negative F1 to maximize F1

    # Run Bayesian optimization
    res = gp_minimize(objective, space, n_calls=n_calls, random_state=42)
    
    best_conf = res.x[0]
    best_F1 = -res.fun
    print("Best Conf: {:.3f}, Best F1: {:.4f}".format(best_conf, best_F1))
    
    torch.cuda.empty_cache()

    return best_conf, best_F1

def get_YOLO_best_batch_size(model_path:str, dataset_name:str, imgsz:int, max_size:int=-1, conf=0, iou=1) -> int:
    """
    Find the optimal batch size for the YOLO model.

    Args:
        model_path (str): model path.
        dataset_name (str): split to use (e.g., "validation").
        imgsz (int): image size.
        max_size (int, optional): max size. Defaults to -1.
        conf (int, optional): conf parameter. Defaults to 0 (retain all the boxes).
        iou (int, optional): iou parameter. Defaults to 1 (disabled). 

    Returns:
        int: optimal batch size.
    """
    
    
    model = YOLO(model_path, task="detect").eval()
    file_names = get_sorted_paths(f"datasets/ami_yolo/images/{dataset_name}", max_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lower_bound = 0.85
    upper_bound = 0.90
    torch.cuda.empty_cache()
    total_memory, _ = torch.cuda.mem_get_info() # Total GPU memory
    print("Total memory",total_memory)
    batch_size = 150
    
    bar = tqdm(desc="Finding optimal batch size...", leave=True)
    
    while True:
        try:
            # Create a dummy input with the current batch size
            batch_files = file_names[0:batch_size]
            
            # Forward pass to check memory usage
            _ = model.predict(batch_files, batch=batch_size, 
                                    conf=conf, 
                                    iou=iou, 
                                    agnostic_nms=True,
                                    imgsz=imgsz,
                                    verbose=False)
            
            # Calculate current memory usage
            free_memory, _ = torch.cuda.mem_get_info()
            memory_usage = 1-(free_memory / total_memory)

            #print(free_memory, total_memory, memory_usage, flush=True)
            bar.set_description(f"Memory usage: {memory_usage * 100:.2f}%")
            
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

def get_gt_boxes_per_image(dataset_name: str, max_size:int) -> List[dict]:
    """
    Get the ground truth boxes for each image in the dataset.

    Args:
        dataset_name (str): dataset name.
        max_size (int): number of images to consider.

    Returns:
        List[dict]: list of ground truth boxes for each image.
    """
    
    labels_names = get_sorted_paths(f"datasets/ami_yolo/labels/{dataset_name}", max_size)
    gt_boxes_per_image = []
    
    h,w = 1126,166 # original image size
    resize_tensor = torch.tensor([w,h,w,h], dtype=torch.float32)
    
    for label_file in tqdm(labels_names, desc="Loading ground truth boxes..."):
        with open(label_file, "r") as f:
            labels_image = f.read().splitlines()
            
            gt_boxes = []
            for label in labels_image:
                label = label.split(" ")
                gt_box = torch.Tensor([float(label[1]), float(label[2]), float(label[3]), float(label[4])])
                gt_boxes.append(xywh2xyxy(gt_box))
            
            gt_boxes = (torch.stack(gt_boxes) * resize_tensor) if len(gt_boxes) > 0 else torch.Tensor([])
            gt_boxes = merge_boxes(gt_boxes)
            
            instance = {"boxes": gt_boxes, 
                        "labels": torch.zeros(len(gt_boxes), dtype=torch.int64)}
            
            gt_boxes_per_image.append(instance)
        
    return gt_boxes_per_image


def get_F1(predictions:List[dict], ground_truths:List[dict]) -> float:
    """
    Calculate the F1 score for the predictions and ground truth boxes using the same approach as 3MAS.

    Args:
        predictions (List[dict]): predicted boxes.
        ground_truths (List[dict]): ground truth boxes.

    Returns:
        float: F1 score.
    """
    
    predictions = [p["boxes"] for p in predictions]
    ground_truths = [g["boxes"] for g in ground_truths]

    #3MAS implementation
    #https://github.com/Lebourdais/3MAS/blob/main/eval_full.py
    
    from pyannote.metrics.detection import DetectionPrecisionRecallFMeasure
    from pyannote.core import Annotation, Segment, Timeline
    
    #Evaluate all audio
    uem = Timeline([Segment(0, 1126)])
    
    F1_scores = []
    
    for i in trange(len(predictions), desc="Calculating F1...", disable=True):
        annotation_p = Annotation()
        if len(predictions[i]) > 0:
            prediction = Segment(predictions[i][:,[1,3]][0][0].item(), predictions[i][:,[1,3]][0][1].item())
            annotation_p[prediction] = 0
        
        annotation_gt = Annotation()
        if len(ground_truths[i]) > 0:
            ground_truth = Segment(ground_truths[i][:,[1,3]][0][0].item(), ground_truths[i][:,[1,3]][0][1].item())
            annotation_gt[ground_truth] = 0
        
        metric = DetectionPrecisionRecallFMeasure()
        F1=metric.compute_metric(metric.compute_components(annotation_p, annotation_gt, uem=uem))
        F1_scores.append(F1)
        
    return np.mean(F1_scores)

def erase_unconfident_boxes(boxes: torch.Tensor, scores: torch.Tensor, conf: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Erase boxes with confidence below the threshold.

    Args:
        boxes (torch.Tensor): tensor of boxes.
        scores (torch.Tensor): tensor of scores.
        conf (float): confidence threshold.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: filtered boxes and scores.
    """
    mask = scores >= conf
    return boxes[mask], scores[mask]

def filter_and_merge_boxes(results:List[dict], conf: float, filter=True)->List[dict]:
    """
    Filter and merge boxes.

    Args:
        results (List[dict]): list of results.
        conf (float): confidence threshold.
        filter (bool, optional): Whatever to filter the boxes. Defaults to True.

    Returns:
        List[dict]: filtered and merged boxes.
    """
    new_results=[]
    for r in tqdm(results, desc="Filtering boxes...", disable=True):
        new_r = {}
        if filter:
            clean_boxes, _ = erase_unconfident_boxes(r["boxes"], r["scores"], conf)
        else:
            clean_boxes = r["boxes"]
        merged_boxes = merge_boxes(clean_boxes)
        new_r["boxes"] = merged_boxes
        new_r["labels"] = r["labels"]
        new_results.append(new_r)
    return new_results

def yolo_inference(model_path:str, dataset_name:str, imgsz:int, batch:int, conf:float, max_size:int) -> List[dict]:
    """
    Perform YOLO inference on the dataset.

    Args:
        model_path (str): path to the model.
        dataset_name (str): dataset name.
        imgsz (int): image size.
        batch (int): batch size.
        conf (float): confidence threshold.
        max_size (int): max number of images to process.

    Returns:
        List[dict]: list of results.
    """
    model = YOLO(model_path, task="detect").eval()
    images_names = get_sorted_paths(f"datasets/ami_yolo/images/{dataset_name}", max_size)
    
    count = 0
    
    total_results = []
    for i in trange(0, len(images_names), batch, desc="Yolo inference..."):
        batch_files = images_names[i:i+batch]
        results = model.predict(batch_files, batch=batch, 
                                conf=conf, 
                                iou=1, 
                                agnostic_nms=True,
                                imgsz=imgsz,
                                verbose=False)
        for r in results:
            r=r.to("cpu")
            new_r = {}
            
            boxes, scores = r.boxes.xyxy, r.boxes.conf
            new_r["boxes"] = boxes
            new_r["scores"] = scores
            new_r["labels"] = torch.zeros(len(boxes), dtype=torch.int64)
            
            total_results.append(new_r)
            count += 1
            
    return total_results

def merge_boxes(boxes: torch.Tensor)->torch.Tensor:
    """
    Merge overlapping boxes.

    Args:
        boxes (torch.Tensor): tensor of boxes.

    Returns:
        torch.Tensor: merged boxes.
    """
    
    if len(boxes) == 0:
        return boxes
    
    #plot_intervals(boxes, Title="Original boxes")
    
    #sort boxes by from left to right (box1 y1 < box2 y1)
    indexes = boxes[:, 1].argsort(dim=0, stable=True)
    boxes = boxes[indexes]
    
    #plot_intervals(boxes, Title="Sorted boxes")
    
    merged_boxes = [boxes[0]]
    
    #merge consecutive boxes in the same box
    # box = [x1, y1, x2, y2] with x constant
    for i, box in enumerate(boxes[1:]):
        
        # if y1 (start) of the current box <= y2 (end) of the last box
        # then the boxes are overlapping and 
        # should be merged by updating the y2 of the last box to the max y2 of the two boxes
        # else, the boxes are not overlapping and current box should be added to the list
        if box[1] <= merged_boxes[-1][3]:
            merged_boxes[-1][3] = max(merged_boxes[-1][3], box[3])
            
        else:
            merged_boxes.append(box)
    
    merge_boxes = torch.stack(merged_boxes) if len(merged_boxes) > 0 else torch.Tensor([])
    
    return merge_boxes

def post_process_yolo(resulted_objects: List[dict], 
                        audio_dict: dict = {}   # {"audio_file1": [starting_Segment, ending_Segment], "audio_file2": [starting_Segment, ending_Segment],}
                    ) -> Tuple[List[np.ndarray], int]:
    """
    Post-process the YOLO results before feeding them to Whisper.
    1) Extract the audio segments corresponding to the detected boxes.
    2) Split the audio segments into 30-second segments.

    Args:
        resulted_objects: (List[dict]): list of YOLO results.
        audio_dict (dict): dictionary of audio files and their starting and ending segments.
    
    Returns:
        Tuple[List[np.ndarray], int]: list of audio segments (waveforms) and total time in seconds.
    """
        
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

    for segment_id, r in tqdm(enumerate(resulted_objects), desc="Processing YOLO results...", total=len(resulted_objects)):
        
        boxes = r["boxes"]
        
        if len(boxes) == 0:
            continue
        
        audio_name = get_audio_file(segment_id, audio_dict)
        starting_segment, ending_segment = audio_dict[audio_name]
        
        if audio_name != current_audio_name:
            waveform = load_audio(audio_name, sample_rate).squeeze(0)
            current_audio_name = audio_name
        
        for box in boxes:
            
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

def whisper_inference(yolo:bool, imgsz:int, max_size:int = -1, batch_size:int = 4, sample_rate:int = 16000, yolo_model_path:str = None) -> Tuple[str, float, int]:
    """
    Perform Whisper inference on the audio files.
    If YOLO is enabled, the audio files are first processed with YOLO segmentation.
    else, the audio files are split into 30-second segments.

    Args:
        yolo (bool): Whatever to use YOLO segmentation.
        imgsz (int): image size.
        max_size (int, optional): max segments to process. Defaults to -1.
        batch_size (int, optional): whisper batch size. Defaults to 4.
        sample_rate (int, optional): sample rate of audios. Defaults to 16000.
        yolo_model_path (str, optional): Yolo model path. Defaults to None (required if YOLO is enabled).

    Returns:
        Tuple[str, float, int]: Whisper text, total execution time in seconds, and total seconds of audio feed to Whisper.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open("datasets/ami_yolo/audio_dict_test.json", "r") as f:
        audio_dict=json.load(f)
    
    if yolo:
        # check if the best batch size and parameters are already calculated
        if os.path.exists(f"experiments/yolo_{model_name}_{image_size}.csv"):
            df = pd.read_csv(f"experiments/yolo_{model_name}_{image_size}.csv")
            batch = int(df["batch"].values[0])
            conf = float(df["conf"].values[0])
        else:
            batch = get_YOLO_best_batch_size(yolo_model_path, "validation", imgsz=imgsz, max_size=max_size)
            conf, f1 = get_YOLO_best_params(yolo_model_path, batch, max_size=max_size)
            
            # save to csv
            df = pd.DataFrame({"batch": [batch], "conf": [conf], "f1": [f1]})
            df.to_csv(f"experiments/yolo_{model_name}_{image_size}.csv", index=False)
    
    print("Starting pipeline...")
    start_time = time.time()
    
    if yolo:
        results = yolo_inference(yolo_model_path, "test", imgsz, batch, conf, max_size)
        results = filter_and_merge_boxes(results, conf, filter=False)
        dataset, total_time = post_process_yolo(results, audio_dict)
        
        #clean memory
        del results
        gc.collect()
        
    else:
        dataset, total_time = post_process_without_yolo(audio_dict, max_size)
    
    print(f"Seconds of audio feed to whisper: {total_time}")
        
    processor, model = load_whisper()
    results = []
    for i in tqdm(range(0, len(dataset), batch_size), desc="Processing Whisper Batches"):
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
    
    return " ".join(results), end_time, total_time

def load_whisper()->Tuple[WhisperProcessor, WhisperForConditionalGeneration]:
    """
    Load the Whisper model and processor.

    Returns:
        Tuple[WhisperProcessor, WhisperForConditionalGeneration]: Whisper processor and model.
    """
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
    max_size = -1
    
    experiments_path = "experiments/"
    os.makedirs(experiments_path, exist_ok=True)
    
    if os.path.exists(f"{experiments_path}/whisper_no_yolo.json") == False:
        text_no_yolo, time_no_yolo, seconds_feed = whisper_inference(False, imgsz=image_size, max_size=max_size)
        
        whisper_no_jolo = {"seconds feed": seconds_feed, "text": text_no_yolo, "time": time_no_yolo}
        
        with open(f"{experiments_path}/whisper_no_yolo.json", "w") as f:
            json.dump(whisper_no_jolo, f)
    else:
        with open(f"{experiments_path}/whisper_no_yolo.json", "r") as f:
            metrics = json.load(f)
            text_no_yolo = metrics["text"]
            time_no_yolo = metrics["time"]
    
    model_path = f"models/yolo_{model_name}_{image_size}.pt"
    text_yolo, time_yolo, seconds_feed = whisper_inference(True, yolo_model_path=model_path, imgsz=image_size, max_size=max_size)
    
    #print(f"Text with YOLO: {text_yolo}")
    #print(f"Text without YOLO: {text_no_yolo}")
    
    #WER calculation
    start = time.time()
    wer_val = wer(text_no_yolo, text_yolo)
    end = time.time() - start
    print(f"Time taken to calculate WER: {end} seconds")
    
    #CER calculation
    start = time.time()
    cer_val = cer(text_no_yolo, text_yolo)
    end = time.time() - start
    print(f"Time taken to calculate CER: {end} seconds")  
    
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
    experiment["Seconds_feed"] = seconds_feed
    
    # save to json
    experiment_name = os.path.join(experiments_path, f"experiment_{model_name}_{image_size}.json")
    with open(f"{experiments_path}/experiment_{model_name}_{image_size}.json", "w") as f:
        json.dump(experiment, f)
