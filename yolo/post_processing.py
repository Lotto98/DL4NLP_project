import time
import json
import numpy as np
import torch
import torchaudio
from ultralytics.engine.results import Results
from ultralytics import YOLO
import os
from tqdm import tqdm
import editdistance
from jiwer import wer, cer

def load_audio(audio_path, sample_rate):
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

def yolo_inference(model_path:str) -> Results:
    model = YOLO(model_path, task="detect").eval()
    file_names = get_sorted_image_paths("datasets/ami_yolo/images/test")
    batch = 512
    
    total_results = []
    
    for i in range(0, len(file_names), batch):  # Process 10 images at a time
        batch_files = file_names[i:i+batch]
        results = model.predict(batch_files, batch=batch, 
                                conf=0.25, 
                                iou=0.2, 
                                agnostic_nms=True)
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
    
    audios = {audio_file:load_audio(audio_file, sample_rate) for audio_file in audio_dict.keys()}

    for segment_id, r in enumerate(resulted_objects):
        
        boxes = r.boxes.xyxy 
        preds = r.boxes.cls
        
        #print(boxes)
        
        boxes, indexes = torch.sort(boxes, dim=0)
        preds = preds[indexes]
        
        #print(boxes)
        
        audio_name = get_audio_file(segment_id, audio_dict)
        starting_segment, ending_segment = audio_dict[audio_name]
        waveform = audios[audio_name].squeeze(0)
        
        #print(audio_name, starting_segment, ending_segment)
        
        for box, pred in zip(boxes, preds):
            
            #print(box)
            
            start = ((box[1] / 99.818181) * sample_rate) + ((segment_id - starting_segment) * seconds_per_segment * sample_rate)
            end = ((box[3] / 99.818181) * sample_rate) + ((segment_id - starting_segment) * seconds_per_segment * sample_rate)
            #print(start/sample_rate, end/sample_rate)
            segment = waveform[int(start):int(end)]
            
            to_whisper.append(segment.numpy())
            
            total_time += segment.shape[0] / sample_rate
    
    to_whisper_30 = []
    new_waveform = np.array([])
    
    for waveform in to_whisper:
        
        new_waveform = np.concatenate((new_waveform, waveform), axis=0)
        
        if new_waveform.shape[0] >= 30 * sample_rate:
            
            cut_new_waveform = new_waveform[:30 * sample_rate]
            to_whisper_30.append(cut_new_waveform)
            new_waveform = new_waveform[30 * sample_rate:]
    
    if new_waveform.shape[0] > 0:
        
        # Pad the last segment
        new_waveform = np.pad(new_waveform, (0, 30 * sample_rate - new_waveform.shape[0]), mode="constant")
        to_whisper_30.append(new_waveform)
    
    return to_whisper_30, total_time

def whisper_inference(yolo:bool, max_size:int = -1, batch_size:int = 4, sample_rate:int = 16000):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open("datasets/ami_yolo/audio_dict_test.json", "r") as f:
        audio_dict=json.load(f)
    
    print("Starting...")
    start_time = time.time()
    
    if yolo:
        results = yolo_inference("best.pt")
        dataset, total_time = post_process_yolo(results, audio_dict)
    else:
        dataset, total_time = post_process_without_yolo(audio_dict)
    
    print(f"Time feed to whisper: {total_time} seconds")
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
    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration


    # Load the model and processor
    model_name = "openai/whisper-tiny.en"  # Replace with the desired Whisper model
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    return processor, model

if __name__ == "__main__":
    text_yolo, time_yolo = whisper_inference(True)
    text_no_yolo, time_no_yolo = whisper_inference(False)
    
    print(f"Text with YOLO: {text_yolo}")
    print(f"Text without YOLO: {text_no_yolo}")
    
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
    