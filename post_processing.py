import time
import torch
import torchaudio
from transformers import pipeline
from ultralytics.engine.results import Results
from ultralytics import YOLO
import os
import gc

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

#whisper = pipeline("automatic-speech-recognition", "openai/whisper-large-v3", torch_dtype=torch.float16, device="cuda:0")

# transcription = whisper("./datasets/ami/audio/EN2001a.wav")
# print(transcription["text"])

def post_process_without_yolo(resulted_objects: list[Results] = []):
    pass

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
        results = model.predict(batch_files, batch=batch, conf=0.000001)
        for r in results:
            r=r.to("cpu")
            total_results.append(r)
    
    return total_results

def post_process_yolo(resulted_objects: list[Results] = [], 
                        audio_dict: dict = {}   # {"audio_file1": [starting_Segment, ending_Segment], "audio_file2": [starting_Segment, ending_Segment],}
                    ):                          
    
    sample_rate = 16000
    seconds_per_segment = 11
    whisper = pipeline("automatic-speech-recognition", "openai/whisper-large-v3", torch_dtype=torch.float16, device="cuda:0")

    def load_audio(audio_path):
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
    
    def get_audio_file(segment_id: int, audio_dict: dict) -> str:
        for audio_file, (start, end) in audio_dict.items():
            if start <= segment_id <= end:
                return audio_file
        return None
    
    audios = {audio_file:load_audio(audio_file) for audio_file in audio_dict.keys()}

    for segment_id, r in enumerate(resulted_objects):
        
        boxes = r.boxes.xyxy 
        preds = r.boxes.cls
        
        audio_name = get_audio_file(segment_id, audio_dict)
        starting_segment, ending_segment = audio_dict[audio_name]
        waveform = audios[audio_name]
        
        for box, pred in zip(boxes, preds):
            start = ((box[0] / 99.818181) * sample_rate) + ((segment_id - starting_segment) * seconds_per_segment * sample_rate)
            end = ((box[2] / 99.818181) * sample_rate) + ((segment_id - starting_segment) * seconds_per_segment * sample_rate)
            segment = waveform[int(start):int(end)]
            
            transcription = whisper(segment.numpy())
            
            print(transcription)

if __name__ == "__main__":
    
    results = yolo_inference("runs/detect/train/weights/best.pt")
    
    import json
    
    with open("datasets/ami_yolo/audio_dict_test.json", "r") as f:
        audio_dict=json.load(f)
    
    post_process_yolo(results, audio_dict)