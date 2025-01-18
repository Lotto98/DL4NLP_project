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
    model = YOLO(model_path).eval()
    
    
    file_names = get_sorted_image_paths("datasets/ami_yolo/images/test")
    
    #file_names = file_names[:10]
    
    #print(file_names[0])
    
    with torch.inference_mode():
    
        results = model.predict(file_names[0:5000], stream=True, batch=1)
        
        for r in results:
            print(r)
    
    return results

def post_process_yolo(resulted_objects: list[Results] = [], 
                        audio_dict: dict = {}   # {"audio_file1": [starting_Segment, ending_Segment], "audio_file2": [starting_Segment, ending_Segment],}
                    ):                          
    
    audios = {audio_file:torchaudio.load(audio_file) for audio_file in audio_dict.keys()}

    for segment_id, r in enumerate(resulted_objects):
        h,w = r.orig_shape
        image_size_xyxy = torch.tensor([w,h,w,h], dtype=torch.float32)
        
        boxes = box_cxcywh_to_xyxy(r.boxes.tensor)  # convert from [x_c, y_c, w, h] to [x1, y1, x2, y2]
        boxes = boxes*image_size_xyxy # rescale to original image size
        
        preds = r.pred.argmax(1)
        
        for box, pred in zip(boxes, preds):
            
            print(f"Segment {segment_id}: {pred} box found at {box}")
        
        print(f"Segment {segment_id}: {preds} boxes found")

if __name__ == "__main__":
    
    yolo_inference("runs/detect/train/weights/best.pt")