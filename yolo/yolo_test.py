from post_processing import yolo_inference, get_gt_boxes_per_image, get_F1, filter_and_merge_boxes
import argparse
import pandas as pd
import os
from torchmetrics.detection import MeanAveragePrecision


parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, choices=["nano", "medium"])
parser.add_argument("--image-size", type=int, choices=[640, 1216])
args = parser.parse_args()

name = args.name
image_size = args.image_size
max_size = 3147

experiment_path = f"experiments/yolo_{name}_{image_size}.csv"

if not os.path.exists(experiment_path):
    raise FileNotFoundError(f"{experiment_path} not found")

df = pd.read_csv(experiment_path)
batch=int(df["batch"].values[0])
conf=float(df["conf"].values[0])

results = yolo_inference(model_path = f"models/yolo_{name}_{image_size}.pt", 
                dataset_name="test", conf=conf, 
                imgsz=image_size, batch=batch, max_size=max_size)
results = filter_and_merge_boxes(results, conf=conf, filter=False)

ground_truth = get_gt_boxes_per_image("test", max_size=max_size)

F1 = get_F1(results, ground_truth)

print(f"Results for YOLO {name} with image size {image_size}:")
print(f"F1: {F1:.3f}")

#save to csv using pandas
os.makedirs("tests", exist_ok=True)
df = pd.DataFrame({"F1": [F1]})
df.to_csv(f"tests/yolo_{name}_{image_size}.csv", index=False)