from ultralytics.models import YOLO
import argparse
import pandas as pd
import os

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, choices=["nano", "medium"])
parser.add_argument("--image-size", type=int, choices=[640, 1216])
args = parser.parse_args()

name = args.name
image_size = args.image_size

experiment_path = f"experiments/yolo_{name}_{image_size}.csv"

if not os.path.exists(experiment_path):
    raise FileNotFoundError(f"{experiment_path} not found")

df = pd.read_csv(experiment_path)
conf=float(df["conf"].values[0])
iou=float(df["iou"].values[0])

model_path = f"models/yolo_{name}_{image_size}.pt"
model = YOLO(model_path, task="detect").eval()
results = model.val(split="test", conf=conf, iou=iou).results_dict

precision = results["metrics/precision(B)"]
recall = results["metrics/recall(B)"]

F1 = 2 * (precision * recall) / (precision + recall)

print(f"Results for YOLO {name} with image size {image_size}:")
print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {F1:.3f}")

#save to csv using pandas
os.makedirs("tests", exist_ok=True)
df = pd.DataFrame({"precision": [precision], "recall": [recall], "F1": [F1]})
df.to_csv(f"tests/yolo_{name}_{image_size}.csv", index=False)