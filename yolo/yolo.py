from ultralytics import YOLO
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="model.pt path", required=True, 
                    choices=["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"])
parser.add_argument("--batch", type=int, help="batch size", required=True)
parser.add_argument("--epochs", type=int, help="number of training epochs", required=True)


args = parser.parse_args()

# Load a model
model = YOLO(args.model)

# Train the model
train_results = model.train(
    data="datasets/ami_yolo/ami.yaml",  # path to dataset YAML
    epochs=args.epochs,  # number of training epochs
    batch=args.batch,  # batch size
    plots=True,  # create plots
    imgsz=1216,  # image size
)