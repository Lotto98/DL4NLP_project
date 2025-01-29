from ultralytics import YOLO
import argparse
import os

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="model.pt path", required=True, 
                    choices=["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt", "resume"])
parser.add_argument("--batch", type=int, help="batch size", required=True)
parser.add_argument("--epochs", type=int, help="number of training epochs", required=True)
parser.add_argument("--imgsz", type=int, help="image size", default=1216)

parser.add_argument("--path-model", type=str, help="custom model.pt path")

args = parser.parse_args()

if args.model == "resume":
    if args.path_model is None:
        raise ValueError("Please provide a model.pt path with --path-model")
    
    if not os.path.exists(args.path_model):
        raise FileNotFoundError(f"Model file not found: {args.path_model}")
    
    args.model = args.path_model
    resume = True
else:
    resume = False

# Load a model
model = YOLO(args.model)

# Train the model
train_results = model.train(
    data="datasets/ami_yolo/ami.yaml",  # path to dataset YAML
    epochs=args.epochs,  # number of training epochs
    batch=args.batch,  # batch size
    plots=True,  # create plots
    imgsz=args.imgsz,  # image size
    resume=resume,  # resume training
)