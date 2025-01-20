from ultralytics import YOLO
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="model.pt path", required=True, 
                    choices=["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"])
parser.add_argument("--device", type=int, help="cuda device, i.e. 0 or 0,1,2,3 or cpu", required=True)
parser.add_argument("--epochs", type=int, help="number of training epochs", required=True)


args = parser.parse_args()

# Load a model
model = YOLO(args.model)

# Train the model
train_results = model.train(
    data="datasets/ami_yolo/ami.yaml",  # path to dataset YAML
    epochs=args.epochs,  # number of training epochs
    device=args.device,  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    batch=0.80,  # batch size
    plots=True,  # create plots
)

# Evaluate model performance on the validation set
metrics = model.val(batch=-1)

# Perform object detection on an image
#results = model("path/to/image.jpg")
#results[0].show()

# Export the model to ONNX format
path = model.export(format="onnx")  # return path to exported model