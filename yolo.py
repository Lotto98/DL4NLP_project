from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")

# Train the model
train_results = model.train(
    data="datasets/ami_yolo/ami.yaml",  # path to dataset YAML
    epochs=500,  # number of training epochs
    device="1",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
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