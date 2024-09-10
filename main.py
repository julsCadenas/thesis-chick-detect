from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # build a new model from scratch

# Use the model
model.train(data="dataset.yaml", epochs=10)  # train the model
model.val()