from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # build a new model from scratch

# use the model
model.train(data="dataset.yaml", epochs=2)  # train the model
model.val() # validate the model