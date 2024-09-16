from ultralytics import YOLO

# Load a model
model = YOLO("last.pt")  # build a new model from scratch

# # use the model-
model.train(data="dataset.yaml", epochs=64)  # train the model
model.val() # validate the model

# uncomment this if isasave mo yung model
# saving the model
# import torch
# torch.save('model1.pt')
# model.save('model1.pt')
# model.export(format="onnx")