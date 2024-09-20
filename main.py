from ultralytics import YOLO

# Load your model
model = YOLO("last.pt")  

# use the model
model.train(data="dataset.yaml", epochs=64)  # train the model
model.val() # validate the model

# if using gpu uncomment this:
# if _name_ == '_main_':
#     import sys
#     from ultralytics import YOLO 

#     modelPath = 'C:/Users/Juls/Desktop/chicken/models/etian-last5.pt'
#     model = YOLO(modelPath)  
#     model.train(data="dataset.yaml", epochs=1)  