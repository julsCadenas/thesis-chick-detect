# this script checks if your labels are in the yolov8 format
import os

# director of the labels in your dataset
annotationsDir = 'C:/Users/Juls/Desktop/chicken/dataset/labels/train' #replace with /val and /test after 

for filename in os.listdir(annotationsDir):
    if filename.endswith('.txt'):  
        with open(os.path.join(annotationsDir, filename), 'r') as file:
            lines = file.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    print(f"Incorrect format in {filename}") #prints the file name of the file with incorrect format 
