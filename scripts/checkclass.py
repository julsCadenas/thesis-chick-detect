# this script checks if the classes set in your labels are 0, since the only label set in this project is chicken (index of 0)
import os

# director of the labels in your dataset
annotationsDir = "path/to/your/datasets" #replace with /val and /test after 

print(f"checking {annotationsDir}")
for filename in os.listdir(annotationsDir):
    if filename.endswith('.txt'):
        with open(os.path.join(annotationsDir, filename), 'r') as file:
            lines = file.readlines()
            for line in lines:
                firstChar = line.strip()[0] if line.strip() else None
                if firstChar != '0':
                    print(f"{filename} has other classes") # prints the filename of files with incorrect classes