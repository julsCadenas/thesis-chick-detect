# deletes files from a list of filenames inside a text file

import os

def deleteLabels(directory, fileList):
    if not os.path.exists(directory):
        print(f"{directory} does not exist.")
        return
    
    if not os.path.isfile(fileList):
        print(f"{fileList} does not exist.")
        return

    with open(fileList, 'r') as f:
        filesDelete = f.read().splitlines()

    for fileName in filesDelete:
        filePath = os.path.join(directory, fileName)
        if os.path.isfile(filePath):
            try:
                os.remove(filePath)
                print(f"deleted {fileName}")
            except Exception as e:
                print(f"error deleting {fileName}: {e}")
        else:
            print(f"{fileName} not found")


folderPath = "C:/Users/Juls/Desktop/dataset(9-22)/labels/train"
listPath = "C:/Users/Juls/Desktop/delete.txt"

deleteLabels(folderPath, listPath)
