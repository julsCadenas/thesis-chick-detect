# testing the temperature mapping from object detection to thermal camera on face recognition

import cv2
import numpy as np

# use haar cascade algorithm to identify faces
faceCascadeAlgo = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# calibration: add known temperature data and pixel values to help the thermal camera identify temperatures
knownTemperature = np.array([10, 20, 30, 40]) # in celsius
pixelValues = np.array([30, 100, 150, 255])  

# numpy interpolation to convert pixel density value to temperature
def pixelToTemperature(pixelValue):
    return np.interp(pixelValue, pixelValues, knownTemperature)

# webcam/rgb camera
webcam = cv2.VideoCapture(1)  

if not webcam.isOpened():
    print("Error: Could not open webcam.")
    exit()

# thermal camera
thermalCamera = cv2.VideoCapture(0)  

if not thermalCamera.isOpened():
    print("Error: Could not open thermal camera.")
    exit()

cv2.namedWindow('Thermal Camera')

while True:
    retWebcam, frameWebcam = webcam.read()
    retThermal, frameThermal = thermalCamera.read()

    # mileseey tr160i thermal camera includes a grayscale and green video feed, this crops the green one out
    height, width, _ = frameThermal.shape
    frameThermal = frameThermal[:height // 2, :] 
    
    if not retWebcam or not retThermal:
        print("Error: Could not read frames from cameras.")
        break

    grayWebcam = cv2.cvtColor(frameWebcam, cv2.COLOR_BGR2GRAY) # rgb to grayscale for easier detection of faces
    faces = faceCascadeAlgo.detectMultiScale(grayWebcam, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)) # detect faces

    thermalImage = cv2.cvtColor(frameThermal, cv2.COLOR_BGR2GRAY) # convert to grayscale for better pixel density clarity
    thermalImageNormalized = cv2.normalize(thermalImage, None, 0, 255, cv2.NORM_MINMAX) # normalize the images (0-255 pixel values) for better contrast
    thermalImageColored = cv2.applyColorMap(thermalImageNormalized, cv2.COLORMAP_JET) # put color into the video feed for better visualization

    for (x, y, w, h) in faces:
        cv2.rectangle(frameWebcam, (x, y), (x + w, y + h), (0, 255, 0), 2) # bounding box

        # calculate the center of the bounding box
        faceCenterX = x + w // 2
        faceCenterY = y + h // 2

        # map the coordinates from the rgb camera to the thermal camera
        scaleX = thermalImage.shape[1] / frameWebcam.shape[1]
        scaleY = thermalImage.shape[0] / frameWebcam.shape[0]
        thermalX = int(faceCenterX * scaleX)
        thermalY = int(faceCenterY * scaleY)

        # retrieve the pixel value of the coordinate
        if 0 <= thermalX < thermalImage.shape[1] and 0 <= thermalY < thermalImage.shape[0]:
            facePixelValue = thermalImage[thermalY, thermalX]
            faceTemperature = pixelToTemperature(facePixelValue)

            print(f"Detected face: ({faceCenterX}, {faceCenterY}) | pixel value {facePixelValue} | Temperature: {faceTemperature:.2f} C") # print the resutls

            # put the bounding boxes and temp data
            fontScale = 0.5
            cv2.putText(thermalImageColored, f'Temp: {faceTemperature:.2f} C', 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255, 255, 255), 1)

            cv2.putText(frameWebcam, f'Face Temp: {faceTemperature:.2f} C', 
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 255, 0), 2)

            cv2.circle(thermalImageColored, (thermalX, thermalY), 5, (255, 0, 0), -1)

    cv2.imshow('Webcam - Face Detection', frameWebcam)
    cv2.imshow('Thermal Camera', thermalImageColored)

    if cv2.waitKey(1) & 0xFF == ord('q'): # press 'q' to exit
        break

webcam.release()
thermalCamera.release()
cv2.destroyAllWindows()
