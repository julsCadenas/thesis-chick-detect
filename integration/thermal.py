import cv2
import numpy as np

# known calibration points (modify these values based on your observations)
knownTemperature = np.array([10, 20, 30, 40])  # known temperatures in °C
pixelValues = np.array([30, 100, 150, 255])      # corresponding pixel values (example)

# interpolation function to map pixel values to temperature
def pixel_to_temperature(pixelValue):
    # Use numpy interpolation to find the temperature based on pixel value
    return np.interp(pixelValue, pixelValues, knownTemperature)

# global variable to store the cursor position
cursorX, cursorY = 0, 0

# mouse callback function to get cursor position
def mouseCallback(event, x, y, flags, param):
    global cursorX, cursorY
    if event == cv2.EVENT_MOUSEMOVE:
        cursorX, cursorY = x, y

cameraIndex = 0  
cap = cv2.VideoCapture(cameraIndex)

if not cap.isOpened():
    print("Error: Could not open the camera.")
else:
    print("Camera opened successfully.")

cv2.namedWindow('Thermal Camera')
cv2.setMouseCallback('Thermal Camera', mouseCallback)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame.")
        break

    # convert frame to grayscale for processing
    thermalImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # normalize the grayscale image to [0, 255] for better contrast
    thermalImageNormalized = cv2.normalize(thermalImage, None, 0, 255, cv2.NORM_MINMAX)

    # apply a colormap to enhance visualization
    thermalImageColored = cv2.applyColorMap(thermalImageNormalized, cv2.COLORMAP_JET)

    # get the pixel value at the cursor position
    if 0 <= cursorX < thermalImage.shape[1] and 0 <= cursorY < thermalImage.shape[0]:
        cursorPixelValue = thermalImage[cursorY, cursorX]
        cursorTemperature = pixel_to_temperature(cursorPixelValue)
        
        # display temperature reading at the top-left corner
        fontScale = 0.5
        cv2.putText(thermalImageColored, f'Temp: {cursorTemperature:.2f} °C', 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255, 255, 255), 1)

    # display the thermal image with color mapping
    cv2.imshow('Thermal Camera', thermalImageColored)

    # exit on q key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
