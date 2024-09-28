import cv2
import numpy as np

# Face detection setup
faceCascadeAlgo = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Thermal camera temperature calibration (modify based on actual thermal camera)
knownTemperature = np.array([10, 20, 30, 40])  # Known temperatures in °C
pixelValues = np.array([30, 100, 150, 255])    # Corresponding pixel values (example)

# Interpolation function to map pixel values to temperature
def pixel_to_temperature(pixelValue):
    return np.interp(pixelValue, pixelValues, knownTemperature)

# Initialize webcam for face detection
webcam = cv2.VideoCapture(1)  # Assuming 1 is the webcam

if not webcam.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize thermal camera
thermalCamera = cv2.VideoCapture(0)  # Assuming 0 is the thermal camera

if not thermalCamera.isOpened():
    print("Error: Could not open thermal camera.")
    exit()

cv2.namedWindow('Thermal Camera')

while True:
    # Read frame from both cameras
    ret_webcam, frame_webcam = webcam.read()
    ret_thermal, frame_thermal = thermalCamera.read()

    if not ret_webcam or not ret_thermal:
        print("Error: Could not read frames from cameras.")
        break

    # Process face detection (convert webcam frame to grayscale)
    gray_webcam = cv2.cvtColor(frame_webcam, cv2.COLOR_BGR2GRAY)
    faces = faceCascadeAlgo.detectMultiScale(gray_webcam, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process thermal camera (convert to grayscale for temperature mapping)
    thermalImage = cv2.cvtColor(frame_thermal, cv2.COLOR_BGR2GRAY)
    thermalImageNormalized = cv2.normalize(thermalImage, None, 0, 255, cv2.NORM_MINMAX)
    thermalImageColored = cv2.applyColorMap(thermalImageNormalized, cv2.COLORMAP_JET)

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Draw the bounding box on the webcam frame
        cv2.rectangle(frame_webcam, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Calculate the center of the face bounding box
        face_center_x = x + w // 2
        face_center_y = y + h // 2

        # Scale the coordinates to match the thermal camera resolution if different
        scale_x = thermalImage.shape[1] / frame_webcam.shape[1]
        scale_y = thermalImage.shape[0] / frame_webcam.shape[0]
        thermal_x = int(face_center_x * scale_x)
        thermal_y = int(face_center_y * scale_y)

        # Ensure the coordinates are valid
        if 0 <= thermal_x < thermalImage.shape[1] and 0 <= thermal_y < thermalImage.shape[0]:
            # Get the pixel value at the center of the detected face from the thermal camera
            face_pixel_value = thermalImage[thermal_y, thermal_x]
            face_temperature = pixel_to_temperature(face_pixel_value)

            # Print temperature for debugging
            print(f"Detected face at ({face_center_x}, {face_center_y}) with pixel value {face_pixel_value} -> Temperature: {face_temperature:.2f} °C")

            # Display the temperature on the thermal camera feed
            fontScale = 0.5
            cv2.putText(thermalImageColored, f'Temp: {face_temperature:.2f} °C', 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255, 255, 255), 1)

            # Display the temperature on the webcam frame for clarity
            cv2.putText(frame_webcam, f'Face Temp: {face_temperature:.2f} °C', 
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 255, 0), 2)

            # Draw a small circle (dot) on the thermal camera feed at the face center
            cv2.circle(thermalImageColored, (thermal_x, thermal_y), 5, (255, 0, 0), -1)

    # Show the webcam frame with face detection
    cv2.imshow('Webcam - Face Detection', frame_webcam)

    # Show the thermal camera feed with the temperature and dot
    cv2.imshow('Thermal Camera', thermalImageColored)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video captures and destroy windows
webcam.release()
thermalCamera.release()
cv2.destroyAllWindows()
