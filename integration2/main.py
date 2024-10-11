from flask import Flask, render_template, Response
import cv2
import numpy as np
import os
import csv
from datetime import datetime

app = Flask(__name__)

# Load Haar Cascade face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Known temperature values for pixel values
knownTemperature = np.array([10, 20, 30, 40])
pixelValues = np.array([30, 100, 150, 255])

# Folder to save images
save_folder = 'saved_frames'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# CSV file to log temperatures
csvFile = 'temperaturelog.csv'
if not os.path.exists(csvFile):
    with open(csvFile, mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Frame", "Temperature"])

def pixelToTemperature(pixelValue):
    return np.interp(pixelValue, pixelValues, knownTemperature)

# Open webcam (RGB feed) and thermal camera
webcam = cv2.VideoCapture(1)
thermalCamera = cv2.VideoCapture(0)

distanceThreshold = 300

def euclideanDistance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# Video generator for the webcam feed with face detection
# Video generator for the webcam feed with face detection
def webcam_stream():
    while True:
        retWebcam, frameWebcam = webcam.read()
        if not retWebcam:
            break

        # Perform face detection using Haar Cascade
        gray = cv2.cvtColor(frameWebcam, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        isolatedFlags = [True] * len(faces)  # Set up isolation flags based on number of detections
        temperatures = []

        for (x, y, w, h) in faces:
            # Draw a rectangle around the detected face
            cv2.rectangle(frameWebcam, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Map bounding box to thermal camera feed
            thermalX1 = int(x * (thermalCamera.get(cv2.CAP_PROP_FRAME_WIDTH) / frameWebcam.shape[1]))
            thermalY1 = int(y * (thermalCamera.get(cv2.CAP_PROP_FRAME_HEIGHT) / frameWebcam.shape[0]))
            thermalX2 = int((x + w) * (thermalCamera.get(cv2.CAP_PROP_FRAME_WIDTH) / frameWebcam.shape[1]))
            thermalY2 = int((y + h) * (thermalCamera.get(cv2.CAP_PROP_FRAME_HEIGHT) / frameWebcam.shape[0]))

            thermalCamera.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Ensure we use the first frame for thermal

            retThermal, frameThermal = thermalCamera.read()
            if retThermal:
                thermalImage = cv2.cvtColor(frameThermal, cv2.COLOR_BGR2GRAY)
                if (0 <= thermalX1 < thermalImage.shape[1] and 0 <= thermalY1 < thermalImage.shape[0]
                        and 0 <= thermalX2 < thermalImage.shape[1] and 0 <= thermalY2 < thermalImage.shape[0]):
                    boundingBoxThermal = thermalImage[thermalY1:thermalY2, thermalX1:thermalX2]
                    maxPixelValue = np.max(boundingBoxThermal)
                    faceTemperature = pixelToTemperature(maxPixelValue)
                    temperatures.append(faceTemperature)

                    # Save frame if temperature exceeds 35°C
                    if faceTemperature > 35:
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = os.path.join(save_folder, f'frame_{timestamp}.jpg')
                        cv2.imwrite(filename, frameWebcam)

                        # Log temperature in CSV
                        with open(csvFile, mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([timestamp, filename, faceTemperature])

                else:
                    temperatures.append(None)
            else:
                temperatures.append(None)

        # Draw isolation and temperature info on the frame
        for idx, ((x, y, w, h), isolated) in enumerate(zip(faces, isolatedFlags)):
            # If isolated, mark the face with a red box
            if isolated:
                cv2.putText(frameWebcam, "Isolated", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.rectangle(frameWebcam, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Display temperature if available
            if temperatures[idx] is not None:
                cv2.putText(frameWebcam, f'Temp: {temperatures[idx]:.2f} C', (x, y - 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Encode the frame and send it as an HTTP response
        ret, jpeg = cv2.imencode('.jpg', frameWebcam)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


# Thermal stream generator for visualization with bounding boxes
def thermal_stream():
    while True:
        retWebcam, frameWebcam = webcam.read()
        retThermal, frameThermal = thermalCamera.read()
        if not retThermal or not retWebcam:
            break

        height, width, _ = frameThermal.shape
        frameThermal = frameThermal[:height // 2, :]

        thermalImage = cv2.cvtColor(frameThermal, cv2.COLOR_BGR2GRAY)
        thermalImageNormalized = cv2.normalize(thermalImage, None, 0, 255, cv2.NORM_MINMAX)
        thermalImageColored = cv2.applyColorMap(thermalImageNormalized, cv2.COLORMAP_JET)

        # Perform face detection on the webcam feed
        gray = cv2.cvtColor(frameWebcam, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            thermalX1 = int(x * (thermalCamera.get(cv2.CAP_PROP_FRAME_WIDTH) / frameWebcam.shape[1]))
            thermalY1 = int(y * (thermalCamera.get(cv2.CAP_PROP_FRAME_HEIGHT) / frameWebcam.shape[0]))
            thermalX2 = int((x + w) * (thermalCamera.get(cv2.CAP_PROP_FRAME_WIDTH) / frameWebcam.shape[1]))
            thermalY2 = int((y + h) * (thermalCamera.get(cv2.CAP_PROP_FRAME_HEIGHT) / frameWebcam.shape[0]))

            # Draw bounding boxes on the thermal feed
            cv2.rectangle(thermalImageColored, (thermalX1, thermalY1), (thermalX2, thermalY2), (0, 255, 0), 2)

            # Extract max temperature in the bounding box
            boundingBoxThermal = thermalImage[thermalY1:thermalY2, thermalX1:thermalX2]
            if boundingBoxThermal.size > 0:
                maxPixelValue = np.max(boundingBoxThermal)
                faceTemperature = pixelToTemperature(maxPixelValue)
                cv2.putText(thermalImageColored, f'Temp: {faceTemperature:.2f} C', 
                            (thermalX1, thermalY1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        ret, jpeg = cv2.imencode('.jpg', thermalImageColored)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/webcam_feed')
def webcam_feed():
    return Response(webcam_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/thermal_feed')
def thermal_feed():
    return Response(thermal_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
