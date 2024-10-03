from flask import Flask, render_template, Response
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLO model for chicken detection
modelPath = "C:/Users/Juls/Desktop/chicken/models/etian35.pt"
model = YOLO(modelPath)
names = model.model.names

# Known temperature values for pixel values
knownTemperature = np.array([10, 20, 30, 40])
pixelValues = np.array([30, 100, 150, 255])

def pixelToTemperature(pixelValue):
    return np.interp(pixelValue, pixelValues, knownTemperature)

# Open webcam (RGB feed) and thermal camera
webcam = cv2.VideoCapture(1)
thermalCamera = cv2.VideoCapture(0)

distanceThreshold = 300

def euclideanDistance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# Initialize centers as a global variable
centers = []

# Video generator for the webcam feed with chicken detection
def webcam_stream():
    global centers  # Declare centers as global to modify it in this function
    while True:
        retWebcam, frameWebcam = webcam.read()
        if not retWebcam:
            break

        # Perform chicken detection using your YOLOv8 model
        results = model.predict(source=frameWebcam)  # Assuming model is your YOLOv8 model
        
        detections = results[0].boxes  # Assuming this is where your detection boxes are stored
        centers = []  # Reset centers for each frame
        isolatedFlags = [True] * len(detections)  # Set up isolation flags based on number of detections
        temperatures = []

        for i, box in enumerate(detections):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Convert box coordinates to integers
            centerX = (x1 + x2) // 2
            centerY = (y1 + y2) // 2
            centers.append((centerX, centerY))

            # Draw a rectangle around the detected chicken
            cv2.rectangle(frameWebcam, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Map bounding box center to thermal camera feed
            thermalX = int(centerX * (thermalCamera.get(cv2.CAP_PROP_FRAME_WIDTH) / frameWebcam.shape[1]))
            thermalY = int(centerY * (thermalCamera.get(cv2.CAP_PROP_FRAME_HEIGHT) / frameWebcam.shape[0]))
            thermalCamera.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Ensure we use the first frame for thermal

            retThermal, frameThermal = thermalCamera.read()
            if retThermal:
                thermalImage = cv2.cvtColor(frameThermal, cv2.COLOR_BGR2GRAY)
                if 0 <= thermalX < thermalImage.shape[1] and 0 <= thermalY < thermalImage.shape[0]:
                    chickenPixelValue = thermalImage[thermalY, thermalX]
                    chickenTemperature = pixelToTemperature(chickenPixelValue)
                    temperatures.append(chickenTemperature)
                else:
                    temperatures.append(None)
            else:
                temperatures.append(None)

        # Ensure the length of detections and centers match before calculating distances
        if len(detections) == len(centers):
            for i in range(len(centers)):
                for j in range(len(centers)):
                    if i != j:
                        dist = euclideanDistance(centers[i], centers[j])
                        if dist < distanceThreshold:
                            isolatedFlags[i] = False
                            isolatedFlags[j] = False

            # Draw isolation and temperature info on the frame
            for idx, (box, isolated) in enumerate(zip(detections, isolatedFlags)):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                # If isolated, mark the chicken with a red box
                if isolated:
                    cv2.putText(frameWebcam, "Isolated", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.rectangle(frameWebcam, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Display temperature if available
                if temperatures[idx] is not None:
                    cv2.putText(frameWebcam, f'Temp: {temperatures[idx]:.2f} C', (x1, y1 - 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Encode the frame and send it as an HTTP response
        ret, jpeg = cv2.imencode('.jpg', frameWebcam)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Thermal stream generator for visualization
def thermal_stream():
    global centers  # Declare centers as global to access it
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

        # Draw dots for detected chicken centers on the thermal feed
        for center in centers:
            thermalX = int(center[0] * (thermalCamera.get(cv2.CAP_PROP_FRAME_WIDTH) / frameWebcam.shape[1]))
            thermalY = int(center[1] * (thermalCamera.get(cv2.CAP_PROP_FRAME_HEIGHT) / frameWebcam.shape[0]))
            if 0 <= thermalX < thermalImageColored.shape[1] and 0 <= thermalY < thermalImageColored.shape[0]:
                cv2.circle(thermalImageColored, (thermalX, thermalY), 5, (255, 0, 0), -1)  # Draw a blue dot

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
