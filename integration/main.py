from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)

faceCascadeAlgo = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

knownTemperature = np.array([10, 20, 30, 40])
pixelValues = np.array([30, 100, 150, 255])

def pixelToTemperature(pixelValue):
    return np.interp(pixelValue, pixelValues, knownTemperature)

webcam = cv2.VideoCapture(1)  
thermalCamera = cv2.VideoCapture(0) 

distanceThreshold = 300

def euclideanDistance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# Video generator for the webcam
def webcam_stream():
    while True:
        retWebcam, frameWebcam = webcam.read()
        if not retWebcam:
            break
        
        grayWebcam = cv2.cvtColor(frameWebcam, cv2.COLOR_BGR2GRAY)
        faces = faceCascadeAlgo.detectMultiScale(grayWebcam, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        centers = []  
        isolatedFlags = [True] * len(faces)
        temperatures = [] 

        for (x, y, w, h) in faces:
            centerX = x + w // 2
            centerY = y + h // 2
            centers.append((centerX, centerY))

            cv2.rectangle(frameWebcam, (x, y), (x + w, y + h), (0, 255, 0), 2)

            thermalX = int(centerX * (thermalCamera.get(cv2.CAP_PROP_FRAME_WIDTH) / frameWebcam.shape[1]))
            thermalY = int(centerY * (thermalCamera.get(cv2.CAP_PROP_FRAME_HEIGHT) / frameWebcam.shape[0]))
            thermalCamera.set(cv2.CAP_PROP_POS_FRAMES, 0)  
            retThermal, frameThermal = thermalCamera.read()
            if retThermal:
                thermalImage = cv2.cvtColor(frameThermal, cv2.COLOR_BGR2GRAY)
                if 0 <= thermalX < thermalImage.shape[1] and 0 <= thermalY < thermalImage.shape[0]:
                    facePixelValue = thermalImage[thermalY, thermalX]
                    faceTemperature = pixelToTemperature(facePixelValue)
                    temperatures.append(faceTemperature)
                else:
                    temperatures.append(None)  
            else:
                temperatures.append(None)

        for i in range(len(centers)):
            for j in range(len(centers)):
                if i != j:
                    dist = euclideanDistance(centers[i], centers[j])  
                    # print(f"Distance between face {i} and face {j}: {dist}")  
                    if dist < distanceThreshold:  
                        isolatedFlags[i] = False
                        isolatedFlags[j] = False

        for idx, (face, isolated) in enumerate(zip(faces, isolatedFlags)):
            x, y, w, h = face
            if isolated:
                cv2.putText(frameWebcam, "Isolated", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)  
                cv2.rectangle(frameWebcam, (x, y), (x + w, y + h), (0, 0, 255), 2) 
            
            if temperatures[idx] is not None:
                cv2.putText(frameWebcam, f'Temp: {temperatures[idx]:.2f} C', (x, y - 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        ret, jpeg = cv2.imencode('.jpg', frameWebcam)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


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

        grayWebcam = cv2.cvtColor(frameWebcam, cv2.COLOR_BGR2GRAY)
        faces = faceCascadeAlgo.detectMultiScale(grayWebcam, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            faceCenterX = x + w // 2
            faceCenterY = y + h // 2

            scaleX = thermalImage.shape[1] / frameWebcam.shape[1]
            scaleY = thermalImage.shape[0] / frameWebcam.shape[0]
            thermalX = int(faceCenterX * scaleX)
            thermalY = int(faceCenterY * scaleY)

            if 0 <= thermalX < thermalImage.shape[1] and 0 <= thermalY < thermalImage.shape[0]:
                facePixelValue = thermalImage[thermalY, thermalX]
                faceTemperature = pixelToTemperature(facePixelValue)

                fontScale = 0.5
                cv2.putText(thermalImageColored, f'Temp: {faceTemperature:.2f} C',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255, 255, 255), 1)
                cv2.circle(thermalImageColored, (thermalX, thermalY), 5, (255, 0, 0), -1) 

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
