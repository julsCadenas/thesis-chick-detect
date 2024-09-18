from flask import Flask, Response, render_template
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html') 

# Load the YOLO model
model = YOLO("C:/Users/Juls/Desktop/models/etian-last4.pt")
names = model.model.names

# Function to calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def generate_frames():
    # Open the video stream from the IP camera
    stream = 'http://192.168.53.43:8080/video'
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        exit()

    # Process the video frame by frame
    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break

        # Run YOLO model on the frame
        results = model(im0)

        # List to store object centers
        centers = []

        # Loop through each detection in the results
        for result in results:
            boxes = result.boxes  # Get the bounding boxes

            for box in boxes:
                # Extract bounding box coordinates and class index
                x1, y1, x2, y2 = box.xyxy[0]  # Bounding box coordinates
                cls = int(box.cls[0])  # Class index
                conf = box.conf[0]  # Confidence score

                # Calculate the center of the bounding box
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                # Append the center point to the list
                centers.append((center_x, center_y))

                # Draw the bounding box and center point on the frame
                cv2.rectangle(im0, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.circle(im0, (center_x, center_y), 5, (0, 255, 0), -1)
                cv2.putText(im0, f"{names[cls]} {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)

        # Calculate distances and draw lines between all object centers
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                # Calculate distance between center i and center j
                dist = euclidean_distance(centers[i], centers[j])

                # Draw a line between the two center points
                cv2.line(im0, centers[i], centers[j], (0, 0, 255), 2)  # Red line with thickness 2

                # Draw the distance on the frame at the midpoint of the line
                mid_point = ((centers[i][0] + centers[j][0]) // 2, (centers[i][1] + centers[j][1]) // 2)
                cv2.putText(im0, f"{dist:.2f} px", mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', im0)
        frame = buffer.tobytes()

        # Yield the frame as part of a multipart response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
