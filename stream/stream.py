from flask import Flask, Response, render_template
import cv2
import numpy as np
from ultralytics import YOLO

# initialize the flask application
app = Flask(__name__)

# render the webpage
@app.route('/')
def index():
    return render_template('index.html')

# load the model
modelPath = "C:/Users/Juls/Desktop/chicken/models/etian-last5.pt"
model = YOLO(modelPath)
names = model.model.names

# set distance to be considered isolation
distanceTreshold = 300

# calculate the distance between centers of each bounding box
def euclideanDistance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def genFrames():
    
    # video stream
    # stream for ip camera stream and webcam to use webcam video stream
    stream = 'http://192.168.1.6:8080/video'
    webcam = 0
    cap = cv2.VideoCapture(stream)
    
    # print this if ip camera is not found or if cant open camera
    if not cap.isOpened():
        print("ERROR: Could not open video stream")
        exit()
    
    # if camera is open
    while cap.isOpened():
        success, im0 = cap.read() # im0 is the frame variable
        if not success: 
            print("Video frame is empty or video processing is completed")
            break
        
        results = model(im0) # run the model on the frame
        centers = [] # store the center of each bounding box to an array
        boundingBoxes = [] # store the bounding boxes of each detection to an array
        
        for result in results:
            boxes = result.boxes # bounding boxes
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0] # bounding box coordinates
                cls = int(box.cls[0]) # class value 
                conf = box.conf[0] # confidence value
                
                # calculate centers of the bounding boxes
                centerX = int((x1 + x2) / 2)
                centerY = int((y1 + y2) / 2)
                
                centers.append((centerX, centerY)) # populate the array with the calculated centers
                boundingBoxes.append((x1, y1, x2, y2)) # populate the array with the detected bounding boxes
                
                cv2.rectangle(im0, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2) # draw the bounding boxes
                cv2.circle(im0, (centerX, centerY), 5, (0, 255, 0), -1) # put circles in the centers of each bounxing box
                cv2.putText(im0, f"{names[cls]} {conf:.2f}", (int(x1), int(y1) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2) # print the detected class name and confidence percentage
                
            isolatedFlags = [True] * len(centers) # initialize the isolation
            
            for i in range(len(centers)):
                for j in range(len(centers)):
                    if i != j: # dont compare a chicken to itself
                        dist = euclideanDistance(centers[i], centers[j]) # calculate the distances between centers
                        
                        # marks chickens less than the threshold as not isolated
                        if dist < distanceTreshold:
                            isolatedFlags[i] = False
                            isolatedFlags[j] = False
                            cv2.line(im0, centers[i], centers[j], (0, 0, 255), 2) # draws the line between chickens that are close to each other  
                            
                        # prints the distances between centers
                        midPoint = ((centers[i][0] + centers[j][0]) // 2, (centers[i][1] + centers[j][1]) // 2)
                        cv2.putText(im0, f"{dist:.2f} px", midPoint, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # checks for isolation
            for idx, (center, bbox, isolated) in enumerate(zip(centers, boundingBoxes, isolatedFlags)):
                if isolated:
                    cv2.putText(im0, "Isolated", (int(bbox[0]), int(bbox[1]) - 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2) # prints isolation mark on top of the bounding boxes
                    cv2.rectangle(im0, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2) # draws red bounding boxes on isolated chickens
                    
                    
            ret, buffer = cv2.imencode('.jpg', im0) # encodes the frames into jpg
            if not ret:
                print("Error: Frame encoding failed")
                continue
            
            frame = buffer.tobytes() # converts the encoded frames into bytes
            
            # yield to generate the frames continuously and not all at once like return
            yield(b'--frame\r\n'
                  b'Content=Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# render/set the route for the video stream
@app.route('/video_feed')
def video_feed():
    return Response(genFrames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# initialize the website
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)     