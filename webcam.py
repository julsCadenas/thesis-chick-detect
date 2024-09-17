import os
from ultralytics import YOLO
import cv2

# to avoid lag reduce resolution and frame rate of the video stream
stream = 'http://192.168.1.2:8080/video'
cap = cv2.VideoCapture(stream)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

ret, frame = cap.read()
if not ret:
    print("Error: Could not read from video stream.")
    cap.release()
    exit()

H, W, _ = frame.shape

video_path_out = 'webcam_output.mp4'
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model_path = "C:/Users/Juls/Desktop/models/etian-last4.pt"
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    exit()

model = YOLO(model_path)

threshold = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from video stream.")
        break

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            label = f"{results.names[int(class_id)].upper()} {score * 100:.2f}%"
            cv2.putText(frame, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    out.write(frame)
    cv2.imshow('YOLO Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
