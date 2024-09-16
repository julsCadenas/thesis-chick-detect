import cv2
from ultralytics import YOLO, solutions

# Load the YOLO model
model = YOLO("C:/Users/Juls/runs/detect/train25/weights/last.pt")
names = model.model.names

# Open the video file
cap = cv2.VideoCapture("C:/Users/Juls/Desktop/vids/chickentry.mp4")
assert cap.isOpened(), "Error reading video file"

# Get video properties: width, height, and frames per second
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Initialize the video writer
video_writer = cv2.VideoWriter("distance_calculation.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
assert video_writer.isOpened(), "Error opening video writer"

# Initialize the distance calculation object
# Ensure this class exists and is properly imported
dist_obj = solutions.DistanceCalculation(names=names, view_img=True)

# Process the video frame by frame
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    # Track objects using the YOLO model
    tracks = model.track(im0, persist=True, show=False)

    # Process the frame for distance calculation
    im0 = dist_obj.start_process(im0, tracks)

    # Write the processed frame to the output video file
    video_writer.write(im0)

# Release resources
cap.release()
video_writer.release()
cv2.destroyAllWindows()