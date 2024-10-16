import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("C:/Users/Juls/Downloads/last.pt")
names = model.model.names

# Open the video file
cap = cv2.VideoCapture("C:/Users/Juls/Desktop/vids/chickentry.mp4")
assert cap.isOpened(), "Error reading video file"

# Get video properties: width, height, and frames per second
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Initialize the video writer
video_writer = cv2.VideoWriter("distance_calculation.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
assert video_writer.isOpened(), "Error opening video writer"

# Function to calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

# Set distance threshold for isolation (in pixels)
distance_threshold = 300  # Lowered threshold to test proximity

# Process the video frame by frame
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    # Run YOLO model on the frame
    results = model(im0)

    # List to store object centers and bounding boxes
    centers = []
    bounding_boxes = []

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

            # Append the center point and bounding box to the lists
            centers.append((center_x, center_y))
            bounding_boxes.append((x1, y1, x2, y2))

            # Draw the bounding box and center point on the frame
            cv2.rectangle(im0, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.circle(im0, (center_x, center_y), 5, (0, 255, 0), -1)
            cv2.putText(im0, f"{names[cls]} {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)

    # Initialize isolation status for all detected chickens
    isolated_flags = [True] * len(centers)  # Start with all chickens isolated

    # Compare distances between each pair of chickens
    for i in range(len(centers)):
        for j in range(len(centers)):
            if i != j:  # Don't compare a chicken with itself
                # Calculate distance between center i and center j
                dist = euclidean_distance(centers[i], centers[j])


                # If any chicken is within the threshold distance, mark them as not isolated
                if dist < distance_threshold:
                    isolated_flags[i] = False
                    isolated_flags[j] = False

                # Draw a line between chickens that are close to each other
                if dist < distance_threshold:
                    cv2.line(im0, centers[i], centers[j], (0, 0, 255), 2)  # Red line with thickness 2

                # Draw the distance on the frame at the midpoint of the line
                mid_point = ((centers[i][0] + centers[j][0]) // 2, (centers[i][1] + centers[j][1]) // 2)
                cv2.putText(im0, f"{dist:.2f} px", mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Indicate isolated chickens
    for idx, (center, bbox, isolated) in enumerate(zip(centers, bounding_boxes, isolated_flags)):
        if isolated:
            # If the chicken is isolated, display "Isolated" near the bounding box
            cv2.putText(im0, "Isolated", (int(bbox[0]), int(bbox[1]) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255), 2)

    # Write the processed frame to the output video file
    video_writer.write(im0)

# Release resources
cap.release()
video_writer.release()
cv2.destroyAllWindows()