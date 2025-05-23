import cv2
import torch
import numpy as np
import csv
import time

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Capture video (0 for webcam, or provide a video file path)
video_path = 1
cap = cv2.VideoCapture(video_path)

# Congestion threshold and monitored classes
CONGESTION_THRESHOLD = 10
MONITOR_CLASSES = [2, 5, 7, 0]

# Open CSV file and write headers
csv_filename = "congestion_data.csv"
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["timestamp", "congestion_status", "object_count"])

def check_congestion(detections):
    count = sum(1 for detection in detections if int(detection[5]) in MONITOR_CLASSES)
    return count > CONGESTION_THRESHOLD, count

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = results.xyxy[0].numpy()
    
    is_congested, object_count = check_congestion(detections)
    congestion_status = "Congestion Detected" if is_congested else "No Congestion"
    
    # Save data to CSV
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), congestion_status, object_count])
    
    # Draw bounding boxes
    for detection in detections:
        x1, y1, x2, y2, conf, class_id = detection
        label = f"{model.names[int(class_id)]} {conf:.2f}"
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display congestion status
    cv2.putText(frame, f"Status: {congestion_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Object Count: {object_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow('Congestion Monitoring', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
