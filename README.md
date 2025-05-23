# Real-Time Traffic Congestion Detection using YOLOv5 and OpenCV

This Python script uses the YOLOv5 object detection model to monitor video feeds (webcam or file) and detect traffic congestion in real time. The results are visualized with bounding boxes and saved to a CSV file with timestamps.

## Features

- Real-time video capture and processing using OpenCV
- Object detection using YOLOv5 via PyTorch Hub
- Congestion detection based on object count for specific classes (vehicles and people)
- Live video display with bounding boxes and congestion status
- CSV logging of timestamps, congestion status, and object counts

## Requirements

- Python 3.x
- PyTorch
- OpenCV (`opencv-python`)
- YOLOv5 (via `torch.hub`)
- NumPy

## Installation

Install dependencies using pip:

```bash
pip install torch torchvision torchaudio opencv-python numpy
```

Note: YOLOv5 model will be automatically downloaded via `torch.hub`.

## How It Works

### Step-by-Step Execution

1. **Model Loading**:
   - The script loads YOLOv5s (a small and fast version of YOLOv5) using PyTorch Hub.

2. **Video Source**:
   - Uses webcam (set to 1) or accepts a file path to a video for processing.

3. **Detection**:
   - Every frame is processed to detect objects.
   - It monitors the following COCO class IDs: 
     - 0 (person), 2 (car), 5 (bus), 7 (truck)

4. **Congestion Logic**:
   - If the number of detected relevant objects exceeds a `CONGESTION_THRESHOLD` (set to 10), congestion is flagged.

5. **Logging**:
   - Every frame’s congestion status and object count are logged to `congestion_data.csv` along with a timestamp.

6. **Visualization**:
   - Bounding boxes and labels are drawn on detected objects.
   - Congestion status and count are overlaid on the frame in real time.

7. **Exit**:
   - Press `q` to quit the video stream.

## Output Files

- `congestion_data.csv`: Logs detection timestamp, congestion status, and object count.

## Notes

- Ensure your webcam is accessible if using a live feed.
- For video file input, replace `video_path = 1` with a file path like `'video.mp4'`.
- Adjust `CONGESTION_THRESHOLD` based on scene complexity and detection granularity.

## License

MIT License
