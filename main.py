import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker
import torch  # Import torch for checking GPU
# Check for GPU availability and set the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load YOLO model on the appropriate device
model = YOLO('models/yolov8s.pt').to(device)

# Video sources: 4 "in" traffic and 4 "out" traffic videos
caps = []
for i in range(1, 5):
    cap = cv2.VideoCapture(f'static/videos/In{i}.mp4')
    if not cap.isOpened():
        print(f"Error: Couldn't open video In{i}.mp4")
    caps.append(cap)

for i in range(1, 5):
    cap = cv2.VideoCapture(f'static/videos/Out{i}.mp4')
    if not cap.isOpened():
        print(f"Error: Couldn't open video Out{i}.mp4")
    caps.append(cap)

# Read class names from COCO dataset
with open("coco.txt", "r") as my_file:
    data = my_file.read()
class_list = data.split("\n")

# Create trackers for each video feed
trackers = [Tracker() for _ in range(8)]

# Define bounding box regions for each video feed
frame_width = 1020
frame_height = 600
margin = 40

regions = [
    ((margin, margin), (frame_width - margin, frame_height - margin)),  # Region for Video 1
    ((margin, margin), (frame_width - margin, frame_height - margin)),  # Region for Video 2
    ((margin, margin), (frame_width - margin, frame_height - margin)),  # Region for Video 3
    ((margin, margin), (frame_width - margin, frame_height - margin)),  # Region for Video 4
    ((margin, margin), (frame_width - margin, frame_height - margin)),  # Region for Video 5
    ((margin, margin), (frame_width - margin, frame_height - margin)),  # Region for Video 6
    ((margin, margin), (frame_width - margin, frame_height - margin)),  # Region for Video 7
    ((margin, margin), (frame_width - margin, frame_height - margin))   # Region for Video 8
]


def is_inside_region(center, region):
    (x1, y1), (x2, y2) = region
    return x1 <= center[0] <= x2 and y1 <= center[1] <= y2


vehicle_count = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0}

def generate_frames(video_id):
    tracker = trackers[video_id - 1]
    cap = caps[video_id - 1]
    region = regions[video_id - 1]

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"End of video feed {video_id} or error.")
            break

        frame = cv2.resize(frame, (1020, 600))

        # Predict with the YOLO model
        results = model.predict(frame)
        a = results[0].boxes.data
        px = pd.DataFrame(a.cpu()).astype("float")

        bbox_list = []
        for _, row in px.iterrows():
            x1, y1, x2, y2 = map(int, row[:4])
            d = int(row[5])
            bbox_list.append([x1, y1, x2, y2])

        bbox_id = tracker.update(bbox_list)
        count = 0  # Vehicle count for this frame

        for bbox in bbox_id:
            x3, y3, x4, y4, id = bbox
            center_x, center_y = (x3 + x4) // 2, (y3 + y4) // 2
            center = (center_x, center_y)

            # Count vehicles if they are inside the defined region
            if is_inside_region(center, region):
                count += 1
                vehicle_count[video_id] = count
        
        print(f"Video {video_id} - Count: {count}\n {vehicle_count}")

        # Draw the bounding box region and display count
        cv2.rectangle(frame, region[0], region[1], (255, 255, 255), 2)
        cv2.putText(frame, f'Count: {count}', (region[0][0], region[0][1] - 10),
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

        # Encode the frame to JPEG format for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')