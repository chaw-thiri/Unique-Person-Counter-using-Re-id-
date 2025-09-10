from ultralytics import YOLO
import cv2
import os
import yaml
import numpy as np

# TODO: Make sure ReID work properly 

# Define a custom BoTSORT configuration with ReID enabled
botsort_config = {
    'tracker_type': 'botsort',
    'track_high_thresh': 0.5,  # Detection confidence threshold
    'track_low_thresh': 0.1,   # Minimum confidence for tracking
    'new_track_thresh': 0.6,   # Threshold for initializing new tracks
    'track_buffer': 300,       # Buffer to tolerate occlusion (as recommended)
    'match_thresh': 0.8,       # Matching threshold
    'proximity_thresh': 0.2,   # Proximity threshold (IoU-based in native implementation)
    'appearance_thresh': 0.3,  # Appearance similarity threshold
    'with_reid': True,         # Enable re-identification
    'fuse_score': True,        # Fuse detection scores with IoU/ReID
    'gmc_method': 'sparseOptFlow',  # Global motion compensation method
    'model': 'yolo11n.pt'      # Explicitly specify the model to avoid cfg.model error
}

# Save the configuration to a temporary YAML file
config_path = "botsort_reid.yaml"
with open(config_path, 'w') as f:
    yaml.dump(botsort_config, f)

# Load YOLO model
model = YOLO("yolo11n.pt")

# Check Ultralytics version
import ultralytics
print(f"Ultralytics version: {ultralytics.__version__}")

# Video capture
video_path = r"C:\Users\chawt\Desktop\human detection\single man.mp4"


cap = cv2.VideoCapture(video_path)
original_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Original FPS: {original_fps}")
# Output folder for saving frames
output_folder = "output_frames"
os.makedirs(output_folder, exist_ok=True)

try:
    # Initialize tracking with ReID
    results_generator = model.track(
        source=video_path,
        tracker=config_path,  # Use custom BoTSORT config with ReID
        conf=0.5,            # Confidence threshold
        classes=[0],         # Track only persons (class 0)
        iou=0.3,             # IoU threshold for tracking
        stream=True,
        verbose=False
    )

    # Keep track of unique IDs
    seen_ids = set()
    frame_idx = 0  # Overall frame index
    process_every_n = 3 # Process every nth frame

    for result in results_generator:
        # Only process YOLO + tracking every nth frame
        if frame_idx % process_every_n != 0:
            frame_idx += 1
            continue

        frame = result.orig_img.copy()
        annotated = result.plot()

        # Extract tracked IDs
        if result.boxes.id is not None:
            ids = result.boxes.id.int().cpu().tolist()
            for pid in ids:
                seen_ids.add(pid)
        

        # Overlay stats on frame
        cv2.putText(annotated, f"Unique persons: {len(seen_ids)}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Save frame
        output_path = os.path.join(output_folder, f"frame_{frame_idx:06d}.jpg")
        cv2.imwrite(output_path, annotated)

        # Live preview (resize to fit screen, e.g., 600px width)
        display_width = 600
        scale = display_width / annotated.shape[1]
        display_height = int(annotated.shape[0] * scale)
        resized = cv2.resize(annotated, (display_width, display_height))

        cv2.imshow("Tracking", resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        

        frame_idx += 1

except AttributeError as e:
    print(f"Error: {e}")
    print("ReID may not be supported in your Ultralytics version. Please ensure Ultralytics >= 8.3.114.")
    print("Falling back to tracking without ReID.")
    # Fallback to tracking without ReID
    botsort_config['with_reid'] = False
    with open(config_path, 'w') as f:
        yaml.dump(botsort_config, f)
    
    results_generator = model.track(
        source=video_path,
        tracker=config_path,
        conf=0.5,
        classes=[0],
        iou=0.3,
        stream=True,
        verbose=False
    )

    seen_ids = set()
    frame_idx = 0

    for result in results_generator:
        if frame_idx % process_every_n != 0:
            frame_idx += 1
            continue

        frame = result.orig_img.copy()
        annotated = result.plot()

        if result.boxes.id is not None:
            ids = result.boxes.id.int().cpu().tolist()
            for pid in ids:
                seen_ids.add(pid)

        cv2.putText(annotated, f"Unique persons: {len(seen_ids)}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        output_path = os.path.join(output_folder, f"frame_{frame_idx:06d}.jpg")
        cv2.imwrite(output_path, annotated)

        # Live preview (resize to fit screen, e.g., 600px width)
        display_width = 600
        scale = display_width / annotated.shape[1]
        display_height = int(annotated.shape[0] * scale)
        resized = cv2.resize(annotated, (display_width, display_height))

        cv2.imshow("Tracking", resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1


# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()

# Recreate video from saved frames
output_video_path = "output_video.mp4"
images = sorted([f for f in os.listdir(output_folder) if f.endswith(".jpg")])

if images:
    first_img = cv2.imread(os.path.join(output_folder, images[0]))
    height, width, _ = first_img.shape
    size = (width, height)
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), original_fps, size)

    for filename in images:
        img = cv2.imread(os.path.join(output_folder, filename))
        out.write(img)
    out.release()

# Clean up temporary config file
if os.path.exists(config_path):
    os.remove(config_path)

print(f"✅ Video created successfully: {output_video_path}")
print(f"👥 Total unique persons detected: {len(seen_ids)}")