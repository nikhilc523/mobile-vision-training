"""
Extract key frames from video to visually inspect what's happening.
"""

import sys
import cv2
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ml.pose.yolo_loader import load_yolo, infer_keypoints_yolo


def draw_keypoints(frame, keypoints):
    """Draw keypoints on frame."""
    h, w = frame.shape[:2]
    
    # COCO keypoint connections
    connections = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
        (5, 11), (6, 12), (11, 12),  # Torso
        (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
    ]
    
    # Draw connections
    for i, j in connections:
        if keypoints[i, 2] >= 0.3 and keypoints[j, 2] >= 0.3:
            pt1 = (int(keypoints[i, 1] * w), int(keypoints[i, 0] * h))
            pt2 = (int(keypoints[j, 1] * w), int(keypoints[j, 0] * h))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
    
    # Draw keypoints
    for i in range(17):
        if keypoints[i, 2] >= 0.3:
            x = int(keypoints[i, 1] * w)
            y = int(keypoints[i, 0] * h)
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
    
    return frame


def extract_frames(video_path: str, output_dir: Path):
    """Extract key frames from video."""
    
    print("="*80)
    print("EXTRACTING KEY FRAMES")
    print("="*80)
    print()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load YOLO
    print("Loading YOLO...")
    yolo_model = load_yolo('yolo11n-pose.pt')
    print("✓ YOLO loaded")
    print()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {video_path}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print()
    
    # Key frames to extract
    key_frames = [
        1,    # Start
        29,   # First detection
        40,   # End of first segment
        68,   # Before false positive
        69,   # False positive start
        74,   # False positive peak
        78,   # False positive end
        100,  # Middle
        140,  # Person visible
        200,  # Later
        300,  # Near end
        360   # End
    ]
    
    print(f"Extracting {len(key_frames)} key frames...")
    print()
    
    frame_idx = 0
    extracted = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        
        if frame_idx in key_frames:
            # Convert to RGB for YOLO
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect keypoints
            keypoints = infer_keypoints_yolo(yolo_model, frame_rgb, 
                                             confidence_threshold=0.3, 
                                             normalize=True)
            
            valid_kps = np.sum(keypoints[:, 2] >= 0.3)
            
            # Draw keypoints on frame
            frame_with_kps = frame.copy()
            frame_with_kps = draw_keypoints(frame_with_kps, keypoints)
            
            # Add text
            text = f"Frame {frame_idx} | Keypoints: {valid_kps}/17"
            cv2.putText(frame_with_kps, text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Save frame
            output_path = output_dir / f"frame_{frame_idx:04d}.jpg"
            cv2.imwrite(str(output_path), frame_with_kps)
            
            extracted.append({
                'frame': frame_idx,
                'valid_kps': valid_kps,
                'path': output_path
            })
            
            print(f"✓ Frame {frame_idx:4d}: {valid_kps}/17 keypoints → {output_path.name}")
    
    cap.release()
    
    print()
    print("="*80)
    print("EXTRACTION COMPLETE")
    print("="*80)
    print()
    print(f"Extracted {len(extracted)} frames to: {output_dir}")
    print()
    print("Key observations:")
    print()
    
    # Analyze extracted frames
    for e in extracted:
        frame = e['frame']
        kps = e['valid_kps']
        
        if frame in [68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78]:
            print(f"  Frame {frame}: {kps}/17 keypoints - FALSE POSITIVE RANGE")
        elif kps == 0:
            print(f"  Frame {frame}: {kps}/17 keypoints - NO PERSON DETECTED")
        elif kps >= 15:
            print(f"  Frame {frame}: {kps}/17 keypoints - PERSON CLEARLY VISIBLE")
        else:
            print(f"  Frame {frame}: {kps}/17 keypoints - PARTIAL DETECTION")
    
    print()
    print("="*80)
    print()
    print("To view frames, open:")
    print(f"  {output_dir.absolute()}")
    print()


def main():
    video_path = 'data/test/2.mp4'
    output_dir = Path('data/test/frames_analysis')
    
    if not Path(video_path).exists():
        print(f"❌ Error: Video not found: {video_path}")
        return
    
    extract_frames(video_path, output_dir)


if __name__ == '__main__':
    main()

