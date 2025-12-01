"""
Prepare Fall Detection Dataset for Gemini Fine-Tuning

This script extracts frames from your fall detection videos and creates
a dataset in the format required by Gemini (image + question + answer).

Usage:
    python finetune/prepare_dataset.py
"""

import cv2
import os
import json
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import shutil

# Configuration
VIDEO_DIR = "data/test"  # Directory containing your test videos
OUTPUT_DIR = "finetune/frames"  # Where to save extracted frames
DATASET_FILE = "finetune/fall_detection_dataset.json"  # Output dataset file

# Video labels (fall vs non-fall)
FALL_VIDEOS = [
    "finalfall.mp4",
    "pleasefall.mp4", 
    "outdoor.mp4",
    "2.mp4"
]

NON_FALL_VIDEOS = [
    "usinglap.mp4",
    "1.mp4",
    "trailfall.mp4",  # Too short to detect, but good negative example
    "secondfall.mp4"  # Too short to detect, but good negative example
]

# Number of frames to extract per video
FRAMES_PER_VIDEO = 10  # Adjust based on video length

# Question templates
QUESTION = "Is there a person falling in this image?"
ANSWER_FALL = "Yes, a person is falling."
ANSWER_NON_FALL = "No, the person is not falling."


def extract_frames(video_path, output_dir, label, num_frames=10):
    """
    Extract evenly spaced frames from a video.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        label: 'fall' or 'non-fall'
        num_frames: Number of frames to extract
        
    Returns:
        List of frame data dictionaries
    """
    video_name = Path(video_path).stem
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {video_path}")
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"  📹 {video_name}: {total_frames} frames, {duration:.2f}s, {fps:.1f} FPS")
    
    # Calculate frame indices to extract (evenly spaced)
    if total_frames < num_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    
    frames_data = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # Save frame
        frame_filename = f"{video_name}_frame_{idx:04d}.jpg"
        frame_path = os.path.join(output_dir, label, frame_filename)
        cv2.imwrite(frame_path, frame)
        
        # Create data entry
        frames_data.append({
            'image_path': frame_path,
            'video': video_name,
            'frame_idx': idx,
            'label': label,
            'question': QUESTION,
            'answer': ANSWER_FALL if label == 'fall' else ANSWER_NON_FALL
        })
    
    cap.release()
    print(f"  ✅ Extracted {len(frames_data)} frames")
    
    return frames_data


def create_dataset():
    """
    Create the complete dataset by extracting frames from all videos.
    """
    print("🚀 Starting dataset preparation...")
    print()
    
    # Create output directories
    os.makedirs(os.path.join(OUTPUT_DIR, 'fall'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'non_fall'), exist_ok=True)
    
    dataset = []
    
    # Process fall videos
    print("📊 Processing FALL videos:")
    for video_file in FALL_VIDEOS:
        video_path = os.path.join(VIDEO_DIR, video_file)
        if os.path.exists(video_path):
            frames = extract_frames(video_path, OUTPUT_DIR, 'fall', FRAMES_PER_VIDEO)
            dataset.extend(frames)
        else:
            print(f"  ⚠️  Video not found: {video_path}")
    
    print()
    
    # Process non-fall videos
    print("📊 Processing NON-FALL videos:")
    for video_file in NON_FALL_VIDEOS:
        video_path = os.path.join(VIDEO_DIR, video_file)
        if os.path.exists(video_path):
            frames = extract_frames(video_path, OUTPUT_DIR, 'non_fall', FRAMES_PER_VIDEO)
            dataset.extend(frames)
        else:
            print(f"  ⚠️  Video not found: {video_path}")
    
    print()
    
    # Save dataset
    print(f"💾 Saving dataset to {DATASET_FILE}...")
    with open(DATASET_FILE, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    # Create summary
    df = pd.DataFrame(dataset)
    
    print()
    print("=" * 60)
    print("✅ DATASET CREATED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Total samples: {len(dataset)}")
    print(f"Fall samples: {len(df[df['label'] == 'fall'])}")
    print(f"Non-fall samples: {len(df[df['label'] == 'non_fall'])}")
    print()
    print(f"Frames saved to: {OUTPUT_DIR}/")
    print(f"Dataset file: {DATASET_FILE}")
    print()
    
    # Show sample
    print("📋 Sample entries:")
    print(df.head(3).to_string())
    print()
    
    return dataset


def create_train_val_test_split(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split dataset into train/val/test sets.
    
    Args:
        dataset: List of data dictionaries
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
    """
    from sklearn.model_selection import train_test_split
    
    df = pd.DataFrame(dataset)
    
    # Split by video (not by frame) to avoid data leakage
    videos = df['video'].unique()
    
    # Separate fall and non-fall videos
    fall_videos = df[df['label'] == 'fall']['video'].unique()
    non_fall_videos = df[df['label'] == 'non_fall']['video'].unique()
    
    # Split fall videos
    fall_train, fall_temp = train_test_split(fall_videos, test_size=(val_ratio + test_ratio), random_state=42)
    fall_val, fall_test = train_test_split(fall_temp, test_size=test_ratio/(val_ratio + test_ratio), random_state=42)
    
    # Split non-fall videos
    non_fall_train, non_fall_temp = train_test_split(non_fall_videos, test_size=(val_ratio + test_ratio), random_state=42)
    non_fall_val, non_fall_test = train_test_split(non_fall_temp, test_size=test_ratio/(val_ratio + test_ratio), random_state=42)
    
    # Combine
    train_videos = list(fall_train) + list(non_fall_train)
    val_videos = list(fall_val) + list(non_fall_val)
    test_videos = list(fall_test) + list(non_fall_test)
    
    # Create splits
    train_df = df[df['video'].isin(train_videos)]
    val_df = df[df['video'].isin(val_videos)]
    test_df = df[df['video'].isin(test_videos)]
    
    print("📊 Dataset Split:")
    print(f"  Train: {len(train_df)} samples ({len(train_videos)} videos)")
    print(f"    - Fall: {len(train_df[train_df['label'] == 'fall'])}")
    print(f"    - Non-fall: {len(train_df[train_df['label'] == 'non_fall'])}")
    print(f"  Val: {len(val_df)} samples ({len(val_videos)} videos)")
    print(f"    - Fall: {len(val_df[val_df['label'] == 'fall'])}")
    print(f"    - Non-fall: {len(val_df[val_df['label'] == 'non_fall'])}")
    print(f"  Test: {len(test_df)} samples ({len(test_videos)} videos)")
    print(f"    - Fall: {len(test_df[test_df['label'] == 'fall'])}")
    print(f"    - Non-fall: {len(test_df[test_df['label'] == 'non_fall'])}")
    print()
    
    # Save splits
    train_df.to_json('finetune/train_split.json', orient='records', indent=2)
    val_df.to_json('finetune/val_split.json', orient='records', indent=2)
    test_df.to_json('finetune/test_split.json', orient='records', indent=2)
    
    print("💾 Splits saved:")
    print("  - finetune/train_split.json")
    print("  - finetune/val_split.json")
    print("  - finetune/test_split.json")
    print()
    
    return train_df, val_df, test_df


def create_zip_for_colab():
    """
    Create a zip file of frames for easy upload to Google Colab.
    """
    print("📦 Creating zip file for Google Colab...")
    
    zip_path = "finetune/fall_detection_frames.zip"
    shutil.make_archive(
        "finetune/fall_detection_frames",
        'zip',
        OUTPUT_DIR
    )
    
    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    
    print(f"✅ Zip file created: {zip_path}")
    print(f"   Size: {zip_size_mb:.2f} MB")
    print()
    print("📤 Upload this file to Google Colab:")
    print("   1. Open your Colab notebook")
    print("   2. Click the folder icon (left sidebar)")
    print("   3. Click 'Upload' button")
    print(f"   4. Upload {zip_path}")
    print("   5. Run: !unzip fall_detection_frames.zip -d /content/")
    print()


if __name__ == "__main__":
    # Create dataset
    dataset = create_dataset()
    
    # Create train/val/test splits
    train_df, val_df, test_df = create_train_val_test_split(dataset)
    
    # Create zip for Colab
    create_zip_for_colab()
    
    print("=" * 60)
    print("🎉 ALL DONE!")
    print("=" * 60)
    print()
    print("📝 Next steps:")
    print("  1. Upload fall_detection_frames.zip to Google Colab")
    print("  2. Open GeminiMultiModalFineTune.ipynb in Colab")
    print("  3. Follow the instructions in INSTRUCTIONS.md")
    print("  4. Modify Cell 13 to load your dataset")
    print("  5. Run the notebook!")
    print()

