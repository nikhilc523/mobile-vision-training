"""
Prepare URFD + Le2i Dataset for Gemini Fine-Tuning

This script extracts frames from your FULL training dataset (URFD + Le2i)
and creates a dataset in the format required by Gemini.

Usage:
    python finetune/prepare_urfd_le2i_dataset.py
"""

import cv2
import os
import json
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import shutil

# Configuration
URFD_DIR = "data/raw/urfd"  # Your URFD dataset directory
LE2I_DIR = "data/raw/le2i"  # Your Le2i dataset directory
UCF101_DIR = "data/raw/ucf101_subset"  # Your UCF101 subset directory (non-falls)
OUTPUT_DIR = "finetune/frames_full"  # Where to save extracted frames
DATASET_FILE = "finetune/fall_detection_dataset_full.json"

# Number of frames to extract per video
FRAMES_PER_VIDEO = 3  # Extract 3 frames per video (to keep dataset manageable)
# With ~400 videos × 3 frames = ~1200 frames total

# Question templates
QUESTION = "Is there a person falling in this image?"
ANSWER_FALL = "Yes, a person is falling."
ANSWER_NON_FALL = "No, the person is not falling."


def find_urfd_image_sequences():
    """
    Find all URFD image sequences and their labels.

    URFD structure:
    - data/raw/urfd/falls/fall-XX-cam0-rgb/ (contains PNG image sequences)
    - data/raw/urfd/adl/adl-XX-cam0-rgb/ (contains PNG image sequences)
    """
    sequences = []

    # Check if URFD directory exists
    if not os.path.exists(URFD_DIR):
        print(f"⚠️  URFD directory not found: {URFD_DIR}")
        return sequences

    # Find all image sequence directories
    for root, dirs, files in os.walk(URFD_DIR):
        # Check if this directory contains PNG files
        png_files = [f for f in files if f.endswith('.png') and not f.endswith('_pose.png')]

        if len(png_files) > 0:
            # This is an image sequence directory
            dir_name = os.path.basename(root)

            # Determine label from path
            if '/falls/' in root or 'fall-' in dir_name:
                label = 'fall'
            elif '/adl/' in root or 'adl-' in dir_name:
                label = 'non_fall'
            else:
                continue  # Skip unknown directories

            sequences.append({
                'path': root,
                'name': dir_name,
                'label': label,
                'dataset': 'URFD',
                'images': sorted(png_files)
            })

    return sequences


def find_le2i_videos():
    """
    Find all Le2i videos and their labels.
    
    Le2i structure:
    - Coffee_room_01/ (contains videos)
    - Home_01/ (contains videos)
    - Lecture_room_01/ (contains videos)
    Each video has annotation file with fall/non-fall labels
    """
    videos = []
    
    # Check if Le2i directory exists
    if not os.path.exists(LE2I_DIR):
        print(f"⚠️  Le2i directory not found: {LE2I_DIR}")
        return videos
    
    # Find all video files in subdirectories
    for root, dirs, files in os.walk(LE2I_DIR):
        for file in files:
            if file.endswith(('.avi', '.mp4', '.mov')):
                video_path = os.path.join(root, file)
                
                # Try to find annotation file
                annotation_file = video_path.replace('.avi', '.txt').replace('.mp4', '.txt')
                
                # Determine label
                label = 'non_fall'  # Default
                if os.path.exists(annotation_file):
                    # Parse annotation to check if there's a fall
                    with open(annotation_file, 'r') as f:
                        content = f.read()
                        if 'fall' in content.lower() or any(char.isdigit() for char in content):
                            # If annotation contains "fall" or frame numbers, it's a fall
                            label = 'fall'
                
                videos.append({
                    'path': video_path,
                    'name': file,
                    'label': label,
                    'dataset': 'Le2i'
                })
    
    return videos


def extract_frames_from_sequence(sequence_path, images, output_dir, label, sequence_name, num_frames=3):
    """
    Extract evenly spaced frames from an image sequence.

    Args:
        sequence_path: Path to image sequence directory
        images: List of image filenames in the sequence
        output_dir: Directory to save frames
        label: 'fall' or 'non_fall'
        sequence_name: Name of the sequence
        num_frames: Number of frames to extract

    Returns:
        List of frame data dictionaries
    """
    if len(images) == 0:
        return []

    # Calculate frame indices to extract (evenly spaced)
    total_frames = len(images)
    if total_frames < num_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

    frames_data = []

    for idx in frame_indices:
        # Read the image
        img_path = os.path.join(sequence_path, images[idx])
        frame = cv2.imread(img_path)

        if frame is None:
            continue

        # Save frame
        frame_filename = f"{sequence_name}_frame_{idx:04d}.jpg"
        frame_path = os.path.join(output_dir, label, frame_filename)

        # Resize frame to reduce size (512x512 is good for Gemini)
        frame_resized = cv2.resize(frame, (512, 512))
        cv2.imwrite(frame_path, frame_resized)

        # Create data entry
        frames_data.append({
            'image_path': frame_path,
            'video': sequence_name,
            'frame_idx': idx,
            'label': label,
            'question': QUESTION,
            'answer': ANSWER_FALL if label == 'fall' else ANSWER_NON_FALL
        })

    return frames_data


def extract_frames_from_video(video_path, output_dir, label, video_name, num_frames=3):
    """
    Extract evenly spaced frames from a video file.

    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        label: 'fall' or 'non_fall'
        video_name: Name of the video
        num_frames: Number of frames to extract

    Returns:
        List of frame data dictionaries
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        return []

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
        video_stem = Path(video_name).stem
        frame_filename = f"{video_stem}_frame_{idx:04d}.jpg"
        frame_path = os.path.join(output_dir, label, frame_filename)

        # Resize frame to reduce size (512x512 is good for Gemini)
        frame_resized = cv2.resize(frame, (512, 512))
        cv2.imwrite(frame_path, frame_resized)

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

    return frames_data


def find_ucf101_videos(max_videos=100):
    """
    Find UCF101 non-fall videos (limit to max_videos to keep dataset balanced).
    """
    videos = []

    if not os.path.exists(UCF101_DIR):
        print(f"⚠️  UCF101 directory not found: {UCF101_DIR}")
        return videos

    # Find all video files in subdirectories
    for root, dirs, files in os.walk(UCF101_DIR):
        for file in files:
            if file.endswith(('.avi', '.mp4', '.mov')):
                videos.append({
                    'path': os.path.join(root, file),
                    'name': file,
                    'label': 'non_fall',
                    'dataset': 'UCF101'
                })

                # Limit to max_videos
                if len(videos) >= max_videos:
                    return videos

    return videos


def create_dataset():
    """
    Create the complete dataset by extracting frames from all videos and image sequences.
    """
    print("🚀 Starting dataset preparation for URFD + Le2i + UCF101...")
    print()

    # Create output directories
    os.makedirs(os.path.join(OUTPUT_DIR, 'fall'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'non_fall'), exist_ok=True)

    # Find all data sources
    print("📂 Finding data sources...")
    urfd_sequences = find_urfd_image_sequences()
    le2i_videos = find_le2i_videos()
    ucf101_videos = find_ucf101_videos(max_videos=100)  # Limit UCF101 to 100 videos

    print(f"  URFD: {len(urfd_sequences)} image sequences")
    print(f"  Le2i: {len(le2i_videos)} videos")
    print(f"  UCF101: {len(ucf101_videos)} videos")
    print(f"  Total: {len(urfd_sequences) + len(le2i_videos) + len(ucf101_videos)} items")
    print()

    # Count fall vs non-fall
    fall_count = sum(1 for v in urfd_sequences if v['label'] == 'fall')
    fall_count += sum(1 for v in le2i_videos if v['label'] == 'fall')
    non_fall_count = sum(1 for v in urfd_sequences if v['label'] == 'non_fall')
    non_fall_count += sum(1 for v in le2i_videos if v['label'] == 'non_fall')
    non_fall_count += len(ucf101_videos)

    print(f"  Fall items: {fall_count}")
    print(f"  Non-fall items: {non_fall_count}")
    print()

    # Extract frames from all sources
    dataset = []

    print("🎬 Extracting frames from URFD image sequences...")
    for seq_info in tqdm(urfd_sequences, desc="Processing URFD"):
        frames = extract_frames_from_sequence(
            seq_info['path'],
            seq_info['images'],
            OUTPUT_DIR,
            seq_info['label'],
            seq_info['name'],
            FRAMES_PER_VIDEO
        )
        dataset.extend(frames)

    print("🎬 Extracting frames from Le2i videos...")
    for video_info in tqdm(le2i_videos, desc="Processing Le2i"):
        frames = extract_frames_from_video(
            video_info['path'],
            OUTPUT_DIR,
            video_info['label'],
            video_info['name'],
            FRAMES_PER_VIDEO
        )
        dataset.extend(frames)

    print("🎬 Extracting frames from UCF101 videos...")
    for video_info in tqdm(ucf101_videos, desc="Processing UCF101"):
        frames = extract_frames_from_video(
            video_info['path'],
            OUTPUT_DIR,
            video_info['label'],
            video_info['name'],
            FRAMES_PER_VIDEO
        )
        dataset.extend(frames)

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

    return dataset


def create_train_val_test_split(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split dataset into train/val/test sets.
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
    train_df.to_json('finetune/train_split_full.json', orient='records', indent=2)
    val_df.to_json('finetune/val_split_full.json', orient='records', indent=2)
    test_df.to_json('finetune/test_split_full.json', orient='records', indent=2)
    
    print("💾 Splits saved:")
    print("  - finetune/train_split_full.json")
    print("  - finetune/val_split_full.json")
    print("  - finetune/test_split_full.json")
    print()
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    print("=" * 60)
    print("URFD + Le2i Dataset Preparation for Gemini Fine-Tuning")
    print("=" * 60)
    print()
    
    # Create dataset
    dataset = create_dataset()
    
    if len(dataset) == 0:
        print("❌ No videos found! Please check your dataset paths:")
        print(f"   URFD_DIR: {URFD_DIR}")
        print(f"   LE2I_DIR: {LE2I_DIR}")
        print()
        print("💡 Update the paths at the top of this script to match your dataset location.")
        exit(1)
    
    # Create train/val/test splits
    train_df, val_df, test_df = create_train_val_test_split(dataset)
    
    print("=" * 60)
    print("🎉 ALL DONE!")
    print("=" * 60)
    print()
    print("📝 Next steps:")
    print("  1. Zip the frames folder:")
    print("     cd finetune && zip -r frames_full.zip frames_full/")
    print("  2. Upload frames_full.zip to Google Colab")
    print("  3. Modify Cell 13 in the notebook to load your dataset")
    print("  4. Run the fine-tuning!")
    print()
    print(f"⚠️  Note: You have {len(dataset)} frames total.")
    print("   This may be too large for Colab. Consider:")
    print("   - Reducing FRAMES_PER_VIDEO (currently 5)")
    print("   - Using a subset of videos")
    print("   - Uploading to Google Cloud Storage instead")
    print()

