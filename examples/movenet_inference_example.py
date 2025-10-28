"""
Example script demonstrating MoveNet pose estimation on fall detection datasets.

This script shows various use cases:
1. Single image inference
2. Batch processing of image sequences
3. Extracting pose features for LSTM training
4. Visualizing pose skeletons

Usage:
    python examples/movenet_inference_example.py
"""

import sys
from pathlib import Path
import numpy as np
import cv2

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml.pose.movenet_loader import (
    load_movenet,
    infer_keypoints,
    visualize_keypoints,
    KEYPOINT_NAMES,
)


def example_1_single_image():
    """Example 1: Run inference on a single image."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Single Image Inference")
    print("="*60 + "\n")
    
    # Load model
    print("Loading MoveNet model...")
    inference_fn = load_movenet()
    print()
    
    # Load test image
    image_path = Path("data/raw/urfd/falls/fall-01-cam0-rgb/fall-01-cam0-rgb-001.png")
    
    if not image_path.exists():
        print(f"⚠ Test image not found: {image_path}")
        return
    
    print(f"Loading image: {image_path.name}")
    frame = cv2.imread(str(image_path))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print(f"✓ Image size: {frame_rgb.shape[1]}x{frame_rgb.shape[0]}\n")
    
    # Run inference
    print("Running pose inference...")
    keypoints = infer_keypoints(inference_fn, frame_rgb, confidence_threshold=0.3)
    print("✓ Inference complete!\n")
    
    # Print results
    print("Detected keypoints:")
    print(f"{'Name':<15} {'Y':<8} {'X':<8} {'Confidence':<12}")
    print("-" * 50)
    
    for idx, (y, x, conf) in enumerate(keypoints):
        if conf >= 0.3:
            print(f"{KEYPOINT_NAMES[idx]:<15} {y:.4f}  {x:.4f}  {conf:.4f}")
    
    high_conf_count = np.sum(keypoints[:, 2] >= 0.3)
    print(f"\nTotal high-confidence keypoints: {high_conf_count}/17")


def example_2_batch_processing():
    """Example 2: Process multiple frames from a sequence."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Batch Processing of Image Sequence")
    print("="*60 + "\n")
    
    # Load model
    inference_fn = load_movenet()
    
    # Process first 10 frames of a fall sequence
    sequence_dir = Path("data/raw/urfd/falls/fall-01-cam0-rgb")
    
    if not sequence_dir.exists():
        print(f"⚠ Sequence directory not found: {sequence_dir}")
        return
    
    print(f"Processing sequence: {sequence_dir.name}")
    print("Processing first 10 frames...\n")
    
    results = []
    
    for i in range(1, 11):
        frame_path = sequence_dir / f"fall-01-cam0-rgb-{i:03d}.png"
        
        if not frame_path.exists():
            continue
        
        # Load frame
        frame = cv2.imread(str(frame_path))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run inference
        keypoints = infer_keypoints(inference_fn, frame_rgb, confidence_threshold=0.3)
        
        # Store results
        results.append({
            'frame_num': i,
            'keypoints': keypoints,
            'high_conf_count': np.sum(keypoints[:, 2] >= 0.3),
            'avg_confidence': np.mean(keypoints[:, 2])
        })
        
        print(f"Frame {i:3d}: {results[-1]['high_conf_count']:2d}/17 keypoints detected "
              f"(avg conf: {results[-1]['avg_confidence']:.3f})")
    
    print(f"\n✓ Processed {len(results)} frames")
    
    # Calculate statistics
    avg_keypoints = np.mean([r['high_conf_count'] for r in results])
    print(f"Average keypoints per frame: {avg_keypoints:.1f}")


def example_3_extract_pose_features():
    """Example 3: Extract pose features for LSTM training."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Extract Pose Features for LSTM")
    print("="*60 + "\n")
    
    # Load model
    inference_fn = load_movenet()
    
    # Process a sequence
    sequence_dir = Path("data/raw/urfd/falls/fall-01-cam0-rgb")
    
    if not sequence_dir.exists():
        print(f"⚠ Sequence directory not found: {sequence_dir}")
        return
    
    print(f"Extracting features from: {sequence_dir.name}")
    print("Processing first 20 frames...\n")
    
    # Collect keypoints for all frames
    sequence_keypoints = []
    
    for i in range(1, 21):
        frame_path = sequence_dir / f"fall-01-cam0-rgb-{i:03d}.png"
        
        if not frame_path.exists():
            continue
        
        # Load frame
        frame = cv2.imread(str(frame_path))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run inference
        keypoints = infer_keypoints(inference_fn, frame_rgb, confidence_threshold=0.3)
        
        # Flatten keypoints to feature vector (17 keypoints × 3 values = 51 features)
        feature_vector = keypoints.flatten()
        sequence_keypoints.append(feature_vector)
    
    # Convert to numpy array
    sequence_features = np.array(sequence_keypoints)
    
    print(f"✓ Extracted features from {len(sequence_keypoints)} frames")
    print(f"Feature matrix shape: {sequence_features.shape}")
    print(f"  - Frames: {sequence_features.shape[0]}")
    print(f"  - Features per frame: {sequence_features.shape[1]} (17 keypoints × 3)")
    print("\nThis feature matrix can be used as input to an LSTM model for fall detection.")


def example_4_visualize_poses():
    """Example 4: Visualize pose skeletons on frames."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Visualize Pose Skeletons")
    print("="*60 + "\n")
    
    # Load model
    inference_fn = load_movenet()
    
    # Process and visualize frames
    sequence_dir = Path("data/raw/urfd/falls/fall-01-cam0-rgb")
    output_dir = Path("examples/output/pose_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not sequence_dir.exists():
        print(f"⚠ Sequence directory not found: {sequence_dir}")
        return
    
    print(f"Visualizing poses from: {sequence_dir.name}")
    print(f"Output directory: {output_dir}\n")
    
    # Visualize frames at intervals (1, 10, 20, 30, 40, 50)
    frame_indices = [1, 10, 20, 30, 40, 50]
    
    for idx in frame_indices:
        frame_path = sequence_dir / f"fall-01-cam0-rgb-{idx:03d}.png"
        
        if not frame_path.exists():
            continue
        
        # Load frame
        frame = cv2.imread(str(frame_path))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run inference
        keypoints = infer_keypoints(inference_fn, frame_rgb, confidence_threshold=0.3)
        
        # Visualize
        output_path = output_dir / f"fall-01-frame-{idx:03d}_pose.png"
        visualize_keypoints(frame_rgb, keypoints, 
                          confidence_threshold=0.3,
                          show=False,
                          save_path=str(output_path))
        
        print(f"✓ Frame {idx:3d} → {output_path.name}")
    
    print(f"\n✓ Visualizations saved to: {output_dir}")


def example_5_compare_fall_vs_adl():
    """Example 5: Compare pose detection in fall vs ADL sequences."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Compare Fall vs ADL Pose Detection")
    print("="*60 + "\n")
    
    # Load model
    inference_fn = load_movenet()
    
    # Process fall sequence
    fall_dir = Path("data/raw/urfd/falls/fall-01-cam0-rgb")
    adl_dir = Path("data/raw/urfd/adl/adl-01-cam0-rgb")
    
    def analyze_sequence(seq_dir, seq_type, num_frames=10):
        """Analyze a sequence and return statistics."""
        if not seq_dir.exists():
            print(f"⚠ {seq_type} sequence not found: {seq_dir}")
            return None
        
        print(f"Analyzing {seq_type} sequence: {seq_dir.name}")
        
        confidences = []
        keypoint_counts = []
        
        for i in range(1, num_frames + 1):
            # Construct frame path based on sequence type
            if seq_type == "Fall":
                frame_path = seq_dir / f"fall-01-cam0-rgb-{i:03d}.png"
            else:
                frame_path = seq_dir / f"adl-01-cam0-rgb-{i:03d}.png"
            
            if not frame_path.exists():
                continue
            
            # Load and process
            frame = cv2.imread(str(frame_path))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            keypoints = infer_keypoints(inference_fn, frame_rgb, confidence_threshold=0.3)
            
            # Collect statistics
            confidences.append(np.mean(keypoints[:, 2]))
            keypoint_counts.append(np.sum(keypoints[:, 2] >= 0.3))
        
        return {
            'avg_confidence': np.mean(confidences),
            'avg_keypoints': np.mean(keypoint_counts),
            'frames_processed': len(confidences)
        }
    
    # Analyze both sequences
    fall_stats = analyze_sequence(fall_dir, "Fall")
    adl_stats = analyze_sequence(adl_dir, "ADL")
    
    # Print comparison
    print("\n" + "-"*60)
    print("COMPARISON RESULTS")
    print("-"*60)
    
    if fall_stats:
        print(f"\nFall Sequence:")
        print(f"  Frames processed: {fall_stats['frames_processed']}")
        print(f"  Avg confidence: {fall_stats['avg_confidence']:.3f}")
        print(f"  Avg keypoints detected: {fall_stats['avg_keypoints']:.1f}/17")
    
    if adl_stats:
        print(f"\nADL Sequence:")
        print(f"  Frames processed: {adl_stats['frames_processed']}")
        print(f"  Avg confidence: {adl_stats['avg_confidence']:.3f}")
        print(f"  Avg keypoints detected: {adl_stats['avg_keypoints']:.1f}/17")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("MoveNet Pose Estimation Examples")
    print("="*60)
    
    try:
        example_1_single_image()
        example_2_batch_processing()
        example_3_extract_pose_features()
        example_4_visualize_poses()
        example_5_compare_fall_vs_adl()
        
        print("\n" + "="*60)
        print("✓ All examples completed successfully!")
        print("="*60 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Examples interrupted by user")
    except Exception as e:
        print(f"\n\n✗ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

