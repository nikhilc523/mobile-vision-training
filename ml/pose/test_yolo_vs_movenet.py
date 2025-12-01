"""
Test YOLO vs MoveNet Pose Estimation

Compare YOLO11-Pose and MoveNet on test videos to see which performs better.
"""

import cv2
import numpy as np
import time
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_on_video(video_path: str, num_frames: int = 30):
    """
    Test both YOLO and MoveNet on a video and compare results.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to test (default: 30)
    """
    print("="*80)
    print(f"TESTING: {video_path}")
    print("="*80)
    print()
    
    # Check if YOLO is available
    try:
        from ultralytics import YOLO
        yolo_available = True
    except ImportError:
        print("⚠️  YOLO not installed. Install with: pip install ultralytics")
        yolo_available = False
    
    # Load models
    print("Loading models...")
    
    if yolo_available:
        from ml.pose.yolo_loader import load_yolo, infer_keypoints_yolo
        yolo_model = load_yolo('yolo11n-pose.pt')
        print("✓ YOLO loaded")
    
    from ml.pose.movenet_loader import load_movenet, infer_keypoints
    movenet_fn = load_movenet()
    print("✓ MoveNet loaded")
    print()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {total_frames} frames @ {fps:.2f} FPS")
    print(f"Testing first {num_frames} frames...")
    print()
    
    # Statistics
    yolo_times = []
    movenet_times = []
    yolo_confidences = []
    movenet_confidences = []
    yolo_valid_kps = []
    movenet_valid_kps = []
    
    # Process frames
    for i in range(min(num_frames, total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Test YOLO
        if yolo_available:
            start = time.time()
            yolo_kps = infer_keypoints_yolo(yolo_model, frame_rgb, normalize=True)
            yolo_time = time.time() - start
            
            yolo_times.append(yolo_time)
            yolo_confidences.append(yolo_kps[:, 2].mean())
            yolo_valid_kps.append(np.sum(yolo_kps[:, 2] >= 0.3))
        
        # Test MoveNet
        start = time.time()
        movenet_kps = infer_keypoints(movenet_fn, frame_rgb)
        movenet_time = time.time() - start
        
        movenet_times.append(movenet_time)
        movenet_confidences.append(movenet_kps[:, 2].mean())
        movenet_valid_kps.append(np.sum(movenet_kps[:, 2] >= 0.3))
        
        # Print progress every 10 frames
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{num_frames} frames...")
    
    cap.release()
    print()
    
    # Print results
    print("="*80)
    print("RESULTS")
    print("="*80)
    print()
    
    if yolo_available:
        print("YOLO11-Pose (Nano):")
        print(f"  Average FPS: {1/np.mean(yolo_times):.2f}")
        print(f"  Average confidence: {np.mean(yolo_confidences):.3f}")
        print(f"  Average valid keypoints: {np.mean(yolo_valid_kps):.1f}/17")
        print(f"  Min/Max confidence: {np.min(yolo_confidences):.3f} / {np.max(yolo_confidences):.3f}")
        print()
    
    print("MoveNet Lightning:")
    print(f"  Average FPS: {1/np.mean(movenet_times):.2f}")
    print(f"  Average confidence: {np.mean(movenet_confidences):.3f}")
    print(f"  Average valid keypoints: {np.mean(movenet_valid_kps):.1f}/17")
    print(f"  Min/Max confidence: {np.min(movenet_confidences):.3f} / {np.max(movenet_confidences):.3f}")
    print()
    
    if yolo_available:
        # Comparison
        print("COMPARISON:")
        yolo_fps = 1/np.mean(yolo_times)
        movenet_fps = 1/np.mean(movenet_times)
        
        if yolo_fps > movenet_fps:
            print(f"  ⚡ YOLO is {yolo_fps/movenet_fps:.2f}x FASTER")
        else:
            print(f"  ⚡ MoveNet is {movenet_fps/yolo_fps:.2f}x FASTER")
        
        yolo_conf = np.mean(yolo_confidences)
        movenet_conf = np.mean(movenet_confidences)
        
        if yolo_conf > movenet_conf:
            print(f"  🎯 YOLO has {(yolo_conf/movenet_conf - 1)*100:.1f}% HIGHER confidence")
        else:
            print(f"  🎯 MoveNet has {(movenet_conf/yolo_conf - 1)*100:.1f}% HIGHER confidence")
        
        yolo_valid = np.mean(yolo_valid_kps)
        movenet_valid = np.mean(movenet_valid_kps)
        
        if yolo_valid > movenet_valid:
            print(f"  ✓ YOLO detects {yolo_valid - movenet_valid:.1f} MORE keypoints on average")
        else:
            print(f"  ✓ MoveNet detects {movenet_valid - yolo_valid:.1f} MORE keypoints on average")
        print()
    
    return {
        'yolo_fps': 1/np.mean(yolo_times) if yolo_available else None,
        'movenet_fps': 1/np.mean(movenet_times),
        'yolo_confidence': np.mean(yolo_confidences) if yolo_available else None,
        'movenet_confidence': np.mean(movenet_confidences),
        'yolo_valid_kps': np.mean(yolo_valid_kps) if yolo_available else None,
        'movenet_valid_kps': np.mean(movenet_valid_kps)
    }


if __name__ == '__main__':
    """Run comparison tests"""
    
    # Test videos
    test_videos = [
        'data/test/finalfall.mp4',
        'data/test/secondfall.mp4'
    ]
    
    results = {}
    
    for video_path in test_videos:
        if Path(video_path).exists():
            results[video_path] = test_on_video(video_path, num_frames=30)
        else:
            print(f"⚠️  Video not found: {video_path}")
            print()
    
    # Summary
    if results:
        print("="*80)
        print("SUMMARY")
        print("="*80)
        print()
        
        for video_path, result in results.items():
            print(f"{Path(video_path).name}:")
            if result['yolo_fps']:
                print(f"  YOLO: {result['yolo_fps']:.1f} FPS, conf={result['yolo_confidence']:.3f}, kps={result['yolo_valid_kps']:.1f}/17")
            print(f"  MoveNet: {result['movenet_fps']:.1f} FPS, conf={result['movenet_confidence']:.3f}, kps={result['movenet_valid_kps']:.1f}/17")
            print()
        
        print("="*80)
        print("RECOMMENDATION")
        print("="*80)
        print()
        
        if results[test_videos[0]]['yolo_fps']:
            avg_yolo_conf = np.mean([r['yolo_confidence'] for r in results.values() if r['yolo_confidence']])
            avg_movenet_conf = np.mean([r['movenet_confidence'] for r in results.values()])
            
            if avg_yolo_conf > avg_movenet_conf * 1.1:  # 10% better
                print("✅ RECOMMENDATION: Switch to YOLO")
                print("   - Higher confidence scores")
                print("   - Better pose detection quality")
                print("   - May improve fall detection on real-world videos")
                print()
                print("   To switch: Use yolo_loader.py instead of movenet_loader.py")
                print("   Minimal code changes needed!")
            else:
                print("✅ RECOMMENDATION: Keep MoveNet")
                print("   - Similar or better performance")
                print("   - Already integrated and working")
                print("   - Lighter weight")
        else:
            print("⚠️  Install YOLO to see comparison: pip install ultralytics")

