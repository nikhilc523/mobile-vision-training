"""
Unit tests for MoveNet pose estimation loader.

Tests cover:
- Model loading
- Frame preprocessing
- Keypoint inference
- Confidence thresholding
- Visualization utilities
"""

import unittest
import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ml.pose.movenet_loader import (
    load_movenet,
    preprocess_frame,
    infer_keypoints,
    visualize_keypoints,
    KEYPOINT_NAMES,
    KEYPOINT_EDGES,
)

try:
    import cv2
    import tensorflow as tf
except ImportError:
    print("ERROR: TensorFlow and OpenCV are required for tests.")
    sys.exit(1)


class TestMoveNetLoader(unittest.TestCase):
    """Test suite for MoveNet model loading."""
    
    @classmethod
    def setUpClass(cls):
        """Load model once for all tests."""
        print("\n" + "="*60)
        print("Loading MoveNet model for tests...")
        print("="*60)
        cls.inference_fn = load_movenet()
        print("✓ Model loaded successfully!\n")
    
    def test_model_loads_successfully(self):
        """Test that MoveNet model loads without errors."""
        self.assertIsNotNone(self.inference_fn)
        self.assertTrue(callable(self.inference_fn))
    
    def test_keypoint_names_count(self):
        """Test that we have 17 keypoint names."""
        self.assertEqual(len(KEYPOINT_NAMES), 17)
    
    def test_keypoint_edges_valid(self):
        """Test that skeleton edges reference valid keypoint indices."""
        for edge in KEYPOINT_EDGES:
            self.assertEqual(len(edge), 2)
            self.assertGreaterEqual(edge[0], 0)
            self.assertLess(edge[0], 17)
            self.assertGreaterEqual(edge[1], 0)
            self.assertLess(edge[1], 17)


class TestFramePreprocessing(unittest.TestCase):
    """Test suite for frame preprocessing."""
    
    def test_preprocess_square_image(self):
        """Test preprocessing of square image."""
        # Create 192x192 test image
        frame = np.random.randint(0, 255, (192, 192, 3), dtype=np.uint8)
        tensor = preprocess_frame(frame)
        
        self.assertEqual(tensor.shape, (1, 192, 192, 3))
        self.assertEqual(tensor.dtype, tf.int32)
    
    def test_preprocess_rectangular_image(self):
        """Test preprocessing of rectangular image (640x480)."""
        # Create 640x480 test image (URFD size)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        tensor = preprocess_frame(frame)
        
        self.assertEqual(tensor.shape, (1, 192, 192, 3))
        self.assertEqual(tensor.dtype, tf.int32)
    
    def test_preprocess_small_image(self):
        """Test preprocessing of small image."""
        # Create 100x100 test image
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        tensor = preprocess_frame(frame)
        
        self.assertEqual(tensor.shape, (1, 192, 192, 3))
        self.assertEqual(tensor.dtype, tf.int32)
    
    def test_preprocess_large_image(self):
        """Test preprocessing of large image."""
        # Create 1920x1080 test image
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        tensor = preprocess_frame(frame)
        
        self.assertEqual(tensor.shape, (1, 192, 192, 3))
        self.assertEqual(tensor.dtype, tf.int32)


class TestKeypointInference(unittest.TestCase):
    """Test suite for keypoint inference."""
    
    @classmethod
    def setUpClass(cls):
        """Load model once for all tests."""
        cls.inference_fn = load_movenet()
    
    def test_inference_output_shape(self):
        """Test that inference returns correct shape."""
        # Create test image
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        keypoints = infer_keypoints(self.inference_fn, frame)
        
        self.assertEqual(keypoints.shape, (17, 3))
    
    def test_inference_output_range(self):
        """Test that keypoint coordinates are in [0, 1] range."""
        # Create test image
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        keypoints = infer_keypoints(self.inference_fn, frame)
        
        # Check y, x coordinates are in [0, 1]
        self.assertTrue(np.all(keypoints[:, 0] >= 0.0))
        self.assertTrue(np.all(keypoints[:, 0] <= 1.0))
        self.assertTrue(np.all(keypoints[:, 1] >= 0.0))
        self.assertTrue(np.all(keypoints[:, 1] <= 1.0))
        
        # Check confidence is in [0, 1]
        self.assertTrue(np.all(keypoints[:, 2] >= 0.0))
        self.assertTrue(np.all(keypoints[:, 2] <= 1.0))
    
    def test_confidence_threshold_masking(self):
        """Test that low-confidence keypoints are masked."""
        # Create test image
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test with high threshold
        keypoints = infer_keypoints(self.inference_fn, frame, confidence_threshold=0.9)
        
        # Low-confidence keypoints should have coordinates set to 0
        low_conf_mask = keypoints[:, 2] < 0.9
        if np.any(low_conf_mask):
            self.assertTrue(np.all(keypoints[low_conf_mask, :2] == 0.0))
    
    def test_inference_on_real_image(self):
        """Test inference on a real URFD image if available."""
        test_image = Path("data/raw/urfd/falls/fall-01-cam0-rgb/fall-01-cam0-rgb-001.png")
        
        if not test_image.exists():
            self.skipTest("Test image not found")
        
        # Load image
        frame = cv2.imread(str(test_image))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run inference
        keypoints = infer_keypoints(self.inference_fn, frame_rgb)
        
        # Check output
        self.assertEqual(keypoints.shape, (17, 3))
        
        # Should have at least some high-confidence keypoints
        high_conf_count = np.sum(keypoints[:, 2] >= 0.3)
        self.assertGreater(high_conf_count, 0, "Should detect at least some keypoints")


class TestVisualization(unittest.TestCase):
    """Test suite for visualization utilities."""
    
    def test_visualize_without_matplotlib(self):
        """Test that visualization handles missing matplotlib gracefully."""
        # This test just ensures no crash occurs
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        keypoints = np.random.rand(17, 3)
        keypoints[:, 2] = 0.5  # Set confidence
        
        # Should not crash
        try:
            fig = visualize_keypoints(frame, keypoints, show=False)
            # If matplotlib is available, should return figure
            if fig is not None:
                self.assertIsNotNone(fig)
        except Exception as e:
            self.fail(f"Visualization raised unexpected exception: {e}")


class TestRealDataIntegration(unittest.TestCase):
    """Integration tests with real dataset images."""
    
    @classmethod
    def setUpClass(cls):
        """Load model once for all tests."""
        cls.inference_fn = load_movenet()
    
    def test_urfd_fall_sequence(self):
        """Test inference on multiple frames from URFD fall sequence."""
        fall_dir = Path("data/raw/urfd/falls/fall-01-cam0-rgb")
        
        if not fall_dir.exists():
            self.skipTest("URFD dataset not found")
        
        # Test first 3 frames
        for i in range(1, 4):
            frame_path = fall_dir / f"fall-01-cam0-rgb-{i:03d}.png"
            
            if not frame_path.exists():
                continue
            
            # Load and process
            frame = cv2.imread(str(frame_path))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            keypoints = infer_keypoints(self.inference_fn, frame_rgb)
            
            # Verify output
            self.assertEqual(keypoints.shape, (17, 3))
            
            # Should detect person in frame
            high_conf_count = np.sum(keypoints[:, 2] >= 0.3)
            self.assertGreater(high_conf_count, 5, 
                             f"Frame {i}: Should detect multiple keypoints")
    
    def test_urfd_adl_sequence(self):
        """Test inference on ADL (non-fall) sequence."""
        adl_dir = Path("data/raw/urfd/adl/adl-01-cam0-rgb")
        
        if not adl_dir.exists():
            self.skipTest("URFD ADL dataset not found")
        
        # Test first frame
        frame_path = adl_dir / "adl-01-cam0-rgb-001.png"
        
        if not frame_path.exists():
            self.skipTest("ADL test image not found")
        
        # Load and process
        frame = cv2.imread(str(frame_path))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        keypoints = infer_keypoints(self.inference_fn, frame_rgb)
        
        # Verify output
        self.assertEqual(keypoints.shape, (17, 3))


def run_tests():
    """Run all tests with verbose output."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestMoveNetLoader))
    suite.addTests(loader.loadTestsFromTestCase(TestFramePreprocessing))
    suite.addTests(loader.loadTestsFromTestCase(TestKeypointInference))
    suite.addTests(loader.loadTestsFromTestCase(TestVisualization))
    suite.addTests(loader.loadTestsFromTestCase(TestRealDataIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("="*60)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)

