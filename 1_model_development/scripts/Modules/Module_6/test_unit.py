"""
Unit Tests for Enhanced ESP32-S3-EYE Container Detection System
Tests individual components in isolation
"""

import unittest
import sys
import os
import numpy as np
import cv2
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch

# Add module path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from local_inference import LocalInferenceEngine, HDCClassifier, TinyNASFeatureExtractor
from camera_manager import CameraManager, LaptopCamera, VideoFileCamera, ESP32Camera

class TestTinyNASFeatureExtractor(unittest.TestCase):
    """Test TinyNAS feature extractor"""
    
    def setUp(self):
        self.model = TinyNASFeatureExtractor(width_mult=1.0, num_classes=5)
        self.model.eval()
        
    def test_model_initialization(self):
        """Test model initializes correctly"""
        self.assertIsInstance(self.model, TinyNASFeatureExtractor)
        self.assertEqual(len(self.model.class_names), 5) if hasattr(self.model, 'class_names') else None
        
    def test_forward_pass(self):
        """Test forward pass with dummy input"""
        # Create dummy input: batch_size=1, channels=3, height=64, width=64
        dummy_input = torch.randn(1, 3, 64, 64)
        
        with torch.no_grad():
            # Test classification output
            output = self.model(dummy_input)
            self.assertEqual(output.shape, (1, 5))  # 5 classes
            
            # Test feature extraction
            features = self.model(dummy_input, return_features=True)
            self.assertEqual(features.shape, (1, 256))  # 256D features
            
    def test_feature_extraction_consistency(self):
        """Test that feature extraction is deterministic"""
        dummy_input = torch.randn(1, 3, 64, 64)
        
        with torch.no_grad():
            features1 = self.model(dummy_input, return_features=True)
            features2 = self.model(dummy_input, return_features=True)
            
            # Should be identical (deterministic)
            torch.testing.assert_close(features1, features2)

class TestHDCClassifier(unittest.TestCase):
    """Test HDC classifier"""
    
    def setUp(self):
        self.hdc = HDCClassifier(feature_dim=256, hd_dim=2048, num_classes=5)
        
        # Create mock projection matrix and prototypes
        self.hdc.projection_matrix = np.random.randn(2048, 256).astype(np.float32)
        self.hdc.class_prototypes = np.random.choice([-1, 1], size=(5, 2048)).astype(np.int8)
        
    def test_initialization(self):
        """Test HDC classifier initialization"""
        self.assertEqual(self.hdc.feature_dim, 256)
        self.assertEqual(self.hdc.hd_dim, 2048)
        self.assertEqual(self.hdc.num_classes, 5)
        self.assertEqual(len(self.hdc.class_names), 5)
        
    def test_feature_encoding(self):
        """Test feature encoding to hyperdimensional space"""
        features = np.random.randn(256).astype(np.float32)
        
        hypervector = self.hdc.encode_features(features)
        
        self.assertEqual(hypervector.shape, (2048,))
        self.assertTrue(np.all(np.isin(hypervector, [-1, 1])))  # Should be bipolar
        
    def test_classification(self):
        """Test HDC classification"""
        features = np.random.randn(256).astype(np.float32)
        
        class_id, confidence, similarities = self.hdc.classify(features)
        
        self.assertIsInstance(class_id, (int, np.integer))
        self.assertTrue(0 <= class_id < 5)
        self.assertTrue(0 <= confidence <= 1)
        self.assertEqual(similarities.shape, (5,))
        
    def test_npz_loading(self):
        """Test loading from NPZ file"""
        # Create temporary NPZ file
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
            np.savez(tmp.name,
                    projection_matrix=self.hdc.projection_matrix,
                    class_prototypes=self.hdc.class_prototypes)
            tmp_path = tmp.name
            
        try:
            new_hdc = HDCClassifier()
            success = new_hdc.load_from_npz(tmp_path)
            
            self.assertTrue(success)
            np.testing.assert_array_equal(new_hdc.projection_matrix, self.hdc.projection_matrix)
            np.testing.assert_array_equal(new_hdc.class_prototypes, self.hdc.class_prototypes)
            
        finally:
            os.unlink(tmp_path)

class TestLocalInferenceEngine(unittest.TestCase):
    """Test local inference engine"""
    
    def setUp(self):
        # Create a mock models directory
        self.temp_dir = tempfile.mkdtemp()
        self.engine = LocalInferenceEngine(models_dir=self.temp_dir)
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def test_initialization(self):
        """Test inference engine initialization"""
        self.assertIsInstance(self.engine, LocalInferenceEngine)
        self.assertEqual(len(self.engine.class_names), 5)
        self.assertEqual(len(self.engine.class_colors), 5)
        
    def test_image_preprocessing(self):
        """Test image preprocessing pipeline"""
        # Create test image
        test_image = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        
        tensor = self.engine.preprocess_image(test_image, target_size=(64, 64))
        
        self.assertEqual(tensor.shape, (1, 3, 64, 64))  # Batch, channels, height, width
        self.assertTrue(tensor.dtype == torch.float32)
        
    def test_iou_calculation(self):
        """Test IoU calculation"""
        det1 = {'x': 10, 'y': 10, 'w': 50, 'h': 50}
        det2 = {'x': 30, 'y': 30, 'w': 50, 'h': 50}
        
        iou = self.engine.calculate_iou(det1, det2)
        
        self.assertTrue(0 <= iou <= 1)
        
        # Test perfect overlap
        iou_perfect = self.engine.calculate_iou(det1, det1)
        self.assertAlmostEqual(iou_perfect, 1.0, places=5)
        
        # Test no overlap
        det3 = {'x': 100, 'y': 100, 'w': 50, 'h': 50}
        iou_none = self.engine.calculate_iou(det1, det3)
        self.assertEqual(iou_none, 0.0)
        
    def test_detection_filtering(self):
        """Test detection filtering with NMS"""
        # Create overlapping detections
        detections = [
            {'x': 10, 'y': 10, 'w': 50, 'h': 50, 'confidence': 0.9, 'class_name': 'axis'},
            {'x': 15, 'y': 15, 'w': 50, 'h': 50, 'confidence': 0.8, 'class_name': 'axis'},
            {'x': 100, 'y': 100, 'w': 50, 'h': 50, 'confidence': 0.7, 'class_name': 'concave'}
        ]
        
        filtered = self.engine.filter_detections(detections, iou_threshold=0.3)
        
        # Should keep highest confidence detection from overlapping pair + non-overlapping one
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0]['confidence'], 0.9)  # Highest confidence kept

class TestCameraManager(unittest.TestCase):
    """Test camera manager"""
    
    def setUp(self):
        self.manager = CameraManager()
        
    def test_initialization(self):
        """Test camera manager initialization"""
        self.assertIsInstance(self.manager, CameraManager)
        self.assertIsNone(self.manager.current_source)
        
    def test_source_scanning(self):
        """Test scanning for available sources"""
        sources = self.manager.get_available_sources()
        self.assertIsInstance(sources, dict)
        
        # Should at least detect some laptop cameras (even if none available)
        # This test might pass with empty dict on headless systems
        
    @patch('cv2.VideoCapture')
    def test_laptop_camera_creation(self, mock_videocapture):
        """Test laptop camera source creation"""
        # Mock successful camera
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_videocapture.return_value = mock_cap
        
        # Add mock laptop camera to sources
        self.manager.available_sources["Test Laptop Camera 0"] = {
            'type': 'laptop',
            'device': 0,
            'description': 'Test camera'
        }
        
        success = self.manager.switch_to_source("Test Laptop Camera 0")
        self.assertTrue(success)
        self.assertIsNotNone(self.manager.current_source)

class TestVideoFileCamera(unittest.TestCase):
    """Test video file camera source"""
    
    def setUp(self):
        # Create temporary video file (empty)
        self.temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        self.temp_video.close()
        
    def tearDown(self):
        os.unlink(self.temp_video.name)
        
    def test_initialization(self):
        """Test video camera initialization"""
        video_cam = VideoFileCamera(self.temp_video.name)
        self.assertEqual(str(video_cam.video_path), self.temp_video.name)
        self.assertFalse(video_cam.is_playing)
        
    def test_playback_controls(self):
        """Test video playback control methods"""
        video_cam = VideoFileCamera(self.temp_video.name)
        
        # Test initial state
        self.assertFalse(video_cam.is_playing)
        
        # Test play/pause (won't actually work with empty file, but methods should exist)
        video_cam.play()
        video_cam.pause()
        video_cam.set_playback_speed(2.0)
        
        progress = video_cam.get_progress()
        self.assertIsInstance(progress, tuple)
        self.assertEqual(len(progress), 2)

class TestImageProcessing(unittest.TestCase):
    """Test image processing utilities"""
    
    def test_create_test_image(self):
        """Test creation of test images"""
        # Test different image sizes and formats
        img_rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img_gray = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        
        self.assertEqual(img_rgb.shape, (480, 640, 3))
        self.assertEqual(img_gray.shape, (480, 640))
        
    def test_center_crop(self):
        """Test center cropping functionality"""
        engine = LocalInferenceEngine()
        
        # Test rectangular image
        rect_image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        
        # This tests the preprocessing which includes center cropping
        tensor = engine.preprocess_image(rect_image, target_size=(64, 64))
        
        # Should produce square output
        self.assertEqual(tensor.shape[-2:], (64, 64))

class TestDetectionVisualization(unittest.TestCase):
    """Test detection visualization"""
    
    def setUp(self):
        self.engine = LocalInferenceEngine()
        
    def test_draw_detections(self):
        """Test drawing detection overlays"""
        # Create test image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Create test detections - NEW CONTAINER DETECTION FORMAT
        detections = [
            {
                'x': 100, 'y': 100, 'w': 64, 'h': 64,
                'damage_type': 'axis', 'is_damaged': True, 'confidence': 0.95
            },
            {
                'x': 200, 'y': 200, 'w': 64, 'h': 64,
                'damage_type': 'no_damage', 'is_damaged': False, 'confidence': 0.88
            }
        ]
        
        result_image = self.engine.draw_detections(image, detections)
        
        # Should return image of same size
        self.assertEqual(result_image.shape, image.shape)
        
        # Should be different from original (has drawings)
        self.assertFalse(np.array_equal(result_image, image))

def run_unit_tests():
    """Run all unit tests"""
    print("--- SCRIPT START ---")
    print("🧪 Running Unit Tests for Enhanced Container Detection System")
    print("=" * 60)
    
    # Create test suite
    print("Creating test suite...")
    test_suite = unittest.TestSuite()
    print("Test suite created.")
    
    # Add test classes
    test_classes = [
        TestTinyNASFeatureExtractor,
        TestHDCClassifier,
        TestLocalInferenceEngine,
        TestCameraManager,
        TestVideoFileCamera,
        TestImageProcessing,
        TestDetectionVisualization
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    print("Creating test runner...")
    runner = unittest.TextTestRunner(verbosity=2)
    print("Test runner created. Running tests...")
    result = runner.run(test_suite)
    print("Tests finished.")
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 UNIT TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\n❌ FAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.splitlines()[-1]}")
    
    if result.errors:
        print(f"\n🚨 ERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.splitlines()[-1]}")
    
    if len(result.failures) == 0 and len(result.errors) == 0:
        print("\n✅ ALL UNIT TESTS PASSED!")
    
    print("--- SCRIPT END ---")
    return result.wasSuccessful()

if __name__ == "__main__":
    print("Executing test_unit.py as main script...")
    success = run_unit_tests()
    print(f"Script finished with success status: {success}")
    sys.exit(0 if success else 1)
