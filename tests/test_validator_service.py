"""
Unit tests for the ValidatorService.
"""
import unittest
import os
from PIL import Image
import tempfile

from services.validator_service import validate_image_service, _validate_single_image_internal
from config import settings
from utils.logger_config import setup_logging

# Configure logger for tests (optional, but good for debugging)
logger = setup_logging(__name__, 'test_logs', log_level_str='DEBUG')

class TestValidatorService(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up for all tests in this class."""
        # Create a temporary directory for test images
        cls.temp_dir = tempfile.TemporaryDirectory()
        logger.info(f"Temporary directory for tests created: {cls.temp_dir.name}")

        # Create some dummy image files for testing
        cls.valid_image_path = cls._create_dummy_image("valid_image.png", (100, 100), "PNG")
        cls.small_image_path = cls._create_dummy_image("small_image.png", (5, 5), "PNG") 
        cls.invalid_format_path = cls._create_dummy_text_file("invalid_format.txt")
        cls.non_existent_path = "non_existent_image.png"

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests in this class."""
        cls.temp_dir.cleanup()
        logger.info(f"Temporary directory for tests cleaned up: {cls.temp_dir.name}")

    @classmethod
    def _create_dummy_image(cls, name, size, img_format):
        """Helper method to create a dummy image file."""
        path = os.path.join(cls.temp_dir.name, name)
        try:
            img = Image.new('RGB', size, color='red')
            img.save(path, format=img_format)
            logger.debug(f"Created dummy image: {path}")
            return path
        except Exception as e:
            logger.error(f"Failed to create dummy image {path}: {e}")
            return None

    @classmethod
    def _create_dummy_text_file(cls, name):
        """Helper method to create a dummy text file."""
        path = os.path.join(cls.temp_dir.name, name)
        try:
            with open(path, 'w') as f:
                f.write("This is not an image.")
            logger.debug(f"Created dummy text file: {path}")
            return path
        except Exception as e:
            logger.error(f"Failed to create dummy text file {path}: {e}")
            return None

    def setUp(self):
        """Set up for each test method."""
        # If ValidatorService has state that needs resetting per test, do it here
        pass

    def tearDown(self):
        """Clean up after each test method."""
        pass

    def test_validate_image_valid(self):
        """Test validation with a valid image."""
        if not self.valid_image_path:
            self.skipTest("Valid test image not created, skipping test.")
            
        is_valid, message, valid_paths = validate_image_service(
            self.valid_image_path, 
            logger, 
            config=settings, 
            is_directory=False
        )
        self.assertTrue(is_valid, f"Validation should pass for a valid image. Message: {message}")
        self.assertEqual(len(valid_paths), 1, "Should return one valid path")
        self.assertEqual(valid_paths[0], self.valid_image_path, "Should return the correct path")

    def test_validate_image_too_small(self):
        """Test validation with an image that is too small."""
        # The current validator service doesn't check for min dimensions,
        # but it should still validate the image as a valid image file
        if not self.small_image_path:
            self.skipTest("Small test image not created, skipping test.")
            
        is_valid, message, valid_paths = validate_image_service(
            self.small_image_path, 
            logger, 
            config=settings, 
            is_directory=False
        )
        # Currently the service only checks if it's a valid image, not dimensions
        self.assertTrue(is_valid, f"Even small images should be valid if they're proper image files. Message: {message}")

    def test_validate_image_invalid_format(self):
        """Test validation with a file that is not a valid image format."""
        if not self.invalid_format_path:
            self.skipTest("Invalid format test file not created, skipping test.")
            
        is_valid, message, valid_paths = validate_image_service(
            self.invalid_format_path, 
            logger, 
            config=settings, 
            is_directory=False
        )
        self.assertFalse(is_valid, f"Validation should fail for a non-image file. Message: {message}")
        self.assertEqual(len(valid_paths), 0, "Should return no valid paths")

    def test_validate_image_non_existent(self):
        """Test validation with a non-existent image path."""
        is_valid, message, valid_paths = validate_image_service(
            self.non_existent_path, 
            logger, 
            config=settings, 
            is_directory=False
        )
        self.assertFalse(is_valid, f"Validation should fail for a non-existent file. Message: {message}")
        self.assertEqual(len(valid_paths), 0, "Should return no valid paths")

    def test_validate_directory_with_valid_images(self):
        """Test validation with a directory containing valid images."""
        # Create a test directory with multiple images
        test_dir = os.path.join(self.temp_dir.name, "test_images")
        os.makedirs(test_dir, exist_ok=True)
        
        # Create test images in the directory
        valid_image_1 = os.path.join(test_dir, "valid1.png")
        valid_image_2 = os.path.join(test_dir, "valid2.jpg")
        invalid_file = os.path.join(test_dir, "invalid.txt")
        
        Image.new('RGB', (50, 50), color='blue').save(valid_image_1)
        Image.new('RGB', (60, 60), color='green').save(valid_image_2)
        with open(invalid_file, 'w') as f:
            f.write("Not an image")
        
        is_valid, message, valid_paths = validate_image_service(
            test_dir, 
            logger, 
            config=settings, 
            is_directory=True
        )
        
        self.assertTrue(is_valid, f"Directory validation should succeed when valid images exist. Message: {message}")
        self.assertEqual(len(valid_paths), 2, f"Should find 2 valid images, found {len(valid_paths)}")
        # Check that the valid paths contain our expected images
        self.assertIn(valid_image_1, valid_paths)
        self.assertIn(valid_image_2, valid_paths)

    def test_validate_directory_no_valid_images(self):
        """Test validation with a directory containing no valid images."""
        test_dir = os.path.join(self.temp_dir.name, "empty_test_dir")
        os.makedirs(test_dir, exist_ok=True)
        
        # Add only invalid files
        invalid_file = os.path.join(test_dir, "invalid.txt")
        with open(invalid_file, 'w') as f:
            f.write("Not an image")
        
        is_valid, message, valid_paths = validate_image_service(
            test_dir, 
            logger, 
            config=settings, 
            is_directory=True
        )
        
        self.assertFalse(is_valid, f"Directory validation should fail when no valid images exist. Message: {message}")
        self.assertEqual(len(valid_paths), 0, "Should find no valid images")

    def test_validate_nonexistent_directory(self):
        """Test validation with a non-existent directory."""
        non_existent_dir = os.path.join(self.temp_dir.name, "non_existent_directory")
        
        is_valid, message, valid_paths = validate_image_service(
            non_existent_dir, 
            logger, 
            config=settings, 
            is_directory=True
        )
        
        self.assertFalse(is_valid, f"Directory validation should fail for non-existent directory. Message: {message}")
        self.assertEqual(len(valid_paths), 0, "Should find no valid images")

    def test_internal_validate_function(self):
        """Test the internal validation function directly."""
        if not self.valid_image_path:
            self.skipTest("Valid test image not created, skipping test.")
            
        # Test with valid image
        result = _validate_single_image_internal(self.valid_image_path, logger)
        self.assertTrue(result, "Internal validation should pass for valid image")
        
        # Test with invalid file
        if self.invalid_format_path:
            result = _validate_single_image_internal(self.invalid_format_path, logger)
            self.assertFalse(result, "Internal validation should fail for invalid file")

if __name__ == '__main__':
    unittest.main()
