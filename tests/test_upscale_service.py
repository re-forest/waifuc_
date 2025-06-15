"""
Unit tests for the UpscaleService.
"""
import unittest
import os
from PIL import Image
import tempfile
from unittest.mock import patch, MagicMock

from services.upscale_service import upscale_image_service, upscale_image_service_entry
from config import settings
from utils.logger_config import setup_logging
from services.file_service import FileService

# Configure logger for tests
logger = setup_logging(__name__, 'test_logs', log_level_str='DEBUG')

class TestUpscaleService(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up for all tests in this class."""
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.mock_models_dir = os.path.join(cls.temp_dir.name, "mock_upscale_models")
        os.makedirs(cls.mock_models_dir, exist_ok=True)
        
        # Set up FileService for image handling
        cls.file_service = FileService(temp_dir=os.path.join(cls.temp_dir.name, "fs_temp"))

        # Create test images
        cls.input_image_path = cls._create_dummy_image("input_image.png", (50, 50), "PNG")
        cls.output_dir = os.path.join(cls.temp_dir.name, "output_images")
        os.makedirs(cls.output_dir, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests in this class."""
        import gc
        import time
        
        # Force garbage collection to release file handles
        gc.collect()
        
        # Add a small delay for Windows to release file handles
        time.sleep(0.1)
        
        try:
            cls.temp_dir.cleanup()
            logger.info(f"Temporary directory for TestUpscaleService cleaned up")
        except (PermissionError, OSError) as e:
            # On Windows, sometimes files are still locked
            logger.warning(f"Could not clean up temporary directory immediately: {e}")
            # Try again after a brief pause
            try:
                time.sleep(0.5)
                cls.temp_dir.cleanup()
                logger.info(f"Temporary directory for TestUpscaleService cleaned up on retry")
            except Exception as retry_e:
                logger.warning(f"Failed to clean up temporary directory: {retry_e}")
                # Let the OS handle it eventually

    @classmethod
    def _create_dummy_image(cls, name, size, img_format):
        """Helper method to create a dummy image file."""
        path = os.path.join(cls.temp_dir.name, name)
        try:
            img = Image.new('RGB', size, color='blue')
            img.save(path, format=img_format)
            logger.debug(f"Created dummy image for upscale test: {path}")
            return path
        except Exception as e:
            logger.error(f"Failed to create dummy image {path}: {e}")
            return None

    def setUp(self):
        """Set up for each test method."""
        if not self.input_image_path:
            self.skipTest("Input image not created, skipping test.")

    def test_upscale_image_service_pil_input(self):
        """Test upscaling with a PIL Image object as input."""
        if not self.input_image_path:
            self.skipTest("Test image not available")
            
        # Load the test image
        pil_image = Image.open(self.input_image_path)
        original_size = pil_image.size
        
        # Mock the actual upscaling function from imgutils
        with patch('services.upscale_service.upscale_with_cdc') as mock_upscale:
            # Mock the upscale function to return a larger image
            mock_upscaled = Image.new('RGB', (original_size[0] * 2, original_size[1] * 2), color='green')
            mock_upscale.return_value = mock_upscaled
            
            # Test the service function - it returns (image, message) tuple
            result_image, message = upscale_image_service(pil_image, logger, config=settings)
            
            self.assertIsNotNone(result_image, "Upscaled image should not be None.")
            if result_image:  # Add None check for type safety
                self.assertIsInstance(result_image, Image.Image, "Result should be a PIL Image.")
                # The actual size depends on the service implementation, but should be larger
                self.assertGreaterEqual(result_image.width, original_size[0], "Width should be at least original size")
                self.assertGreaterEqual(result_image.height, original_size[1], "Height should be at least original size")
            
            # Verify that the mock was called
            mock_upscale.assert_called_once()
            logger.info("test_upscale_image_service_pil_input completed successfully.")

    def test_upscale_image_service_entry_with_path(self):
        """Test upscaling with an image file path as input."""
        if not self.input_image_path:
            self.skipTest("Test image not available")
            
        original_img = Image.open(self.input_image_path)
        original_size = original_img.size
        
        # Mock the actual upscaling function
        with patch('services.upscale_service.upscale_with_cdc') as mock_upscale:
            mock_upscaled = Image.new('RGB', (original_size[0] * 2, original_size[1] * 2), color='green')
            mock_upscale.return_value = mock_upscaled
            
            # Test the service entry function - it returns (image, output_path, message) tuple
            result_image, output_path, message = upscale_image_service_entry(
                self.input_image_path, 
                logger, 
                config=settings, 
                output_path=None  # Let it use default
            )
            
            self.assertIsNotNone(result_image, "Upscaled image should not be None.")
            self.assertIsInstance(result_image, Image.Image, "Result should be a PIL Image.")
            # Note: Due to service's default resizing behavior, we check it's at least as large
            self.assertGreaterEqual(result_image.width, original_size[0], "Width should be at least original size")
            self.assertGreaterEqual(result_image.height, original_size[1], "Height should be at least original size")
            self.assertIsNone(output_path, "Output path should be None when not provided")
            
            mock_upscale.assert_called_once()
            logger.info("test_upscale_image_service_entry_with_path completed successfully.")

    def test_upscale_and_save_to_file(self):
        """Test upscaling an image and saving the result."""
        if not self.input_image_path:
            self.skipTest("Test image not available")
            
        original_img = Image.open(self.input_image_path)
        original_size = original_img.size
        output_filename = "upscaled_output.png"
        
        # Mock the upscaling function
        with patch('services.upscale_service.upscale_with_cdc') as mock_upscale:
            mock_upscaled = Image.new('RGB', (original_size[0] * 2, original_size[1] * 2), color='green')
            mock_upscale.return_value = mock_upscaled
            
            # Get the upscaled image from service entry
            upscaled_image, _, _ = upscale_image_service_entry(
                self.input_image_path, 
                logger, 
                config=settings
            )
            
            self.assertIsNotNone(upscaled_image)
            
            # Save using FileService
            result_path = self.file_service.save_processed_image(
                upscaled_image, 
                output_filename, 
                self.output_dir
            )
            
            self.assertIsNotNone(result_path, "Result path should not be None.")
            if result_path:  # Additional None check for type safety
                self.assertTrue(os.path.exists(result_path), f"Output file {result_path} should exist.")
                
                # Verify image properties - note the service may resize beyond simple 2x due to default settings
                saved_img = Image.open(result_path)
                self.assertGreaterEqual(saved_img.width, original_size[0], "Saved image width should be at least original")
                self.assertGreaterEqual(saved_img.height, original_size[1], "Saved image height should be at least original")
                logger.info(f"test_upscale_and_save_to_file completed. Output at {result_path}")

    def test_upscale_service_with_model_error(self):
        """Test upscaling when the model encounters an error."""
        if not self.input_image_path:
            self.skipTest("Test image not available")
            
        pil_image = Image.open(self.input_image_path)
        
        # Mock the upscaling function to raise an exception
        with patch('services.upscale_service.upscale_with_cdc') as mock_upscale:
            mock_upscale.side_effect = Exception("Mock model error")
            
            # The service should handle the error gracefully and return (None, error_message)
            result_image, message = upscale_image_service(pil_image, logger, config=settings)
            
            # Verify the error was handled
            self.assertIsNone(result_image, "Result image should be None on error")
            self.assertIn("failed", message.lower(), f"Error message should indicate failure: {message}")
            mock_upscale.assert_called_once()
            logger.info("test_upscale_service_with_model_error completed successfully.")

    def test_upscale_service_entry_with_invalid_path(self):
        """Test upscaling with an invalid image path."""
        invalid_path = "non_existent_image.png"
        
        # The service should handle the invalid path gracefully by raising an exception
        with self.assertRaises(Exception) as context:
            upscale_image_service_entry(invalid_path, logger, config=settings)
        
        # Check that the exception mentions the file issue
        error_message = str(context.exception).lower()
        self.assertTrue(
            any(keyword in error_message for keyword in ["not found", "file", "path", "exist"]),
            f"Error message should indicate file issue: {context.exception}"
        )
        logger.info("test_upscale_service_entry_with_invalid_path completed successfully.")

    def test_upscale_service_with_text_file(self):
        """Test upscaling with a text file instead of an image."""
        # Create a dummy text file
        dummy_text_file = os.path.join(self.temp_dir.name, "not_an_image.txt")
        with open(dummy_text_file, 'w') as f:
            f.write("this is text")
        
        # The service should handle this gracefully by raising an exception
        with self.assertRaises(Exception) as context:
            upscale_image_service_entry(dummy_text_file, logger, config=settings)
        
        # Check that the exception indicates image loading issue
        error_message = str(context.exception).lower()
        self.assertTrue(
            any(keyword in error_message for keyword in ["image", "load", "format", "open"]),
            f"Error message should indicate image loading issue: {context.exception}"
        )
        logger.info("test_upscale_service_with_text_file completed successfully.")

if __name__ == '__main__':
    unittest.main()
