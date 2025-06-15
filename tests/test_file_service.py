"""
Unit tests for the FileService.
"""
import unittest
import os
from PIL import Image
import tempfile
import shutil
from unittest.mock import patch, MagicMock, mock_open
from typing import cast

from services.file_service import FileService
from config import settings
from utils.logger_config import setup_logging

# Configure logger for tests
logger = setup_logging(__name__, 'test_logs', log_level_str='DEBUG')

class TestFileService(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up for all tests in this class."""
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.test_temp_dir = os.path.join(cls.temp_dir.name, "file_service_temp")
        cls.test_output_dir = os.path.join(cls.temp_dir.name, "output")
        
        # Create test directories
        os.makedirs(cls.test_temp_dir, exist_ok=True)
        os.makedirs(cls.test_output_dir, exist_ok=True)
        
        logger.info(f"Temporary directories created for FileService tests")

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests in this class."""
        cls.temp_dir.cleanup()
        logger.info(f"Temporary directories for FileService tests cleaned up")

    def setUp(self):
        """Set up for each test method."""
        self.file_service = FileService(temp_dir=self.test_temp_dir)

    def test_file_service_initialization(self):
        """Test FileService initialization."""
        # Test with provided temp_dir
        fs = FileService(temp_dir=self.test_temp_dir)
        self.assertEqual(fs.temp_dir, self.test_temp_dir)
        self.assertTrue(os.path.exists(fs.temp_dir))
        
        # Test with non-existent directory (should create it)
        new_temp_dir = os.path.join(self.temp_dir.name, "new_temp")
        fs_new = FileService(temp_dir=new_temp_dir)
        self.assertTrue(os.path.exists(new_temp_dir))

    def test_prepare_preview_image_with_pil_image(self):
        """Test prepare_preview_image with PIL Image input."""
        # Create a test PIL image
        test_image = Image.new('RGB', (100, 100), color='red')
        test_image.format = 'PNG'
        
        preview_path = self.file_service.prepare_preview_image(test_image, "test_preview")
        
        self.assertIsNotNone(preview_path)
        if preview_path is not None:
            self.assertTrue(os.path.exists(preview_path))
            self.assertTrue(preview_path.startswith(self.test_temp_dir))
            self.assertIn("test_preview", preview_path)
            self.assertTrue(preview_path.endswith('.png'))
            
            # Verify the saved image can be opened
            loaded_image = Image.open(preview_path)
            self.assertEqual(loaded_image.size, (100, 100))

    def test_prepare_preview_image_with_file_path(self):
        """Test prepare_preview_image with file path input."""
        # Create a test image file
        test_image_path = os.path.join(self.test_temp_dir, "test_input.jpg")
        Image.new('RGB', (50, 50), color='blue').save(test_image_path)
        
        preview_path = self.file_service.prepare_preview_image(test_image_path)
        
        self.assertEqual(preview_path, test_image_path)
        if preview_path is not None:
            self.assertTrue(os.path.exists(preview_path))

    def test_prepare_preview_image_with_invalid_input(self):
        """Test prepare_preview_image with invalid input."""
        # Test with non-existent file
        preview_path = self.file_service.prepare_preview_image("non_existent.png")
        self.assertIsNone(preview_path)
        
        # Test with invalid type
        preview_path = self.file_service.prepare_preview_image(cast(Image.Image, 123))
        self.assertIsNone(preview_path)

    def test_save_processed_image_success(self):
        """Test save_processed_image with valid inputs."""
        # Create a test PIL image
        test_image = Image.new('RGB', (80, 80), color='green')
        test_image.format = 'PNG'
        
        saved_path = self.file_service.save_processed_image(
            test_image, 
            "test_processed.png", 
            self.test_output_dir
        )
        
        self.assertIsNotNone(saved_path)
        if saved_path is not None:
            self.assertTrue(os.path.exists(saved_path))
            self.assertTrue(saved_path.startswith(self.test_output_dir))
            self.assertIn("test_processed", saved_path)
            
            # Verify the saved image
            loaded_image = Image.open(saved_path)
            self.assertEqual(loaded_image.size, (80, 80))

    def test_save_processed_image_filename_collision(self):
        """Test save_processed_image handles filename collisions."""
        test_image = Image.new('RGB', (60, 60), color='yellow')
        test_image.format = 'PNG'
        
        # Save first image
        saved_path1 = self.file_service.save_processed_image(
            test_image, 
            "collision_test.png", 
            self.test_output_dir
        )
        
        # Save second image with same name
        saved_path2 = self.file_service.save_processed_image(
            test_image, 
            "collision_test.png", 
            self.test_output_dir
        )
        
        self.assertIsNotNone(saved_path1)
        self.assertIsNotNone(saved_path2)
        self.assertNotEqual(saved_path1, saved_path2)
        if saved_path1 is not None:
            self.assertTrue(os.path.exists(saved_path1))
        if saved_path2 is not None:
            self.assertTrue(os.path.exists(saved_path2))
            self.assertIn("collision_test_1", saved_path2)

    def test_save_processed_image_invalid_input(self):
        """Test save_processed_image with invalid inputs."""
        # Test with non-PIL object
        saved_path = self.file_service.save_processed_image(
            cast(Image.Image, "not_an_image"), 
            "test.png", 
            self.test_output_dir
        )
        self.assertIsNone(saved_path)

    def test_save_processed_image_create_output_dir(self):
        """Test save_processed_image creates output directory if it doesn't exist."""
        test_image = Image.new('RGB', (40, 40), color='purple')
        test_image.format = 'PNG'
        
        new_output_dir = os.path.join(self.temp_dir.name, "new_output")
        self.assertFalse(os.path.exists(new_output_dir))
        
        saved_path = self.file_service.save_processed_image(
            test_image, 
            "test_new_dir.png", 
            new_output_dir
        )
        
        self.assertIsNotNone(saved_path)
        self.assertTrue(os.path.exists(new_output_dir))
        if saved_path is not None:
            self.assertTrue(os.path.exists(saved_path))

    def test_is_url_method(self):
        """Test the _is_url internal method."""
        # Test valid URLs
        self.assertTrue(self.file_service._is_url("https://example.com/image.png"))
        self.assertTrue(self.file_service._is_url("http://test.org/photo.jpg"))
        self.assertTrue(self.file_service._is_url("ftp://files.com/pic.gif"))
        
        # Test invalid URLs
        self.assertFalse(self.file_service._is_url("not_a_url"))
        self.assertFalse(self.file_service._is_url("/local/path/image.png"))
        self.assertFalse(self.file_service._is_url("relative/path.jpg"))
        self.assertFalse(self.file_service._is_url(""))

    @patch('requests.get')
    def test_download_image_success(self, mock_get):
        """Test _download_image with successful download."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'image/png'}
        mock_response.iter_content.return_value = [b'fake_image_data']
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        downloaded_path = self.file_service._download_image("https://example.com/test.png")
        
        self.assertIsNotNone(downloaded_path)
        if downloaded_path is not None:
            self.assertTrue(os.path.exists(downloaded_path))
            self.assertTrue(downloaded_path.endswith('.png'))
        mock_get.assert_called_once()

    @patch('requests.get')
    def test_download_image_failure(self, mock_get):
        """Test _download_image with failed download."""
        # Mock failed response
        mock_get.side_effect = Exception("Network error")
        
        downloaded_path = self.file_service._download_image("https://invalid-url.com/test.png")
        
        self.assertIsNone(downloaded_path)

    def test_handle_input_path_local_file(self):
        """Test handle_input_path with local file."""
        # Create a test file
        test_file_path = os.path.join(self.test_temp_dir, "local_test.jpg")
        Image.new('RGB', (30, 30), color='orange').save(test_file_path)
        
        result = self.file_service.handle_input_path(test_file_path)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], test_file_path)

    def test_handle_input_path_directory(self):
        """Test handle_input_path with directory (currently returns empty list)."""
        result = self.file_service.handle_input_path(self.test_temp_dir)
        
        self.assertEqual(len(result), 0)

    @patch.object(FileService, '_download_image')
    def test_handle_input_path_url(self, mock_download):
        """Test handle_input_path with URL."""
        mock_download.return_value = "/fake/downloaded/path.jpg"
        
        result = self.file_service.handle_input_path("https://example.com/image.jpg")
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], "/fake/downloaded/path.jpg")
        mock_download.assert_called_once_with("https://example.com/image.jpg")

    @patch.object(FileService, '_download_image')
    def test_handle_input_path_url_download_failure(self, mock_download):
        """Test handle_input_path with URL download failure."""
        mock_download.return_value = None
        
        result = self.file_service.handle_input_path("https://example.com/image.jpg")
        
        self.assertEqual(len(result), 0)

    def test_handle_input_path_invalid(self):
        """Test handle_input_path with invalid input."""
        # Test with non-existent file
        result = self.file_service.handle_input_path("non_existent_file.png")
        self.assertEqual(len(result), 0)
        
        # Test with None (cast to avoid type error)
        result = self.file_service.handle_input_path(cast(str, None))
        self.assertEqual(len(result), 0)

    def test_filename_sanitization(self):
        """Test filename sanitization in save_processed_image."""
        test_image = Image.new('RGB', (20, 20), color='cyan')
        test_image.format = 'PNG'
        
        # Test with problematic filename
        saved_path = self.file_service.save_processed_image(
            test_image, 
            "../../dangerous/../filename<>|?.png", 
            self.test_output_dir
        )
        
        self.assertIsNotNone(saved_path)
        if saved_path is not None:
            self.assertTrue(os.path.exists(saved_path))
            # Verify the filename was sanitized
            filename = os.path.basename(saved_path)
            self.assertNotIn("../", filename)
            self.assertNotIn("<", filename)
            self.assertNotIn(">", filename)

if __name__ == '__main__':
    unittest.main()
