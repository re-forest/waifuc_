"""
FileService for handling file-related operations such as preparing previews,
saving processed images, and handling input paths.
"""
import os
from PIL import Image
import tempfile
import requests
from urllib.parse import urlparse
import binascii # Added for random name generation

from config.settings import GRADIO_TEMP_DIR # Assuming GRADIO_TEMP_DIR is defined in settings
from utils.logger_config import setup_logging # Changed get_logger to setup_logging

logger = setup_logging(__name__, 'logs') # Assuming 'logs' is the desired log directory for this service

class FileService:
    def __init__(self, temp_dir=None):
        self.temp_dir = temp_dir or GRADIO_TEMP_DIR
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir, exist_ok=True)
        logger.info(f"FileService initialized with temp_dir: {self.temp_dir}")

    def prepare_preview_image(self, image_input, output_filename_prefix="preview"):
        """
        Prepares an image for preview in Gradio.
        If input is a PIL Image, saves it to a temporary directory.
        If input is a path, checks if it's a valid image file.
        Returns a safe path for Gradio to display.

        Args:
            image_input (Image.Image | str): PIL Image object or path to an image.
            output_filename_prefix (str): Prefix for the temporary filename if saving a PIL image.

        Returns:
            str | None: Path to the (potentially temporary) image file for preview, or None if invalid.
        """
        try:
            if isinstance(image_input, Image.Image):
                img_format = image_input.format or 'PNG' # Default to PNG if format is not available
                # Generate a random name for the temporary file
                random_suffix = binascii.hexlify(os.urandom(4)).decode()
                temp_filename = f"{output_filename_prefix}_{random_suffix}.{img_format.lower()}"
                temp_file_path = os.path.join(self.temp_dir, temp_filename)
                image_input.save(temp_file_path)
                logger.debug(f"PIL Image saved to temporary preview path: {temp_file_path}")
                return temp_file_path
            elif isinstance(image_input, str) and os.path.isfile(image_input):
                # For now, assume the path is safe if it's a file.
                # Further security checks (e.g., allowed paths) could be added here or in settings.
                logger.debug(f"Using existing image path for preview: {image_input}")
                return image_input
            else:
                logger.warning(f"Invalid image input for preview: {image_input}")
                return None
        except Exception as e:
            logger.error(f"Error preparing preview image: {e}", exc_info=True)
            return None

    def save_processed_image(self, pil_image, original_filename, output_dir):
        """
        Saves a processed PIL Image object to the specified output directory.

        Args:
            pil_image (Image.Image): The PIL Image to save.
            original_filename (str): The original filename, used to derive the new filename.
            output_dir (str): The directory to save the image to.

        Returns:
            str | None: Path to the saved image, or None if saving failed.
        """
        if not isinstance(pil_image, Image.Image):
            logger.error("Invalid input: pil_image must be a PIL.Image object.")
            return None
        if not os.path.isdir(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
                logger.info(f"Created output directory: {output_dir}")
            except OSError as e:
                logger.error(f"Error creating output directory {output_dir}: {e}", exc_info=True)
                return None

        base, ext = os.path.splitext(original_filename)
        # Consider adding a suffix like '_processed' or a timestamp if needed
        # For now, it saves with the original name in the new directory.
        # Ensure the extension is appropriate for the image format.
        img_format = pil_image.format or 'PNG'
        if not ext or ext.lower().strip('.') != img_format.lower():
            ext = f".{img_format.lower()}"
            
        # Sanitize base filename to prevent path traversal or invalid characters
        safe_base = "".join(c for c in base if c.isalnum() or c in (' ', '.', '_')).rstrip()
        if not safe_base: # if original filename was all special chars
            safe_base = "processed_image"
            
        output_filename = f"{safe_base}{ext}"
        output_path = os.path.join(output_dir, output_filename)
        
        counter = 1
        while os.path.exists(output_path): # Avoid overwriting
            output_filename = f"{safe_base}_{counter}{ext}"
            output_path = os.path.join(output_dir, output_filename)
            counter += 1
            if counter > 100: # Safety break
                logger.error(f"Could not find a unique filename after 100 attempts for {original_filename} in {output_dir}")
                return None


        try:
            pil_image.save(output_path)
            logger.info(f"Processed image saved to: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error saving processed image to {output_path}: {e}", exc_info=True)
            return None

    def _is_url(self, path_or_url):
        """Checks if the given string is a URL."""
        try:
            result = urlparse(path_or_url)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    def _download_image(self, url):
        """Downloads an image from a URL and saves it to a temporary file."""
        try:
            response = requests.get(url, stream=True, timeout=10) # Added timeout
            response.raise_for_status() # Raise an exception for HTTP errors
            
            # Try to get a reasonable filename from URL or content type
            content_type = response.headers.get('content-type')
            extension = '.jpg' # default
            if content_type:
                if 'jpeg' in content_type:
                    extension = '.jpg'
                elif 'png' in content_type:
                    extension = '.png'
                elif 'gif' in content_type:
                    extension = '.gif'
                # Add more mimetypes as needed
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=extension, dir=self.temp_dir)
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            temp_file.close()
            logger.info(f"Image downloaded from {url} to {temp_file.name}")
            return temp_file.name
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading image from {url}: {e}", exc_info=True)
            return None
        except Exception as e: # Catch other potential errors
            logger.error(f"An unexpected error occurred while downloading {url}: {e}", exc_info=True)
            return None


    def handle_input_path(self, input_path_or_url):
        """
        Handles an input path or URL. If it's a URL, downloads the image.
        If it's a local path, validates it.
        Returns a list of local file paths. For a single image, it's a list with one item.
        (Future: Could be extended to handle directories)

        Args:
            input_path_or_url (str): Path or URL to an image.

        Returns:
            list[str]: List containing the local file path, or an empty list if invalid/error.
        """
        if self._is_url(input_path_or_url):
            logger.info(f"Input is a URL: {input_path_or_url}. Attempting to download.")
            local_path = self._download_image(input_path_or_url)
            return [local_path] if local_path else []
        elif isinstance(input_path_or_url, str) and os.path.isfile(input_path_or_url):
            # Basic validation: check if it's a file.
            # More robust validation (e.g., file type, readability) could be added.
            logger.info(f"Input is a local file path: {input_path_or_url}")
            return [input_path_or_url]
        elif isinstance(input_path_or_url, str) and os.path.isdir(input_path_or_url):
            # Placeholder for directory handling - currently returns empty
            logger.warning(f"Input is a directory (not yet fully supported for individual processing): {input_path_or_url}")
            # For now, we will return an empty list.
            # Future: list image files in the directory.
            # image_files = []
            # supported_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp']
            # for f_name in os.listdir(input_path_or_url):
            #     if any(f_name.lower().endswith(ext) for ext in supported_extensions):
            #         image_files.append(os.path.join(input_path_or_url, f_name))
            # if not image_files:
            #    logger.warning(f"No supported image files found in directory: {input_path_or_url}")
            # return image_files
            return []
        else:
            logger.error(f"Invalid input path or URL: {input_path_or_url}")
            return []

# Example Usage (for testing purposes, typically not run directly from here)
if __name__ == '__main__':
    # Ensure config.settings.GRADIO_TEMP_DIR is usable or mock it
    # from config import settings
    # settings.GRADIO_TEMP_DIR = "h:/python_project/test_waifuc/waifuc_/temp_gradio_previews" # Example
    
    fs = FileService(temp_dir="h:/python_project/test_waifuc/waifuc_/temp_gradio_previews_test") # Override for testing
    
    # Test prepare_preview_image with a PIL Image
    try:
        img = Image.new('RGB', (60, 30), color = 'red')
        preview_path_pil = fs.prepare_preview_image(img)
        print(f"PIL Preview path: {preview_path_pil}")
        if preview_path_pil and os.path.exists(preview_path_pil):
            print(f"PIL Preview image created successfully at {preview_path_pil}")
        else:
            print("Failed to create PIL preview image.")
    except Exception as e:
        print(f"Error testing PIL preview: {e}")

    # Test prepare_preview_image with an existing file path
    # Create a dummy file for testing
    dummy_image_path = os.path.join(fs.temp_dir, "dummy_test_image.png")
    try:
        if not os.path.exists(fs.temp_dir):
            os.makedirs(fs.temp_dir)
        Image.new('RGB', (50, 50), color = 'blue').save(dummy_image_path)
        preview_path_file = fs.prepare_preview_image(dummy_image_path)
        print(f"File Preview path: {preview_path_file}")
        if preview_path_file == dummy_image_path:
            print("File preview path returned correctly.")
        else:
            print("Failed to get file preview path.")
    except Exception as e:
        print(f"Error testing file preview: {e}")


    # Test save_processed_image
    output_test_dir = "h:/python_project/test_waifuc/waifuc_/output_test_images"
    try:
        img_to_save = Image.new('RGB', (100, 100), color = 'green')
        img_to_save.format = 'PNG' # Set format for consistent extension
        saved_path = fs.save_processed_image(img_to_save, "my_processed_image.png", output_test_dir)
        print(f"Saved image path: {saved_path}")
        if saved_path and os.path.exists(saved_path):
            print(f"Image saved successfully to {saved_path}")
        else:
            print("Failed to save image.")
        
        # Test saving again to check unique naming
        saved_path_2 = fs.save_processed_image(img_to_save, "my_processed_image.png", output_test_dir)
        print(f"Saved image path (2nd time): {saved_path_2}")
        if saved_path_2 and os.path.exists(saved_path_2) and saved_path_2 != saved_path:
            print(f"Image saved successfully with unique name to {saved_path_2}")
        else:
            print("Failed to save image with unique name.")

    except Exception as e:
        print(f"Error testing save_processed_image: {e}")

    # Test handle_input_path with a URL
    # (Ensure this URL is active or replace with a reliable one for testing)
    test_url = "https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png" 
    downloaded_paths = fs.handle_input_path(test_url)
    print(f"Downloaded paths from URL: {downloaded_paths}")
    if downloaded_paths and os.path.exists(downloaded_paths[0]):
        print(f"Image downloaded successfully from URL to {downloaded_paths[0]}")
        # os.remove(downloaded_paths[0]) # Clean up test download
    else:
        print("Failed to download image from URL.")

    # Test handle_input_path with a local file
    local_file_paths = fs.handle_input_path(dummy_image_path) # Use the dummy image created earlier
    print(f"Local file paths: {local_file_paths}")
    if local_file_paths and local_file_paths[0] == dummy_image_path:
        print("Local file path handled correctly.")
    else:
        print("Failed to handle local file path.")

    # Test handle_input_path with a directory (expected to return empty list for now)
    dir_paths = fs.handle_input_path(fs.temp_dir)
    print(f"Directory input paths: {dir_paths}")
    if not dir_paths:
        print("Directory input handled as expected (returned empty list).")
    else:
        print(f"Directory input failed, returned: {dir_paths}")
        
    # Test with invalid input
    invalid_paths = fs.handle_input_path("non_existent_file.png")
    print(f"Invalid input paths: {invalid_paths}")
    if not invalid_paths:
        print("Invalid input handled correctly.")
    else:
        print("Failed to handle invalid input.")
        
    print("FileService tests completed. Check logs and output directories.")
