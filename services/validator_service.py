# services/validator_service.py
import os
from PIL import Image
from config import settings as default_settings
from utils.error_handler import safe_execute

def _validate_single_image_internal(image_path, logger):
    """
    內部輔助函數，用於驗證單一圖片。
    """
    try:
        img = Image.open(image_path)
        img.verify()  # 驗證圖片檔案的完整性
        # 重新開啟圖片以進行後續操作，因為 verify() 會破壞圖片物件
        img = Image.open(image_path)
        img.load() # 確保圖片資料已載入
        logger.info(f"Image {image_path} is valid.")
        return True
    except Exception as e:
        logger.error(f"Invalid image {image_path}: {e}")
        return False

def validate_image_service(image_path_or_dir, logger, config=None, is_directory=False):
    logger.info(f"[ValidatorService] Starting validation for: {image_path_or_dir}")

    if is_directory:
        processed_count = 0
        removed_count = 0
        valid_image_paths = []
        invalid_image_paths = []
        
        if not os.path.isdir(image_path_or_dir):
            message = f"Error: Directory not found at {image_path_or_dir}"
            logger.error(message)
            return False, message, []

        # 支持遞歸掃描
        def scan_directory(directory):
            all_files = []
            for root, dirs, files in os.walk(directory):
                for filename in files:
                    file_path = os.path.join(root, filename)
                    # 檢查是否為圖片文件
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp', '.tiff')):
                        all_files.append(file_path)
            return all_files

        image_files = scan_directory(image_path_or_dir)
        
        for file_path in image_files:
            processed_count += 1
            is_valid = safe_execute(
                _validate_single_image_internal,
                file_path,
                logger,
                logger=logger,
                default_return=False,
                error_msg_prefix=f"Validating image {file_path} in directory"
            )
            if is_valid:
                valid_image_paths.append(file_path)
            else:
                invalid_image_paths.append(file_path)
                # 可選：移動無效圖片到隔離資料夾
                if config and getattr(config, 'VALIDATION_QUARANTINE_INVALID', False):
                    quarantine_dir = getattr(config, 'VALIDATION_QUARANTINE_DIR', 
                                           os.path.join(os.path.dirname(image_path_or_dir), 'invalid_images'))
                    try:
                        os.makedirs(quarantine_dir, exist_ok=True)
                        quarantine_path = os.path.join(quarantine_dir, os.path.basename(file_path))
                        import shutil
                        shutil.move(file_path, quarantine_path)
                        removed_count += 1
                        logger.info(f"[ValidatorService] Moved invalid image to quarantine: {quarantine_path}")
                    except Exception as e:
                        logger.error(f"[ValidatorService] Failed to quarantine invalid image {file_path}: {e}")

        message = f"Directory validation complete for {image_path_or_dir}."
        logger.info(f"{message} Processed: {processed_count}, Valid: {len(valid_image_paths)}, Invalid: {len(invalid_image_paths)}, Quarantined: {removed_count}")
        
        if not valid_image_paths and processed_count > 0:
             return False, f"{message} No valid images found. Processed: {processed_count}, Invalid: {len(invalid_image_paths)}.", []
        elif not valid_image_paths and processed_count == 0:
            return False, f"{message} No images found to process in the directory.", []

        return True, f"{message} Processed: {processed_count}, Valid: {len(valid_image_paths)}, Invalid: {len(invalid_image_paths)}, Quarantined: {removed_count}.", valid_image_paths
    else:
        # 處理單一圖片
        if not os.path.isfile(image_path_or_dir):
            message = f"Error: Image file not found at {image_path_or_dir}"
            logger.error(message)
            return False, message, []
            
        is_valid = safe_execute(
            _validate_single_image_internal,
            image_path_or_dir,
            logger,
            logger=logger,
            default_return=False,
            error_msg_prefix=f"Validating single image {image_path_or_dir} in service"
        )
        if is_valid:
            logger.info(f"Image {image_path_or_dir} validated successfully by service.")
            return True, "Image validated successfully.", [image_path_or_dir]
        else:
            logger.error(f"Image {image_path_or_dir} failed validation in service.")
            return False, "Image validation failed.", []

def validate_image(image_pil: Image.Image, logger, config=None):
    """
    Entry point for orchestrator - validates a PIL image.
    Returns: (is_valid, message_or_pil, path_list)
    """
    import tempfile
    import os
    
    try:
        # Create temporary file to validate the PIL image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_path = temp_file.name
            image_pil.save(temp_path, 'PNG')
        
        # Use existing validation logic
        is_valid, message, path_list = validate_image_service(temp_path, logger, config, is_directory=False)
        
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except:
            pass
            
        if is_valid:
            # Return the original PIL image for success
            return True, image_pil, []
        else:
            return False, message, []
            
    except Exception as e:
        logger.error(f"[ValidatorService] Error in validate_image entry: {e}")
        return False, f"Validation error: {str(e)}", []
