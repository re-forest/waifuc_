# services/validator_service.py
import os
from PIL import Image
from config import settings
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
        removed_count = 0 # 在此服務中，我們不實際刪除，僅記錄
        valid_image_paths = []
        invalid_image_paths = []
        
        if not os.path.isdir(image_path_or_dir):
            message = f"Error: Directory not found at {image_path_or_dir}"
            logger.error(message)
            return False, message, []

        for filename in os.listdir(image_path_or_dir):
            file_path = os.path.join(image_path_or_dir, filename)
            if not os.path.isfile(file_path):
                continue

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
                # 根據配置決定是否 "移除" (在此僅記錄)
                # removed_count +=1 

        message = f"Directory validation complete for {image_path_or_dir}."
        logger.info(f"{message} Processed: {processed_count}, Valid: {len(valid_image_paths)}, Invalid: {len(invalid_image_paths)}")
        
        if not valid_image_paths and processed_count > 0 :
             return False, f"{message} No valid images found. Processed: {processed_count}, Invalid: {len(invalid_image_paths)}.", []
        elif not valid_image_paths and processed_count == 0:
            return False, f"{message} No images found to process in the directory.", []

        return True, f"{message} Processed: {processed_count}, Valid: {len(valid_image_paths)}, Invalid: {len(invalid_image_paths)}.", valid_image_paths
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
