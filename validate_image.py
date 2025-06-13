import os
from PIL import Image
from tqdm import tqdm
from logger_config import get_logger
from error_handler import safe_execute, DirectoryError, ImageProcessingError, WaifucError

# 設定日誌記錄器
logger = get_logger('validate_image')

# 支援的影像格式
SUPPORTED_EXT = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".webp",
]

def validate_single_image(file_path):
    """
    驗證單一圖片檔案的完整性
    
    Args:
        file_path (str): 圖片檔案路徑
    
    Returns:
        bool: 圖片是否有效
    """
    try:
        with Image.open(file_path) as img:
            img.load()
        return True
    except Exception as e:
        logger.debug(f"圖片 {file_path} 驗證失敗: {str(e)}")
        return False

def validate_and_remove_invalid_images(directory_path):
    """
    驗證目錄中所有圖片的完整性，並刪除無效或損壞的圖片
    
    Args:
        directory_path (str): 圖片目錄路徑
    
    Returns:
        tuple: 包含處理的總檔案數和刪除的檔案數
    
    Raises:
        DirectoryError: 當目錄不存在或無法存取時
        WaifucError: 當處理過程中發生其他錯誤時
    """
    logger.info(f"開始驗證目錄中的圖片: {directory_path}")
    
    # 驗證目錄是否存在
    if not os.path.isdir(directory_path):
        raise DirectoryError(f"目錄 '{directory_path}' 不存在或不是有效目錄", directory_path)
    
    # 檢查目錄是否可讀取
    if not os.access(directory_path, os.R_OK):
        raise DirectoryError(f"無法讀取目錄 '{directory_path}'，請檢查權限", directory_path)
    
    try:
        files = os.listdir(directory_path)
    except PermissionError:
        raise DirectoryError(f"權限不足，無法列舉目錄 '{directory_path}' 中的檔案", directory_path)
    except Exception as e:
        raise WaifucError(f"列舉目錄檔案時發生錯誤: {str(e)}")
    
    removed_count = 0
    valid_image_files = []
    
    # 過濾出支援的圖片檔案
    for filename in files:
        ext = os.path.splitext(filename)[-1].lower()
        if ext in SUPPORTED_EXT:
            valid_image_files.append(filename)
    
    logger.info(f"找到 {len(valid_image_files)} 個圖片檔案進行驗證")
    
    for filename in tqdm(valid_image_files, desc="驗證圖片進度"):
        file_path = os.path.join(directory_path, filename)
        
        if not os.path.isfile(file_path):
            logger.warning(f"檔案 {file_path} 不存在，跳過")
            continue
        
        # 使用 safe_execute 安全驗證圖片
        is_valid = safe_execute(
            validate_single_image,
            file_path,
            logger=logger,
            default_return=False,
            error_msg_prefix=f"驗證圖片 {filename} 時"
        )
        
        if not is_valid:
            # 嘗試移除損壞的圖片
            remove_result = safe_execute(
                os.remove,
                file_path,
                logger=logger,
                default_return=False,
                error_msg_prefix=f"移除損壞圖片 {filename} 時"
            )
            
            if remove_result is not False:
                removed_count += 1
                logger.info(f"已移除損壞圖片: {filename}")
            else:
                logger.error(f"無法移除損壞圖片: {filename}")
    
    total_processed = len(valid_image_files)
    logger.info(f"圖片驗證完成，共處理 {total_processed} 張圖片，移除 {removed_count} 張無效圖片")
    return total_processed, removed_count

if __name__ == "__main__":
    from dotenv import load_dotenv
    
    # 載入環境變數
    load_dotenv()
    
    # 從環境變數獲取目錄
    directory_path = os.getenv("directory")
    if not directory_path:
        logger.error("未設定 directory 環境變數")
        print("錯誤: 未設定 directory 環境變數")
        exit(1)
    
    # 使用 safe_execute 安全執行驗證
    result = safe_execute(
        validate_and_remove_invalid_images,
        directory_path,
        logger=logger,
        default_return=(0, 0),
        error_msg_prefix="執行圖片驗證時"
    )
    
    if result:
        total, removed = result
        logger.info(f"驗證完成: 共處理 {total} 個檔案，移除 {removed} 個無效檔案")
        print(f"驗證完成: 共處理 {total} 個檔案，移除 {removed} 個無效檔案")
    else:
        logger.error("圖片驗證失敗")
        print("圖片驗證失敗")
