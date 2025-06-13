"""
統一錯誤處理模組
提供專案中所有模組使用的標準化錯誤處理和異常定義
"""

import os
import sys
from typing import Optional, Union, Any
from logger_config import get_logger

# 獲取錯誤處理專用的日誌記錄器
error_logger = get_logger('error_handler')

class WaifucError(Exception):
    """Waifuc 專案的基礎異常類別"""
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[dict] = None):
        self.message = message
        self.error_code = error_code or "WAIFUC_ERROR"
        self.details = details or {}
        super().__init__(self.message)

class DirectoryError(WaifucError):
    """目錄相關錯誤"""
    def __init__(self, message: str, path: str):
        super().__init__(message, "DIR_ERROR", {"path": path})
        self.path = path

class ImageProcessingError(WaifucError):
    """圖像處理相關錯誤"""
    def __init__(self, message: str, image_path: Optional[str] = None):
        details = {"image_path": image_path} if image_path else {}
        super().__init__(message, "IMG_PROC_ERROR", details)
        self.image_path = image_path

class ModelError(WaifucError):
    """模型載入或執行錯誤"""
    def __init__(self, message: str, model_name: Optional[str] = None):
        details = {"model_name": model_name} if model_name else {}
        super().__init__(message, "MODEL_ERROR", details)
        self.model_name = model_name

class ConfigurationError(WaifucError):
    """配置錯誤"""
    def __init__(self, message: str, config_key: Optional[str] = None):
        details = {"config_key": config_key} if config_key else {}
        super().__init__(message, "CONFIG_ERROR", details)
        self.config_key = config_key

def safe_execute(func, *args, logger=None, default_return=None, error_msg_prefix="", **kwargs):
    """
    安全執行函數，提供統一的錯誤處理
    
    Args:
        func: 要執行的函數
        *args: 函數參數
        logger: 日誌記錄器，如果沒有提供則使用預設的
        default_return: 發生錯誤時的預設返回值
        error_msg_prefix: 錯誤訊息前綴
        **kwargs: 函數關鍵字參數
    
    Returns:
        函數執行結果或預設返回值
    """
    if logger is None:
        logger = error_logger
    
    try:
        return func(*args, **kwargs)
    except WaifucError as e:
        # 專案自定義錯誤
        error_msg = f"{error_msg_prefix}發生 {e.error_code} 錯誤: {e.message}"
        if e.details:
            error_msg += f" 詳細資訊: {e.details}"
        logger.error(error_msg)
        return default_return
    except FileNotFoundError as e:
        error_msg = f"{error_msg_prefix}檔案未找到: {str(e)}"
        logger.error(error_msg)
        return default_return
    except PermissionError as e:
        error_msg = f"{error_msg_prefix}權限不足: {str(e)}"
        logger.error(error_msg)
        return default_return
    except OSError as e:
        error_msg = f"{error_msg_prefix}系統錯誤: {str(e)}"
        logger.error(error_msg)
        return default_return
    except MemoryError:
        error_msg = f"{error_msg_prefix}記憶體不足，請嘗試減少批次大小或釋放系統資源"
        logger.error(error_msg)
        return default_return
    except ImportError as e:
        error_msg = f"{error_msg_prefix}模組導入錯誤: {str(e)}，請檢查依賴是否正確安裝"
        logger.error(error_msg)
        return default_return
    except Exception as e:
        error_msg = f"{error_msg_prefix}未預期的錯誤: {type(e).__name__}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return default_return

def validate_directory(path: str, create_if_missing: bool = False, check_write: bool = False) -> str:
    """
    驗證目錄路徑
    
    Args:
        path: 目錄路徑
        create_if_missing: 如果目錄不存在是否建立
        check_write: 是否檢查寫入權限
    
    Returns:
        驗證後的絕對路徑
    
    Raises:
        DirectoryError: 目錄驗證失敗
    """
    if not path:
        raise DirectoryError("目錄路徑不能為空", path or "")
    
    abs_path = os.path.abspath(path)
    
    if not os.path.exists(abs_path):
        if create_if_missing:
            try:
                os.makedirs(abs_path, exist_ok=True)
                error_logger.info(f"已建立目錄: {abs_path}")
            except OSError as e:
                raise DirectoryError(f"無法建立目錄: {str(e)}", abs_path)
        else:
            raise DirectoryError("目錄不存在", abs_path)
    
    if not os.path.isdir(abs_path):
        raise DirectoryError("路徑不是有效的目錄", abs_path)
    
    if check_write:
        test_file = os.path.join(abs_path, '.waifuc_write_test')
        try:
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
        except (OSError, PermissionError):
            raise DirectoryError("目錄沒有寫入權限", abs_path)
    
    return abs_path

def validate_image_file(file_path: str) -> bool:
    """
    驗證是否為有效的圖像檔案
    
    Args:
        file_path: 圖像檔案路徑
    
    Returns:
        是否為有效圖像
    """
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff'}
    
    if not os.path.isfile(file_path):
        return False
    
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return False
    
    try:
        from PIL import Image
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False

def get_user_friendly_error_message(error: Exception) -> str:
    """
    將技術性錯誤轉換為使用者友好的錯誤訊息
    
    Args:
        error: 異常物件
    
    Returns:
        使用者友好的錯誤訊息
    """
    if isinstance(error, WaifucError):
        return error.message
    elif isinstance(error, FileNotFoundError):
        return f"找不到檔案或資料夾: {str(error)}"
    elif isinstance(error, PermissionError):
        return f"權限不足，無法存取: {str(error)}"
    elif isinstance(error, MemoryError):
        return "系統記憶體不足，請嘗試關閉其他程式或減少處理檔案數量"
    elif isinstance(error, ImportError):
        return f"缺少必要的程式庫，請檢查安裝: {str(error)}"
    elif "CUDA" in str(error) or "GPU" in str(error):
        return "GPU 相關錯誤，請檢查 CUDA 安裝或嘗試使用 CPU 模式"
    elif "model" in str(error).lower():
        return "AI 模型載入或執行出現問題，請檢查模型檔案或網路連線"
    else:
        return f"發生未預期的錯誤: {str(error)}"
