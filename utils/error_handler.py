import logging
import traceback
from typing import Optional, Callable, Any

# Custom Exception Classes
class WaifucBaseError(Exception):
    """Base class for custom exceptions in this application."""
    def __init__(self, message, context=None):
        super().__init__(message)
        self.message = message
        self.context = context

    def __str__(self):
        if self.context:
            return f"{self.__class__.__name__}: {self.message} (Context: {self.context})"
        return f"{self.__class__.__name__}: {self.message}"

class DirectoryError(WaifucBaseError):
    """Custom exception for directory-related errors."""
    pass

class ImageProcessingError(WaifucBaseError):
    """Custom exception for image processing errors."""
    pass

class ModelError(WaifucBaseError):
    """Custom exception for model loading or execution errors."""
    pass

class ConfigError(WaifucBaseError):
    """Custom exception for configuration errors."""
    pass

def handle_exception(exc_type, exc_value, exc_traceback, logger_instance: Optional[logging.Logger], context: str = "General"):
    """
    通用異常處理函數。
    """
    if logger_instance:
        tb_str = "".join(traceback.format_tb(exc_traceback))
        error_message = (
            f"Unhandled exception in {context}:\\n"
            f"Type: {exc_type.__name__}\\n"
            f"Value: {exc_value}\\n"
            f"Traceback:\\n{tb_str}"
        )
        logger_instance.error(error_message)
    else:
        # 如果沒有 logger，至少打印到 stderr
        print(f"CRITICAL ERROR in {context} (logger not available):")
        traceback.print_exception(exc_type, exc_value, exc_traceback)

def safe_execute(
    func: Callable[..., Any],
    *args: Any,
    logger: Optional[logging.Logger] = None,
    default_return: Any = None,
    error_msg_prefix: str = "Error executing function",
    **kwargs: Any
) -> Any:
    """
    安全地執行一個函數，捕獲異常，記錄錯誤，並返回預設值。
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if logger:
            logger.error(f"{error_msg_prefix} '{func.__name__}': {e}", exc_info=True)
        else:
            print(f"ERROR: {error_msg_prefix} '{func.__name__}': {e}")
            traceback.print_exc()
        return default_return
