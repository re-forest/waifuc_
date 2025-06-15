import logging
import os
from logging.handlers import RotatingFileHandler
import datetime

def setup_logging(module_name: str,
                  log_dir: str,
                  log_level_str: str = "INFO",
                  max_bytes: int = 10*1024*1024,
                  backup_count: int = 5):
    logger = logging.getLogger(module_name)

    # 防止重複添加 handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    try:
        numeric_level = getattr(logging, log_level_str.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f'Invalid log level: {log_level_str}')
        logger.setLevel(numeric_level)
    except ValueError:
        logger.setLevel(logging.INFO) # 預設為 INFO
        print(f"Warning: Invalid log level '{log_level_str}'. Defaulting to INFO.")

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)')

    timestamp = datetime.datetime.now().strftime("%Y%m%d")

    # 一般日誌檔案
    log_file = os.path.join(log_dir, f"{module_name.replace('.', '_')}_{timestamp}.log")
    file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 錯誤日誌檔案
    error_log_file = os.path.join(log_dir, f"{module_name.replace('.', '_')}_error_{timestamp}.log")
    error_file_handler = RotatingFileHandler(error_log_file, maxBytes=max_bytes, backupCount=backup_count, encoding='utf-8')
    error_file_handler.setFormatter(formatter)
    error_file_handler.setLevel(logging.ERROR) # 只記錄 ERROR 及以上級別
    logger.addHandler(error_file_handler)
    
    return logger
