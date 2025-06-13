import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

class LoggerConfig:
    """日誌系統配置類別 (單例模式)"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LoggerConfig, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        # 在測試環境中或實例被重置後重新初始化
        if hasattr(self, 'log_level') and not os.getenv('FORCE_REINIT', '').lower() == 'true':
            return
            
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
        # 驗證日誌級別是否有效
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        self.log_level = log_level if log_level in valid_levels else 'INFO'
        self.log_to_file = os.getenv('LOG_TO_FILE', 'true').lower() == 'true'
        self.log_directory = os.getenv('LOG_DIRECTORY', 'logs')
        self.log_format = os.getenv('LOG_FORMAT', 
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')
        self.date_format = '%Y-%m-%d %H:%M:%S'
        
        # 確保日誌目錄存在
        if self.log_to_file:
            Path(self.log_directory).mkdir(exist_ok=True)
            
        LoggerConfig._initialized = True
    
    @classmethod
    def reset_instance(cls):
        """重置單例實例 (主要用於測試)"""
        cls._instance = None
        cls._initialized = False
    
    def setup_logger(self, name: Optional[str] = None) -> logging.Logger:
        """
        設定並返回日誌記錄器
        
        Args:
            name: 日誌記錄器名稱，預設為調用模組名稱
            
        Returns:
            配置好的日誌記錄器
        """
        if name is None:
            # 獲取調用此方法的模組名稱
            frame = sys._getframe(1)
            name = frame.f_globals.get('__name__', 'waifuc')
        
        logger = logging.getLogger(name)
        
        # 避免重複設定
        if logger.handlers:
            return logger
        
        logger.setLevel(getattr(logging, self.log_level, logging.INFO))
        
        # 建立格式化器
        formatter = logging.Formatter(self.log_format, self.date_format)
        
        # 控制台處理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 檔案處理器（如果啟用）
        if self.log_to_file:
            # 建立日期標記的日誌檔案
            log_filename = f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
            log_filepath = os.path.join(self.log_directory, log_filename)
            
            file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
            file_handler.setLevel(getattr(logging, self.log_level, logging.INFO))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            # 錯誤專用日誌檔案
            error_filename = f"{name}_error_{datetime.now().strftime('%Y%m%d')}.log"
            error_filepath = os.path.join(self.log_directory, error_filename)
            
            error_handler = logging.FileHandler(error_filepath, encoding='utf-8')
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(formatter)
            logger.addHandler(error_handler)
        
        return logger
    
    def get_logger(self, name: Optional[str] = None) -> logging.Logger:
        """獲取已配置的日誌記錄器，如果不存在則建立"""
        return self.setup_logger(name)
    
    def setup_logging(self) -> None:
        """
        設定全域日誌系統
        
        這個方法主要用於向後相容性和測試
        """
        # 設定根日誌記錄器
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.log_level, logging.INFO))
        
        # 清除現有處理器
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # 建立格式化器
        formatter = logging.Formatter(self.log_format, self.date_format)
        
        # 控制台處理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # 檔案處理器（如果啟用）
        if self.log_to_file:
            # 建立日期標記的日誌檔案
            log_filename = f"waifuc_{datetime.now().strftime('%Y%m%d')}.log"
            log_filepath = os.path.join(self.log_directory, log_filename)
            
            file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
            file_handler.setLevel(getattr(logging, self.log_level, logging.INFO))
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            
            # 錯誤專用日誌檔案
            error_filename = f"waifuc_error_{datetime.now().strftime('%Y%m%d')}.log"
            error_filepath = os.path.join(self.log_directory, error_filename)
            
            error_handler = logging.FileHandler(error_filepath, encoding='utf-8')
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(formatter)
            root_logger.addHandler(error_handler)

# 全域日誌配置實例
logger_config = LoggerConfig()

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    獲取日誌記錄器的便利函數
    
    Args:
        name: 日誌記錄器名稱
        
    Returns:
        配置好的日誌記錄器
    """
    return logger_config.get_logger(name)

# 為向後相容性提供的預設日誌記錄器
logger = get_logger('waifuc')
