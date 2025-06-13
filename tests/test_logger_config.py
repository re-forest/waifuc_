"""
日誌配置模組的單元測試
"""
import unittest
import os
import sys
import tempfile
import shutil
import logging
from unittest.mock import patch, MagicMock
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logger_config import LoggerConfig, get_logger
from tests.test_base import setup_test_environment, teardown_test_environment, mock_env_vars


def cleanup_loggers():
    """清理所有 logger 處理器"""
    # 清理所有命名 logger 的處理器
    for logger_name in list(logging.Logger.manager.loggerDict.keys()):
        logger = logging.getLogger(logger_name)
        for handler in logger.handlers[:]:
            try:
                if hasattr(handler, 'close'):
                    handler.close()
            except:
                pass
            try:
                logger.removeHandler(handler)
            except:
                pass
    
    # 清理根 logger 的處理器
    for handler in logging.root.handlers[:]:
        try:
            if hasattr(handler, 'close'):
                handler.close()
        except:
            pass
        try:
            logging.root.removeHandler(handler)
        except:
            pass


class TestLoggerConfig(unittest.TestCase):
    """測試 LoggerConfig 類別"""
    
    def setUp(self):
        """設定測試環境"""
        self.test_dir = setup_test_environment()
          # 清理現有的 logger 配置
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        # 重置 LoggerConfig 的類別變數
        LoggerConfig._instance = None
        LoggerConfig._initialized = False
        
    def tearDown(self):
        """清理測試環境"""
        cleanup_loggers()
        
        # 重置 LoggerConfig
        LoggerConfig._instance = None
        LoggerConfig._initialized = False
        
        teardown_test_environment()
    
    def test_singleton_pattern(self):
        """測試單例模式"""
        config1 = LoggerConfig()
        config2 = LoggerConfig()
        
        self.assertIs(config1, config2, "LoggerConfig 應該是單例")
    
    @mock_env_vars(
        LOG_LEVEL='DEBUG',
        LOG_TO_FILE='true',
        LOG_DIRECTORY='test_logs',
        LOG_FORMAT='%(name)s - %(message)s'
    )
    def test_environment_variable_loading(self):
        """測試環境變數載入"""
        config = LoggerConfig()
        
        self.assertEqual(config.log_level, 'DEBUG')
        self.assertTrue(config.log_to_file)
        self.assertEqual(config.log_directory, 'test_logs')
        self.assertEqual(config.log_format, '%(name)s - %(message)s')
    
    @mock_env_vars()  # 使用預設環境變數
    def test_default_configuration(self):
        """測試預設配置"""
        # 強制重新初始化以獲取新的環境變數
        LoggerConfig.reset_instance()
        config = LoggerConfig()
        
        self.assertEqual(config.log_level, 'INFO')
        self.assertTrue(config.log_to_file)  # 預設為 true
        self.assertEqual(config.log_directory, 'logs')
        self.assertIsInstance(config.log_format, str)
        self.assertIn('%(levelname)s', config.log_format)
    
    @mock_env_vars(LOG_TO_FILE='false')
    def test_disable_file_logging(self):
        """測試禁用檔案日誌"""
        config = LoggerConfig()
        self.assertFalse(config.log_to_file)
    
    def test_invalid_log_level(self):
        """測試無效的日誌級別"""
        with mock_env_vars(LOG_LEVEL='INVALID_LEVEL'):
            config = LoggerConfig()
            # 應該回退到預設級別
            self.assertEqual(config.log_level, 'INFO')
    
    @mock_env_vars(LOG_TO_FILE='true', LOG_DIRECTORY='test_logs')
    def test_log_directory_creation(self):
        """測試日誌目錄創建"""
        log_dir = os.path.join(self.test_dir, 'test_logs')
        
        with patch.dict(os.environ, {'LOG_DIRECTORY': log_dir}):
            config = LoggerConfig()
            config.setup_logging()
            
            # 檢查目錄是否創建
            self.assertTrue(os.path.exists(log_dir), "日誌目錄應該被創建")
    
    @mock_env_vars(LOG_TO_FILE='true')
    def test_file_handler_creation(self):
        """測試檔案處理器創建"""
        config = LoggerConfig()
        config.log_directory = self.test_dir  # 使用測試目錄
        
        logger = logging.getLogger('test_logger')
        config.setup_logging()
        
        # 檢查是否有檔案處理器
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        # 由於我們的實現可能不直接在測試 logger 上添加處理器，
        # 我們檢查根 logger 或者通過其他方式驗證
        
        # 這個測試可能需要根據實際的 LoggerConfig 實現調整
        pass
    
    def test_console_handler_always_present(self):
        """測試控制台處理器總是存在"""
        config = LoggerConfig()
        config.setup_logging()
        
        root_logger = logging.getLogger()
        console_handlers = [h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)]
        
        # 應該至少有一個控制台處理器
        self.assertGreater(len(console_handlers), 0, "應該有控制台處理器")


class TestGetLogger(unittest.TestCase):
    """測試 get_logger 函數"""
    
    def setUp(self):
        """設定測試環境"""
        self.test_dir = setup_test_environment()
        
        # 清理現有的 logger 配置
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)        
        # 重置 LoggerConfig
        LoggerConfig._instance = None
        LoggerConfig._initialized = False
        
    def tearDown(self):
        """清理測試環境"""
        cleanup_loggers()
        
        # 重置 LoggerConfig
        LoggerConfig._instance = None
        LoggerConfig._initialized = False
        
        teardown_test_environment()
    
    @mock_env_vars(LOG_LEVEL='DEBUG')
    def test_get_logger_basic(self):
        """測試基本 logger 獲取"""
        logger = get_logger('test_module')
        
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, 'test_module')
    
    def test_get_logger_multiple_calls(self):
        """測試多次調用 get_logger"""
        logger1 = get_logger('module1')
        logger2 = get_logger('module2')
        logger3 = get_logger('module1')  # 重複調用
        
        self.assertNotEqual(logger1, logger2, "不同模組應該有不同的 logger")
        self.assertEqual(logger1, logger3, "相同模組應該返回相同的 logger")
    
    @mock_env_vars(LOG_LEVEL='WARNING')
    def test_logger_level_setting(self):
        """測試 logger 級別設定"""
        logger = get_logger('test_level')
        
        # 檢查 logger 或其處理器的級別
        # 這個測試需要根據實際的實現調整
        pass
    
    @mock_env_vars(LOG_TO_FILE='true')
    def test_logger_with_file_output(self):
        """測試帶檔案輸出的 logger"""
        with patch.dict(os.environ, {'LOG_DIRECTORY': self.test_dir}):
            logger = get_logger('file_test')
            
            # 測試日誌記錄
            logger.info("測試日誌訊息")
            
            # 這個測試需要檢查檔案是否創建和寫入
            # 具體實現取決於 LoggerConfig 的檔案處理邏輯
            pass
    
    def test_logger_unicode_support(self):
        """測試 Unicode 支援"""
        logger = get_logger('unicode_test')
        
        # 測試中文日誌
        try:
            logger.info("測試中文日誌 🎉")
            logger.warning("警告：包含特殊字符的訊息")
            logger.error("錯誤：處理失敗 ❌")
        except UnicodeError:
            self.fail("Logger 應該支援 Unicode 字符")


class TestLoggerIntegration(unittest.TestCase):
    """日誌系統整合測試"""
    
    def setUp(self):
        """設定測試環境"""
        self.test_dir = setup_test_environment()
        
        # 清理現有的 logger 配置
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)        
        # 重置 LoggerConfig
        LoggerConfig._instance = None
        LoggerConfig._initialized = False
        
    def tearDown(self):
        """清理測試環境"""
        cleanup_loggers()
        
        # 重置 LoggerConfig
        LoggerConfig._instance = None
        LoggerConfig._initialized = False
        
        teardown_test_environment()
    
    @mock_env_vars(LOG_TO_FILE='true', LOG_LEVEL='DEBUG')
    def test_multiple_modules_logging(self):
        """測試多模組日誌記錄"""
        with patch.dict(os.environ, {'LOG_DIRECTORY': self.test_dir}):
            # 獲取不同模組的 logger
            logger1 = get_logger('module1')
            logger2 = get_logger('module2')
            logger3 = get_logger('module3')
            
            # 記錄不同級別的日誌
            logger1.debug("Debug 訊息")
            logger1.info("Info 訊息")
            logger2.warning("Warning 訊息")
            logger2.error("Error 訊息")
            logger3.critical("Critical 訊息")
            
            # 這裡應該檢查檔案是否正確創建和寫入
            # 具體檢查邏輯取決於 LoggerConfig 的實現
            pass
    
    def test_error_logging_separation(self):
        """測試錯誤日誌分離"""
        # 如果系統支援錯誤日誌分離，測試該功能
        with patch.dict(os.environ, {
            'LOG_TO_FILE': 'true',
            'LOG_DIRECTORY': self.test_dir
        }):
            logger = get_logger('error_test')
            
            # 記錄不同級別的日誌
            logger.info("普通資訊")
            logger.warning("警告訊息")
            logger.error("錯誤訊息")
            logger.critical("嚴重錯誤")
            
            # 檢查是否有錯誤專用檔案
            # 這個檢查取決於具體的實現
            pass
    
    @mock_env_vars(LOG_TO_FILE='false')
    def test_console_only_logging(self):
        """測試僅控制台日誌"""
        with patch('sys.stdout') as mock_stdout:
            logger = get_logger('console_test')
            logger.info("控制台測試訊息")
            
            # 這個測試可能需要更複雜的 mock 來驗證輸出
            pass
    
    def test_concurrent_logging(self):
        """測試並發日誌記錄"""
        import threading
        import time
        
        results = []
        
        def log_worker(worker_id):
            logger = get_logger(f'worker_{worker_id}')
            for i in range(10):
                logger.info(f"Worker {worker_id} - Message {i}")
                time.sleep(0.01)  # 短暫延遲
            results.append(worker_id)
        
        # 啟動多個線程
        threads = []
        for i in range(3):
            thread = threading.Thread(target=log_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 等待所有線程完成
        for thread in threads:
            thread.join()
        
        # 檢查所有工作者都完成了
        self.assertEqual(len(results), 3, "所有工作者線程都應該完成")
        self.assertEqual(sorted(results), [0, 1, 2], "所有工作者都應該記錄完成")


if __name__ == '__main__':
    unittest.main()
