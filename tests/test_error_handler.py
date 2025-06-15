"""
錯誤處理模組的單元測試
"""
import unittest
import os
import tempfile
import shutil
import logging
from unittest.mock import patch, MagicMock
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.error_handler import (
    WaifucBaseError, DirectoryError, ImageProcessingError, ModelError, ConfigError,
    safe_execute, handle_exception
)
from utils.logger_config import setup_logging


class TestWaifucExceptions(unittest.TestCase):
    """測試自定義異常類別"""
    
    def test_waifuc_base_error_basic(self):
        """測試基礎 WaifucBaseError"""
        error = WaifucBaseError("測試錯誤")
        self.assertEqual(str(error), "WaifucBaseError: 測試錯誤")
        self.assertEqual(error.message, "測試錯誤")
        self.assertIsNone(error.context)
    
    def test_waifuc_base_error_with_context(self):
        """測試帶有上下文的 WaifucBaseError"""
        context = {"file": "test.jpg", "line": 42}
        error = WaifucBaseError("測試錯誤", context)
        self.assertEqual(error.message, "測試錯誤")
        self.assertEqual(error.context, context)
        self.assertIn("Context:", str(error))
    
    def test_directory_error(self):
        """測試 DirectoryError"""
        error = DirectoryError("目錄不存在", {"path": "/nonexistent/path"})
        self.assertIsInstance(error, WaifucBaseError)
        self.assertEqual(error.message, "目錄不存在")
        self.assertIsNotNone(error.context)
        if error.context is not None:  # 類型保護
            self.assertEqual(error.context["path"], "/nonexistent/path")
    
    def test_image_processing_error(self):
        """測試 ImageProcessingError"""
        error = ImageProcessingError("圖像處理失敗", {"image_path": "corrupted.jpg"})
        self.assertIsInstance(error, WaifucBaseError)
        self.assertEqual(error.message, "圖像處理失敗")
        self.assertIsNotNone(error.context)
        if error.context is not None:  # 類型保護
            self.assertEqual(error.context["image_path"], "corrupted.jpg")
    
    def test_model_error(self):
        """測試 ModelError"""
        error = ModelError("模型載入失敗", {"model_name": "test_model"})
        self.assertIsInstance(error, WaifucBaseError)
        self.assertEqual(error.message, "模型載入失敗")
        self.assertIsNotNone(error.context)
        if error.context is not None:  # 類型保護
            self.assertEqual(error.context["model_name"], "test_model")
    
    def test_config_error(self):
        """測試 ConfigError"""
        error = ConfigError("配置錯誤", {"config_key": "test_setting"})
        self.assertIsInstance(error, WaifucBaseError)
        self.assertEqual(error.message, "配置錯誤")
        self.assertIsNotNone(error.context)
        if error.context is not None:  # 類型保護
            self.assertEqual(error.context["config_key"], "test_setting")


class TestSafeExecute(unittest.TestCase):
    """測試安全執行函數"""
    
    def setUp(self):
        """設定測試環境"""
        self.test_logger = setup_logging(__name__, 'test_logs', log_level_str='DEBUG')
    
    def test_safe_execute_success(self):
        """測試成功執行的情況"""
        def successful_function(x, y):
            return x + y
        
        result = safe_execute(successful_function, 3, 5)
        self.assertEqual(result, 8)
    
    def test_safe_execute_with_kwargs(self):
        """測試帶有關鍵字參數的成功執行"""
        def function_with_kwargs(a, b=10):
            return a * b
        
        result = safe_execute(function_with_kwargs, 3, b=7)
        self.assertEqual(result, 21)
    
    def test_safe_execute_with_exception(self):
        """測試處理異常"""
        def function_with_error():
            raise ValueError("測試錯誤")
        
        mock_logger = MagicMock()
        result = safe_execute(
            function_with_error, 
            logger=mock_logger,
            default_return="default"
        )
        self.assertEqual(result, "default")
        mock_logger.error.assert_called_once()
    
    def test_safe_execute_file_not_found(self):
        """測試處理 FileNotFoundError"""
        def function_with_file_error():
            raise FileNotFoundError("檔案未找到")
        
        mock_logger = MagicMock()
        result = safe_execute(
            function_with_file_error, 
            logger=mock_logger,
            default_return=None
        )
        self.assertIsNone(result)
        mock_logger.error.assert_called_once()
    
    def test_safe_execute_permission_error(self):
        """測試處理 PermissionError"""
        def function_with_permission_error():
            raise PermissionError("權限不足")
        
        mock_logger = MagicMock()
        result = safe_execute(
            function_with_permission_error, 
            logger=mock_logger,
            default_return=False
        )
        self.assertFalse(result)
        mock_logger.error.assert_called_once()
    
    def test_safe_execute_custom_error_prefix(self):
        """測試自定義錯誤訊息前綴"""
        def function_with_error():
            raise ValueError("測試錯誤")
        
        mock_logger = MagicMock()
        safe_execute(
            function_with_error,
            logger=mock_logger,
            error_msg_prefix="模組A處理時",
            default_return=None
        )
        
        # 檢查日誌訊息是否包含前綴
        call_args = mock_logger.error.call_args[0][0]
        self.assertIn("模組A處理時", call_args)
    
    def test_safe_execute_without_logger(self):
        """測試沒有 logger 的情況"""
        def function_with_error():
            raise ValueError("測試錯誤")
        
        # 應該返回預設值而不拋出異常
        with patch('builtins.print') as mock_print:
            result = safe_execute(
                function_with_error,
                default_return="fallback"
            )
            self.assertEqual(result, "fallback")
            mock_print.assert_called()
    
    def test_safe_execute_waifuc_base_error(self):
        """測試處理 WaifucBaseError"""
        def function_with_waifuc_error():
            raise WaifucBaseError("測試錯誤", {"test": "context"})
        
        mock_logger = MagicMock()
        result = safe_execute(
            function_with_waifuc_error, 
            logger=mock_logger,
            default_return="error_handled"
        )
        self.assertEqual(result, "error_handled")
        mock_logger.error.assert_called_once()


class TestHandleException(unittest.TestCase):
    """測試異常處理函數"""
    
    def test_handle_exception_with_logger(self):
        """測試有 logger 的異常處理"""
        mock_logger = MagicMock()
        
        try:
            raise ValueError("測試異常")
        except Exception:
            import sys
            exc_info = sys.exc_info()
            handle_exception(
                exc_info[0], exc_info[1], exc_info[2], 
                mock_logger, "TestContext"
            )
        
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args[0][0]
        self.assertIn("TestContext", call_args)
        self.assertIn("ValueError", call_args)
        self.assertIn("測試異常", call_args)
    
    def test_handle_exception_without_logger(self):
        """測試沒有 logger 的異常處理"""
        try:
            raise ValueError("測試異常")
        except Exception:
            import sys
            exc_info = sys.exc_info()
            
            with patch('builtins.print') as mock_print:
                with patch('traceback.print_exception') as mock_traceback:
                    handle_exception(
                        exc_info[0], exc_info[1], exc_info[2], 
                        None, "TestContext"
                    )
                    
                    mock_print.assert_called_once()
                    mock_traceback.assert_called_once()
                    
                    print_call = mock_print.call_args[0][0]
                    self.assertIn("TestContext", print_call)
                    self.assertIn("CRITICAL ERROR", print_call)


if __name__ == '__main__':
    unittest.main()
