"""
test_upscale.py
upscale 模組的單元測試

測試項目:
1. upscale_to_target_size 圖像放大功能
2. safe_upscale_single_image 單圖處理
3. upscale_images_in_directory 批次處理
4. 權限錯誤、模型錯誤、圖像處理錯誤等異常處理
5. 整合測試與邊界條件測試
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch, Mock, MagicMock, call, mock_open
from typing import Optional, List, Dict, Any
from PIL import Image

# 確保測試時能正確導入相關模組
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from upscale import (
    upscale_to_target_size,
    upscale_and_center_crop,
    safe_upscale_single_image,
    upscale_images_in_directory
)
from error_handler import DirectoryError, ImageProcessingError, ModelError, WaifucError
from tests.test_base import setup_test_environment, teardown_test_environment, create_test_image


class TestUpscaleBase(unittest.TestCase):
    """測試基礎類別"""
    
    def setUp(self):
        """設定測試環境"""
        self.test_dir = setup_test_environment()
        self.input_dir = os.path.join(self.test_dir, "input")
        self.output_dir = os.path.join(self.test_dir, "output")
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 創建測試圖像
        self.test_images = []
        for i in range(3):
            img_path = os.path.join(self.input_dir, f"test_{i}.jpg")
            create_test_image(img_path, size=(100, 100))  # 小尺寸圖像用於測試放大
            self.test_images.append(img_path)
        
        # 設定環境變數
        self.env_patcher = patch.dict(os.environ, {
            'upscale_target_width': '512',
            'upscale_target_height': '512',
            'min_size_for_upscale': '200',
            'num_threads': '2'
        })
        self.env_patcher.start()
    
    def tearDown(self):
        """清理測試環境"""
        self.env_patcher.stop()
        teardown_test_environment()


class TestUpscaleToTargetSize(TestUpscaleBase):
    """測試 upscale_to_target_size 函數"""
    
    @patch('upscale.upscale_with_cdc')
    def test_successful_upscale(self, mock_upscale):
        """測試成功的圖像放大"""
        # 創建模擬的放大後圖像
        mock_upscaled_image = Mock()
        mock_upscaled_image.size = (512, 512)
        mock_upscaled_image.resize.return_value = mock_upscaled_image
        mock_upscale.return_value = mock_upscaled_image
        
        # 創建測試圖像
        test_img = Image.new('RGB', (100, 100), color='red')
        
        # 執行測試
        result = upscale_to_target_size(test_img, 512, 512)
          # 驗證結果
        self.assertEqual(result.size, (512, 512))
        mock_upscale.assert_called_once()
    
    @patch('upscale.upscale_with_cdc')
    def test_upscale_with_model_error(self, mock_upscale):
        """測試放大模型錯誤"""
        mock_upscale.side_effect = Exception("Model failed")
        test_img = Image.new('RGB', (100, 100), color='red')
        
        with self.assertRaises(Exception):
            upscale_to_target_size(test_img, 512, 512)
    
    def test_invalid_image_input(self):
        """測試無效圖像輸入"""
        with self.assertRaises((TypeError, ImageProcessingError)):
            upscale_to_target_size(None, 512, 512)


class TestSafeUpscaleSingleImage(TestUpscaleBase):
    """測試 safe_upscale_single_image 函數"""
    
    @patch('upscale.upscale_and_center_crop')
    @patch('PIL.Image.open')
    def test_successful_single_image_upscale(self, mock_image_open, mock_upscale):
        """測試成功的單圖放大"""
        # 設定 mock
        mock_img = Mock()
        mock_img.size = (100, 100)
        mock_img.mode = 'RGB'
        mock_img.convert.return_value = mock_img
        mock_img.close = Mock()  # 添加 close 方法
        mock_image_open.return_value = mock_img
        
        mock_upscaled = Mock()
        mock_upscaled.size = (512, 512)
        mock_upscaled.save = Mock()  # 添加 save 方法
        mock_upscale.return_value = mock_upscaled
        
        test_image = self.test_images[0]
        
        # 執行測試
        result = safe_upscale_single_image(test_image, 512, 512, 'test_model', False, 200)
        
        # 驗證結果
        self.assertIsInstance(result, dict)
        self.assertTrue(result.get('success'))
        mock_upscale.assert_called_once()
    
    def test_skip_large_image(self):
        """測試跳過大圖像"""
        with patch('PIL.Image.open') as mock_open:
            mock_img = Mock()
            mock_img.size = (1000, 1000)  # 大於最小處理尺寸
            mock_img.mode = 'RGB'
            mock_img.convert.return_value = mock_img
            mock_img.close = Mock()
            mock_open.return_value = mock_img
            
            test_image = self.test_images[0]
            result = safe_upscale_single_image(test_image, 512, 512, 'test_model', False, 200)
            
            # 應該跳過處理
            self.assertIsInstance(result, dict)
            self.assertTrue(result.get('skipped'))
    
    def test_file_not_found_error(self):
        """測試檔案不存在錯誤"""
        nonexistent_file = "/path/to/nonexistent.jpg"
        
        with self.assertRaises(ImageProcessingError):
            safe_upscale_single_image(nonexistent_file, 512, 512, 'test_model', False, 200)
    
    @patch('PIL.Image.open')
    def test_permission_error_handling(self, mock_open):
        """測試權限錯誤處理"""
        mock_open.side_effect = PermissionError("Permission denied")
        
        with self.assertRaises(ImageProcessingError):
            safe_upscale_single_image(self.test_images[0], 512, 512, 'test_model', False, 200)


class TestUpscaleImagesInDirectory(TestUpscaleBase):
    """測試 upscale_images_in_directory 主函數"""
    
    @patch('upscale.safe_execute')
    @patch('concurrent.futures.ThreadPoolExecutor')
    def test_successful_upscale_processing(self, mock_executor, mock_safe_execute):
        """測試成功的圖像放大處理"""
        # 設定 mock - safe_execute 應該返回 dict 格式
        mock_safe_execute.return_value = {
            "status": "success",
            "upscaled": True,
            "skipped": False,
            "original_size": (100, 100),
            "final_size": (1024, 1024),
            "processing_time": 1.0
        }
        mock_executor_instance = MagicMock()
        mock_executor_instance.map.return_value = self.test_images
        mock_executor.return_value = mock_executor_instance
          # 執行測試
        result = upscale_images_in_directory(self.input_dir)
        
        # 驗證結果 - 應該返回 (processed_count, upscaled_count)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
    
    def test_invalid_input_directory(self):
        """測試無效的輸入目錄"""
        invalid_dir = "/non/existent/directory"
        
        with self.assertRaises(DirectoryError) as context:
            upscale_images_in_directory(invalid_dir)
        
        self.assertIn("不存在", str(context.exception))
    
    @patch('os.access')
    def test_unreadable_input_directory(self, mock_access):
        """測試無法讀取的輸入目錄"""
        mock_access.return_value = False
        
        with self.assertRaises(DirectoryError) as context:
            upscale_images_in_directory(self.input_dir)
        
        self.assertIn("無法讀取", str(context.exception))
    
    def test_empty_directory(self):
        """測試空目錄"""
        empty_dir = os.path.join(self.test_dir, "empty")
        os.makedirs(empty_dir, exist_ok=True)
        
        result = upscale_images_in_directory(empty_dir)
        
        self.assertEqual(result, (0, 0))
    
    def test_no_image_files(self):
        """測試沒有圖像檔案的目錄"""
        # 創建非圖像檔案
        text_file = os.path.join(self.input_dir, "text.txt")
        with open(text_file, 'w') as f:
            f.write("test content")
        
        # 移除圖像檔案
        for img in self.test_images:
            os.remove(img)
        
        result = upscale_images_in_directory(self.input_dir)
        
        self.assertEqual(result, (0, 0))


class TestUpscaleEnvironmentVariables(TestUpscaleBase):
    """測試環境變數處理"""
    
    @patch.dict(os.environ, {'upscale_target_width': 'invalid'})
    @patch('upscale.safe_execute')
    def test_invalid_target_width(self, mock_safe_execute):
        """測試無效的目標寬度"""
        # 模擬 safe_upscale_single_image 的正確返回值
        mock_safe_execute.return_value = {
            "success": True,
            "original_size": (100, 100),
            "new_size": (512, 512),
            "elapsed_time": 1.0,
            "save_path": self.test_images[0]
        }
        
        # 應該使用預設值
        result = upscale_images_in_directory(self.input_dir)
        self.assertEqual(result, (3, 3))
    
    @patch.dict(os.environ, {'min_size_for_upscale': 'invalid'})
    @patch('upscale.safe_execute')
    def test_invalid_min_size(self, mock_safe_execute):
        """測試無效的最小尺寸"""
        # 模擬 safe_upscale_single_image 的正確返回值
        mock_safe_execute.return_value = {
            "success": True,
            "original_size": (100, 100),
            "new_size": (512, 512),
            "elapsed_time": 1.0,
            "save_path": self.test_images[0]
        }
        
        # 應該使用預設值
        result = upscale_images_in_directory(self.input_dir)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)


class TestUpscaleErrorScenarios(TestUpscaleBase):
    """錯誤場景測試"""
    
    @patch('upscale.safe_execute')
    def test_cuda_memory_error_simulation(self, mock_safe_execute):
        """測試 CUDA 記憶體不足錯誤模擬"""
        # safe_execute 會返回 default_return (None) 而不是拋出異常
        mock_safe_execute.return_value = None
        
        result = upscale_images_in_directory(self.input_dir)        # 應該返回 (3, 0) - 找到3個文件，但都因錯誤而未能處理成功
        self.assertEqual(result, (3, 0))
    
    @patch('upscale.safe_execute')
    def test_model_error_handling(self, mock_safe_execute):
        """測試模型錯誤處理"""
        # safe_execute 會捕獲異常並返回 default_return (None)
        mock_safe_execute.return_value = None
        
        result = upscale_images_in_directory(self.input_dir)
        # 應該返回 (3, 0) - 找到3個文件，但都因錯誤而未能處理成功  
        self.assertEqual(result, (3, 0))
    
    @patch('os.walk')
    def test_permission_error_listing_files(self, mock_walk):
        """測試列舉檔案權限錯誤"""
        mock_walk.side_effect = PermissionError("Permission denied")
        
        with self.assertRaises(DirectoryError):
            upscale_images_in_directory(self.input_dir)


class TestUpscaleIntegration(TestUpscaleBase):
    """整合測試"""
    
    @patch('upscale.safe_execute')
    def test_large_dataset_simulation(self, mock_safe_execute):
        """測試大數據集處理模擬"""
        mock_safe_execute.return_value = {
            "status": "success",
            "upscaled": True,
            "skipped": False,
            "original_size": (100, 100),
            "final_size": (1024, 1024),
            "processing_time": 1.0
        }
        
        # 模擬大量檔案
        with patch('os.walk', return_value=[
            (self.input_dir, [], [f"test_{i}.jpg" for i in range(100)])
        ]):
            with patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
                mock_executor_instance = MagicMock()
                mock_executor_instance.map.return_value = ["processed"] * 100
                mock_executor.return_value = mock_executor_instance
                
                result = upscale_images_in_directory(self.input_dir)
                
                # 檢查返回值是元組格式 (processed_count, upscaled_count)
                self.assertIsInstance(result, tuple)
                self.assertEqual(len(result), 2)


if __name__ == '__main__':
    unittest.main()
