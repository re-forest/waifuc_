"""
test_tag.py
tag 模組的單元測試

測試項目:
1. safe_process_single_image 函數與錯誤處理
2. process_tag 主要處理流程
3. 標記模型錯誤、圖像處理錯誤等異常處理
4. 檔案權限和存取錯誤處理
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

from tag import (
    safe_process_single_image,
    process_image,
    tag_image
)
from error_handler import DirectoryError, ImageProcessingError, ModelError, WaifucError
from tests.test_base import setup_test_environment, teardown_test_environment, create_test_image


class TestTagBase(unittest.TestCase):
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
            create_test_image(img_path)
            self.test_images.append(img_path)
        
        # 設定環境變數
        self.env_patcher = patch.dict(os.environ, {
            'custom_character_tag': 'test_character',
            'output_txt_filename': 'tags',
            'remove_low_threshold_tags': 'true',
            'low_threshold': '0.3',
            'CUDA_VISIBLE_DEVICES': '0'
        })
        self.env_patcher.start()
    
    def tearDown(self):
        """清理測試環境"""
        self.env_patcher.stop()
        teardown_test_environment()


class TestSafeProcessSingleImage(TestTagBase):
    """測試 safe_process_single_image 函數"""
    
    @patch('tag.get_wd14_tags')
    @patch('tag.tags_to_text')
    def test_successful_image_processing(self, mock_tags_to_text, mock_get_tags):
        """測試成功的圖像標記處理"""
        # 設定 mock 返回值
        mock_get_tags.return_value = (
            {'general': 0.9, 'sensitive': 0.1},  # rating
            {'character1': 0.8, 'hair_color': 0.7, 'low_tag': 0.2},  # features
            {'known_char': 0.9}  # characters
        )
        mock_tags_to_text.return_value = "character1, hair_color, known_char, test_character"
        
        test_image = self.test_images[0]
        
        # 執行測試
        result = safe_process_single_image(test_image)
        
        # 驗證結果
        self.assertTrue(result['success'])
        self.assertEqual(result['image_path'], test_image)
        self.assertIn('tag_text', result)
        self.assertIn('features_count', result)
        self.assertIn('characters_count', result)
        
        # 驗證函數調用
        mock_get_tags.assert_called_once_with(test_image, model_name="EVA02_Large")
        mock_tags_to_text.assert_called_once()
    
    def test_nonexistent_image(self):
        """測試不存在的圖像檔案"""
        nonexistent_path = os.path.join(self.input_dir, "nonexistent.jpg")
        
        with self.assertRaises(ImageProcessingError) as context:
            safe_process_single_image(nonexistent_path)
        
        self.assertIn("不存在", str(context.exception))
        self.assertEqual(context.exception.image_path, nonexistent_path)
    
    def test_unreadable_image(self):
        """測試無法讀取的圖像檔案"""
        test_image = self.test_images[0]
        
        with patch('os.access', return_value=False):
            with self.assertRaises(ImageProcessingError) as context:
                safe_process_single_image(test_image)
        
        self.assertIn("無法讀取", str(context.exception))
        self.assertEqual(context.exception.image_path, test_image)
    
    @patch('tag.get_wd14_tags')
    def test_model_error_handling(self, mock_get_tags):
        """測試模型錯誤處理"""
        mock_get_tags.side_effect = Exception("Model download failed")
        test_image = self.test_images[0]
        
        with self.assertRaises(ModelError) as context:
            safe_process_single_image(test_image)
        
        self.assertIn("標記模型處理失敗", str(context.exception))
        self.assertEqual(context.exception.model_name, "wd14_tagger")
    
    @patch('tag.get_wd14_tags')
    def test_onnx_model_error_handling(self, mock_get_tags):
        """測試 ONNX 模型錯誤處理"""
        mock_get_tags.side_effect = Exception("ONNX runtime error")
        test_image = self.test_images[0]
        
        with self.assertRaises(ModelError) as context:
            safe_process_single_image(test_image)
        
        self.assertIn("標記模型處理失敗", str(context.exception))
    
    @patch('tag.get_wd14_tags')
    def test_image_format_error_handling(self, mock_get_tags):
        """測試圖像格式錯誤處理"""
        mock_get_tags.side_effect = Exception("Invalid image format")
        test_image = self.test_images[0]
        
        with self.assertRaises(ImageProcessingError) as context:
            safe_process_single_image(test_image)
        
        self.assertIn("標記模型處理失敗", str(context.exception))
    
    @patch('tag.get_wd14_tags')
    @patch('tag.tags_to_text')
    def test_low_threshold_filtering(self, mock_tags_to_text, mock_get_tags):
        """測試低閾值標籤過濾"""
        # 設定 mock 返回值，包含低於閾值的標籤
        mock_get_tags.return_value = (
            {'general': 0.9},
            {'high_tag': 0.8, 'medium_tag': 0.4, 'low_tag': 0.2},
            {}
        )
        mock_tags_to_text.return_value = "high_tag, medium_tag"  # 低閾值標籤被過濾
        
        test_image = self.test_images[0]
        result = safe_process_single_image(test_image)
          # 驗證低閾值標籤被過濾
        self.assertTrue(result['success'])
        # 驗證 tags_to_text 被調用時的參數（不包含threshold，因為已預先過濾）
        call_args = mock_tags_to_text.call_args[0]  # 位置參數
        # 檢查傳入的features字典是否已經過濾了低閾值的標籤
        passed_features = call_args[0]        # 確認低閾值標籤不在過濾後的字典中
        self.assertNotIn('low_tag', passed_features)
    
    @patch('tag.get_wd14_tags')
    @patch('tag.tags_to_text')
    @patch.dict(os.environ, {'custom_character_tag': 'test_character'})
    def test_custom_character_tag_addition(self, mock_tags_to_text, mock_get_tags):
        """測試自定義角色標籤添加"""
        mock_get_tags.return_value = (
            {'general': 0.9},
            {'tag1': 0.8},
            {}  # 沒有檢測到角色
        )
        mock_tags_to_text.return_value = "tag1"
        
        test_image = self.test_images[0]
        result = safe_process_single_image(test_image)
        
        # 驗證結果包含自定義角色標籤
        self.assertTrue(result['success'])
        self.assertIn('test_character', result['tag_text'])


class TestTagImage(TestTagBase):
    """測試 tag_image 主函數"""
    
    @patch('tag.process_image')
    @patch('tag.ThreadPoolExecutor')
    def test_successful_tag_processing(self, mock_executor, mock_process_image):
        """測試成功的標記處理"""
        # 設定 mock
        mock_process_image.return_value = self.test_images[0]
        
        # 建立一個具有完整上下文管理器功能的mock
        mock_executor_instance = MagicMock()
        mock_executor_instance.map.return_value = self.test_images
        mock_executor_instance.__enter__ = MagicMock(return_value=mock_executor_instance)
        mock_executor_instance.__exit__ = MagicMock(return_value=None)
        mock_executor.return_value = mock_executor_instance
        
        # 執行測試
        result = tag_image(self.input_dir)
        
        # 驗證結果
        self.assertEqual(result, 3)  # 處理了3張圖片
    
    def test_invalid_input_directory(self):
        """測試無效的輸入目錄"""
        invalid_dir = "/non/existent/directory"
        
        with self.assertRaises(DirectoryError) as context:
            tag_image(invalid_dir)
        
        self.assertIn("不存在", str(context.exception))
    
    @patch('os.access')
    def test_unreadable_input_directory(self, mock_access):
        """測試無法讀取的輸入目錄"""
        mock_access.return_value = False
        
        with self.assertRaises(DirectoryError) as context:
            tag_image(self.input_dir)
        
        self.assertIn("無法讀取", str(context.exception))
    
    @patch('os.walk')
    def test_permission_error_listing_files(self, mock_walk):
        """測試列舉檔案權限錯誤"""
        mock_walk.side_effect = PermissionError("Permission denied")
        
        with self.assertRaises(DirectoryError) as context:
            tag_image(self.input_dir)
        
        self.assertIn("權限不足", str(context.exception))
    
    def test_empty_directory(self):
        """測試空目錄"""
        empty_dir = os.path.join(self.test_dir, "empty")
        os.makedirs(empty_dir, exist_ok=True)
        
        result = tag_image(empty_dir)
        
        self.assertEqual(result, 0)
    
    def test_no_image_files(self):
        """測試沒有圖像檔案的目錄"""
        # 創建非圖像檔案
        text_file = os.path.join(self.input_dir, "text.txt")
        with open(text_file, 'w') as f:
            f.write("test content")
          # 移除圖像檔案
        for img in self.test_images:
            os.remove(img)
        
        result = tag_image(self.input_dir)
        
        self.assertEqual(result, 0)
    
    @patch('tag.ThreadPoolExecutor')
    def test_partial_processing_failure(self, mock_executor):
        """測試部分處理失敗"""
        # 模擬處理過程
        mock_executor_instance = MagicMock()
        mock_executor_instance.map.return_value = self.test_images
        mock_executor_instance.__enter__ = MagicMock(return_value=mock_executor_instance)
        mock_executor_instance.__exit__ = MagicMock(return_value=None)
        mock_executor.return_value = mock_executor_instance
        
        result = tag_image(self.input_dir)
        
        self.assertEqual(result, 3)  # 返回處理的圖片數量


class TestTagIntegration(TestTagBase):
    """整合測試"""
    
    @patch('tag.get_wd14_tags')
    @patch('tag.tags_to_text')
    def test_tag_file_creation(self, mock_tags_to_text, mock_get_tags):
        """測試標籤檔案創建"""
        # 設定 mock
        mock_get_tags.return_value = (
            {'general': 0.9},
            {'tag1': 0.8, 'tag2': 0.7},
            {}
        )
        mock_tags_to_text.return_value = "tag1, tag2, test_character"
        
        test_image = self.test_images[0]
        
        with patch('builtins.open', mock_open()) as mock_file:
            result = safe_process_single_image(test_image)
            
            # 驗證標籤檔案被寫入
            self.assertTrue(result['success'])
            mock_file.assert_called()
    
    @patch('tag.get_wd14_tags')
    def test_unsupported_image_format(self, mock_get_tags):
        """測試不支援的圖像格式"""
        # 創建不支援的格式檔案
        unsupported_file = os.path.join(self.input_dir, "test.gif")
        with open(unsupported_file, 'w') as f:
            f.write("fake gif content")
        
        result = tag_image(self.input_dir)
        
        # 驗證只處理支援的格式
        self.assertEqual(result, 3)  # 只計算支援的格式
    
    @patch('tag.safe_process_single_image')
    @patch('tag.load_dotenv')  # Mock load_dotenv to avoid .env file dependency
    def test_large_dataset_simulation(self, mock_load_dotenv, mock_process_image):
        """測試大數據集處理模擬"""
        mock_process_image.return_value = {
            'success': True,
            'image_path': 'test.jpg',
            'tag_text': 'tag1',
            'features_count': 1,
            'characters_count': 0        }
        
        # 模擬大量檔案 - 使用 os.walk 而不是 os.listdir
        mock_walk_result = [(self.input_dir, [], [f"large_test_{i}.jpg" for i in range(100)])]
        with patch('os.walk', return_value=mock_walk_result):
            with patch('tag.ThreadPoolExecutor') as mock_executor:
                mock_future = Mock()
                mock_future.result.return_value = mock_process_image.return_value
                mock_executor_instance = MagicMock()
                mock_executor_instance.submit.return_value = mock_future
                mock_executor_instance.__enter__ = MagicMock(return_value=mock_executor_instance)
                mock_executor_instance.__exit__ = MagicMock(return_value=None)
                mock_executor.return_value = mock_executor_instance
                result = tag_image(self.input_dir)
                
                self.assertEqual(result, 100)


class TestTagErrorScenarios(TestTagBase):
    """錯誤場景測試"""
    
    @patch('tag.get_wd14_tags')
    def test_cuda_memory_error_simulation(self, mock_get_tags):
        """測試 CUDA 記憶體不足錯誤模擬"""
        mock_get_tags.side_effect = Exception("CUDA out of memory")
        test_image = self.test_images[0]
        
        with self.assertRaises(ModelError) as context:
            safe_process_single_image(test_image)
        
        self.assertIn("標記模型處理失敗", str(context.exception))
    
    def test_disk_full_error_simulation(self):
        """測試磁碟空間不足錯誤模擬"""
        test_image = self.test_images[0]
        
        with patch('builtins.open', side_effect=OSError("No space left on device")):
            with patch('tag.get_wd14_tags', return_value=({}, {}, {})):
                with patch('tag.tags_to_text', return_value="test"):
                    with self.assertRaises(WaifucError):
                        safe_process_single_image(test_image)
    
    @patch('tag.get_wd14_tags')
    def test_model_corruption_error(self, mock_get_tags):
        """測試模型檔案損壞錯誤"""
        mock_get_tags.side_effect = Exception("Model file corrupted")
        test_image = self.test_images[0]
        
        with self.assertRaises(ModelError) as context:
            safe_process_single_image(test_image)
        
        self.assertIn("標記模型處理失敗", str(context.exception))


if __name__ == '__main__':
    unittest.main()
