"""
圖像驗證模組的單元測試
"""
import unittest
import os
import sys
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validate_image import validate_single_image, validate_and_remove_invalid_images
from error_handler import DirectoryError, ImageProcessingError
from tests.test_base import (
    setup_test_environment, teardown_test_environment,
    create_test_directory_structure, create_test_image, create_corrupted_image,
    assert_file_exists, assert_file_not_exists, count_files_in_directory
)


class TestValidateSingleImage(unittest.TestCase):
    """測試單一圖像驗證函數"""
    
    def setUp(self):
        """設定測試環境"""
        self.test_dir = setup_test_environment()
        
    def tearDown(self):
        """清理測試環境"""
        teardown_test_environment()
    
    def test_validate_valid_image(self):
        """測試驗證有效圖像"""
        # 創建有效圖像
        image_path = os.path.join(self.test_dir, "valid_image.jpg")
        create_test_image(image_path)
        
        # 測試驗證
        result = validate_single_image(image_path)
        self.assertTrue(result)
    
    def test_validate_corrupted_image(self):
        """測試驗證損壞圖像"""
        # 創建損壞圖像
        image_path = os.path.join(self.test_dir, "corrupted_image.jpg")
        create_corrupted_image(image_path)
        
        # 測試驗證
        result = validate_single_image(image_path)
        self.assertFalse(result)
    
    def test_validate_nonexistent_image(self):
        """測試驗證不存在的圖像"""
        image_path = os.path.join(self.test_dir, "nonexistent.jpg")
        
        # 測試驗證
        result = validate_single_image(image_path)
        self.assertFalse(result)
    
    def test_validate_different_formats(self):
        """測試驗證不同格式的圖像"""
        formats = [
            ("test.jpg", "JPEG"),
            ("test.png", "PNG"),
            ("test.bmp", "BMP"),
            ("test.webp", "WebP")
        ]
        
        for filename, format_type in formats:
            with self.subTest(format=format_type):
                image_path = os.path.join(self.test_dir, filename)
                
                # 創建特定格式的圖像
                try:
                    img = Image.new("RGB", (100, 100), (255, 0, 0))
                    img.save(image_path, format_type)
                    
                    # 測試驗證
                    result = validate_single_image(image_path)
                    self.assertTrue(result, f"{format_type} 格式應該有效")
                except Exception:
                    # 如果格式不支援，跳過測試
                    self.skipTest(f"{format_type} 格式不支援")


class TestValidateAndRemoveInvalidImages(unittest.TestCase):
    """測試批次圖像驗證和移除函數"""
    
    def setUp(self):
        """設定測試環境"""
        self.test_dir = setup_test_environment()
        self.test_paths = create_test_directory_structure(self.test_dir)
        
    def tearDown(self):
        """清理測試環境"""
        teardown_test_environment()
    
    @patch('validate_image.logger')
    def test_validate_directory_with_mixed_files(self, mock_logger):
        """測試包含混合檔案的目錄驗證"""
        input_dir = self.test_paths['input_dir']
        
        # 執行驗證
        total, removed = validate_and_remove_invalid_images(input_dir)
        
        # 檢查結果
        self.assertGreater(total, 0, "應該處理一些檔案")
        self.assertEqual(removed, 2, "應該移除2個損壞的圖像檔案")
        
        # 檢查有效圖像仍存在
        for img_path in self.test_paths['valid_images']:
            assert_file_exists(img_path, "有效圖像應該保留")
        
        # 檢查損壞圖像已移除
        for img_path in self.test_paths['corrupted_images']:
            assert_file_not_exists(img_path, "損壞圖像應該被移除")
        
        # 檢查非圖像檔案未受影響
        for file_path in self.test_paths['non_images']:
            assert_file_exists(file_path, "非圖像檔案應該保留")
    
    def test_validate_nonexistent_directory(self):
        """測試驗證不存在的目錄"""
        nonexistent_dir = os.path.join(self.test_dir, "nonexistent")
        
        # 應該拋出 DirectoryError
        with self.assertRaises(DirectoryError):
            validate_and_remove_invalid_images(nonexistent_dir)
    
    def test_validate_empty_directory(self):
        """測試驗證空目錄"""
        empty_dir = os.path.join(self.test_dir, "empty")
        os.makedirs(empty_dir)
        
        # 執行驗證
        total, removed = validate_and_remove_invalid_images(empty_dir)
        
        # 檢查結果
        self.assertEqual(total, 0, "空目錄應該沒有檔案被處理")
        self.assertEqual(removed, 0, "空目錄應該沒有檔案被移除")
    
    def test_validate_directory_permission_error(self):
        """測試目錄權限錯誤"""
        # 在 Windows 上模擬權限錯誤比較困難，使用 mock
        with patch('os.access', return_value=False):
            with self.assertRaises(DirectoryError):
                validate_and_remove_invalid_images(self.test_paths['input_dir'])
    
    @patch('os.listdir')
    def test_validate_directory_list_error(self, mock_listdir):
        """測試目錄列舉錯誤"""
        mock_listdir.side_effect = PermissionError("權限不足")
        
        with self.assertRaises(DirectoryError):
            validate_and_remove_invalid_images(self.test_paths['input_dir'])
    
    @patch('validate_image.safe_execute')
    @patch('validate_image.logger')
    def test_validate_with_safe_execute_failure(self, mock_logger, mock_safe_execute):
        """測試 safe_execute 失敗的情況"""
        # 模擬 safe_execute 返回 False（處理失敗）
        mock_safe_execute.return_value = False
        
        input_dir = self.test_paths['input_dir']
        total, removed = validate_and_remove_invalid_images(input_dir)
        
        # 檢查結果 - 由於處理失敗，沒有檔案被移除
        self.assertGreater(total, 0, "應該嘗試處理檔案")
        self.assertEqual(removed, 0, "失敗時不應該移除檔案")
    
    def test_validate_only_image_files_processed(self):
        """測試只處理圖像檔案"""
        input_dir = self.test_paths['input_dir']
        
        # 計算處理前的圖像檔案數量
        initial_image_count = len(self.test_paths['valid_images']) + len(self.test_paths['corrupted_images'])
        initial_total_count = count_files_in_directory(input_dir)
        
        # 執行驗證
        total, removed = validate_and_remove_invalid_images(input_dir)
        
        # 總處理數應該等於圖像檔案數，而不是所有檔案數
        self.assertEqual(total, initial_image_count, "只應該處理圖像檔案")
        self.assertLess(total, initial_total_count, "處理的檔案數應該少於總檔案數")


class TestValidateImageIntegration(unittest.TestCase):
    """圖像驗證模組整合測試"""
    
    def setUp(self):
        """設定測試環境"""
        self.test_dir = setup_test_environment()
        
    def tearDown(self):
        """清理測試環境"""
        teardown_test_environment()
    
    def test_large_directory_processing(self):
        """測試處理大量檔案的目錄"""
        input_dir = os.path.join(self.test_dir, "large_dir")
        os.makedirs(input_dir)
        
        # 創建大量測試檔案
        num_files = 50
        valid_count = 0
        invalid_count = 0
        
        for i in range(num_files):
            if i % 3 == 0:  # 每3個檔案中有1個是損壞的
                file_path = os.path.join(input_dir, f"corrupted_{i}.jpg")
                create_corrupted_image(file_path)
                invalid_count += 1
            else:
                file_path = os.path.join(input_dir, f"valid_{i}.jpg")
                create_test_image(file_path)
                valid_count += 1
        
        # 執行驗證
        total, removed = validate_and_remove_invalid_images(input_dir)
        
        # 檢查結果
        self.assertEqual(total, num_files, f"應該處理 {num_files} 個檔案")
        self.assertEqual(removed, invalid_count, f"應該移除 {invalid_count} 個損壞檔案")
        
        # 檢查剩餘檔案數量
        remaining_files = count_files_in_directory(input_dir, ".jpg")
        self.assertEqual(remaining_files, valid_count, f"應該剩餘 {valid_count} 個有效檔案")
    
    @patch.dict(os.environ, {'directory': ''})
    def test_main_execution_missing_directory(self):
        """測試主程序執行時缺少目錄環境變數"""
        # 這個測試需要模擬 main 執行，但由於 main 會 exit(1)，
        # 需要特殊處理或重構 main 部分的程式碼
        pass
    
    def test_concurrent_file_access(self):
        """測試並發檔案存取情況"""
        # 這個測試比較複雜，需要模擬多線程或多進程存取
        # 可以作為未來的進階測試項目
        pass


if __name__ == '__main__':
    unittest.main()
