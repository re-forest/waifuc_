"""
test_crop.py
crop 模組的單元測試

測試項目:
1. safe_waifuc_process 函數與錯誤處理
2. safe_move_file 函數與檔案移動
3. process_single_folder 主要處理流程
4. classify_files_in_directory 分類功能
5. 權限錯誤、模型錯誤、圖像處理錯誤等異常處理
6. 整合測試與邊界條件測試
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch, Mock, MagicMock, call
from typing import Optional, List, Dict, Any
from PIL import Image

# 確保測試時能正確導入相關模組
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crop import (
    safe_waifuc_process,
    safe_move_file,
    process_single_folder,
    classify_files_in_directory
)
from error_handler import DirectoryError, ImageProcessingError, ModelError, WaifucError
from tests.test_base import setup_test_environment, teardown_test_environment, create_test_image


class TestCropBase(unittest.TestCase):
    """測試基礎類別"""
    
    def setUp(self):
        """設定測試環境"""
        self.test_dir = setup_test_environment()
        self.input_dir = os.path.join(self.test_dir, "input")
        self.output_dir = os.path.join(self.test_dir, "output")
        os.makedirs(self.input_dir, exist_ok=True)
        
        # 創建測試圖像
        self.test_images = []
        for i in range(3):
            img_path = os.path.join(self.input_dir, f"test_{i}.jpg")
            create_test_image(img_path)
            self.test_images.append(img_path)
    
    def tearDown(self):
        """清理測試環境"""
        teardown_test_environment()



class TestSafeWaifucProcess(TestCropBase):
    """測試 safe_waifuc_process 函數"""
    
    @patch('crop.LocalSource')
    @patch('crop.ThreeStageSplitAction')
    @patch('crop.SaveExporter')
    def test_successful_process(self, mock_save_exporter, mock_action, mock_local_source):
        """測試成功的 waifuc 處理"""
        # 設定 mock
        mock_source = Mock()
        mock_local_source.return_value = mock_source
        mock_source.attach.return_value = mock_source
        
        # 執行測試
        safe_waifuc_process(self.input_dir, self.output_dir)
        
        # 驗證調用
        mock_local_source.assert_called_once_with(self.input_dir)
        mock_source.attach.assert_called_once()
        mock_source.export.assert_called_once()
        
        # 驗證輸出目錄已創建
        self.assertTrue(os.path.exists(self.output_dir))
    
    @patch('crop.LocalSource')
    def test_permission_error_handling(self, mock_local_source):
        """測試權限錯誤處理"""
        mock_local_source.side_effect = PermissionError("Permission denied")
        
        with self.assertRaises(DirectoryError) as context:
            safe_waifuc_process(self.input_dir, self.output_dir)
        
        self.assertIn("權限不足", str(context.exception))
        self.assertIn(self.output_dir, str(context.exception))
    
    @patch('crop.LocalSource')
    def test_file_not_found_error_handling(self, mock_local_source):
        """測試檔案不存在錯誤處理"""
        mock_local_source.side_effect = FileNotFoundError("Directory not found")
        
        with self.assertRaises(DirectoryError) as context:
            safe_waifuc_process("/non/existent/path", self.output_dir)
        
        self.assertIn("不存在", str(context.exception))
    
    @patch('crop.LocalSource')
    def test_model_error_handling(self, mock_local_source):
        """測試模型錯誤處理"""
        mock_local_source.side_effect = Exception("Model download failed")
        
        with self.assertRaises(ModelError) as context:
            safe_waifuc_process(self.input_dir, self.output_dir)
        
        self.assertIn("waifuc 處理失敗", str(context.exception))
    
    @patch('crop.LocalSource')
    def test_image_processing_error_handling(self, mock_local_source):
        """測試圖像處理錯誤處理"""
        mock_local_source.side_effect = Exception("Invalid image format")
        
        with self.assertRaises(ImageProcessingError) as context:
            safe_waifuc_process(self.input_dir, self.output_dir)
        
        self.assertIn("waifuc 處理失敗", str(context.exception))


class TestSafeMoveFile(TestCropBase):
    """測試 safe_move_file 函數"""
    
    def test_successful_file_move(self):
        """測試成功的檔案移動"""
        source_file = os.path.join(self.input_dir, "test_move.jpg")
        target_file = os.path.join(self.output_dir, "moved_test.jpg")
        
        # 創建源檔案
        create_test_image(source_file)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 執行移動
        result = safe_move_file(source_file, target_file)
          # 驗證結果
        self.assertTrue(result)
        self.assertFalse(os.path.exists(source_file))
        self.assertTrue(os.path.exists(target_file))
    
    def test_move_nonexistent_file(self):
        """測試移動不存在的檔案"""
        source_file = os.path.join(self.input_dir, "nonexistent.jpg")
        target_file = os.path.join(self.output_dir, "target.jpg")
        
        # safe_move_file 應該返回 False 而不是拋出異常
        result = safe_move_file(source_file, target_file)
        self.assertFalse(result)
    
    def test_move_to_existing_file(self):
        """測試移動到已存在的檔案"""
        source_file = os.path.join(self.input_dir, "source.jpg")
        target_file = os.path.join(self.output_dir, "target.jpg")
          # 創建源檔案和目標檔案
        create_test_image(source_file)
        os.makedirs(self.output_dir, exist_ok=True)
        create_test_image(target_file)
          # safe_move_file 在目標檔案已存在時應該成功移動（覆蓋）
        result = safe_move_file(source_file, target_file)
        self.assertTrue(result)  # 移動應該成功
    
    @patch('shutil.move')
    def test_permission_error_handling(self, mock_move):
        """測試權限錯誤處理"""
        mock_move.side_effect = PermissionError("Permission denied")
        
        source_file = os.path.join(self.input_dir, "test.jpg")
        target_file = os.path.join(self.output_dir, "moved.jpg")
        create_test_image(source_file)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # safe_move_file 應該返回 False 而不是拋出異常
        result = safe_move_file(source_file, target_file)
        self.assertFalse(result)


class TestClassifyFilesInDirectory(TestCropBase):
    """測試 classify_files_in_directory 函數"""
    
    def setUp(self):
        """設定測試環境"""
        super().setUp()
        
        # 創建測試檔案 (模擬 waifuc 輸出)
        self.test_files = [
            "image_001_person1_head.jpg",
            "image_002_person1_halfbody.jpg", 
            "image_003_person1.jpg",
            "image_004_person1_head.png",
            "other_file.jpg"
        ]
        
        for filename in self.test_files:
            create_test_image(os.path.join(self.output_dir, filename))
    
    def test_successful_classification(self):
        """測試成功的檔案分類"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 執行分類
        classify_files_in_directory(self.output_dir)
        
        # 驗證分類結果
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "head")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "halfbody")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "person")))
        
        # 驗證檔案移動
        head_files = os.listdir(os.path.join(self.output_dir, "head"))
        halfbody_files = os.listdir(os.path.join(self.output_dir, "halfbody"))
        person_files = os.listdir(os.path.join(self.output_dir, "person"))
        
        self.assertEqual(len([f for f in head_files if "_person1_head" in f]), 2)
        self.assertEqual(len([f for f in halfbody_files if "_person1_halfbody" in f]), 1)
        self.assertEqual(len([f for f in person_files if "_person1.jpg" in f]), 1)
    
    def test_nonexistent_directory(self):
        """測試不存在的目錄"""
        nonexistent_dir = os.path.join(self.test_dir, "nonexistent")
        
        with self.assertRaises(DirectoryError) as context:
            classify_files_in_directory(nonexistent_dir)
        
        self.assertIn("不存在", str(context.exception))
    
    @patch('os.listdir')
    def test_permission_error_handling(self, mock_listdir):
        """測試權限錯誤處理"""
        mock_listdir.side_effect = PermissionError("Permission denied")
        
        with self.assertRaises(DirectoryError) as context:            classify_files_in_directory(self.output_dir)
        
        self.assertIn("權限不足", str(context.exception))
    
    @patch('os.makedirs')
    def test_directory_creation_error(self, mock_makedirs):
        """測試目錄創建錯誤"""
        # 創建測試圖像檔案
        test_image = os.path.join(self.input_dir, "test_person1_head.jpg")
        create_test_image(test_image)
        
        # 設定 mock 在第二次調用時拋出異常（第一次調用可能是創建測試目錄）
        mock_makedirs.side_effect = [None, PermissionError("Permission denied")]
        
        with self.assertRaises(DirectoryError) as context:
            classify_files_in_directory(self.input_dir)
        
        self.assertIn("權限不足", str(context.exception))


class TestProcessSingleFolder(TestCropBase):
    """測試 process_single_folder 函數"""
    
    @patch('crop.safe_execute')
    def test_successful_process(self, mock_safe_execute):
        """測試成功的單一資料夾處理"""
        mock_safe_execute.return_value = True
        
        # 執行測試
        process_single_folder(self.input_dir, self.output_dir)
        
        # 驗證 safe_execute 被調用了兩次 (waifuc處理和分類)
        self.assertEqual(mock_safe_execute.call_count, 2)
        
        # 驗證第一次調用是 waifuc 處理
        first_call = mock_safe_execute.call_args_list[0]
        self.assertEqual(first_call[0][0], safe_waifuc_process)
        
        # 驗證第二次調用是分類
        second_call = mock_safe_execute.call_args_list[1]
        self.assertEqual(second_call[0][0], classify_files_in_directory)


class TestCropIntegration(TestCropBase):
    """整合測試"""
    
    def test_empty_directory_classification(self):
        """測試空目錄的分類"""
        empty_dir = os.path.join(self.test_dir, "empty")
        os.makedirs(empty_dir, exist_ok=True)
        
        # 應該不會拋出異常
        classify_files_in_directory(empty_dir)
        
        # 驗證子目錄被創建
        self.assertTrue(os.path.exists(os.path.join(empty_dir, "head")))
        self.assertTrue(os.path.exists(os.path.join(empty_dir, "halfbody")))
        self.assertTrue(os.path.exists(os.path.join(empty_dir, "person")))
    
    @patch('crop.safe_waifuc_process')
    @patch('crop.classify_files_in_directory')
    def test_end_to_end_process(self, mock_classify, mock_waifuc):
        """測試端對端處理流程"""
        with patch('crop.safe_execute', side_effect=lambda func, *args, **kwargs: func(*args)):
            process_single_folder(self.input_dir, self.output_dir)
            
            mock_waifuc.assert_called_once_with(self.input_dir, self.output_dir)
            mock_classify.assert_called_once_with(self.output_dir)


class TestCropErrorScenarios(TestCropBase):
    """錯誤場景測試"""
    
    def test_readonly_directory_error(self):
        """測試只讀目錄錯誤"""
        readonly_dir = os.path.join(self.test_dir, "readonly")
        os.makedirs(readonly_dir, exist_ok=True)
          # 在 Windows 上模擬只讀權限
        if os.name == 'nt':
            import stat
            os.chmod(readonly_dir, stat.S_IREAD)
        
        try:
            with patch('os.access', return_value=False):
                with self.assertRaises(DirectoryError):
                    classify_files_in_directory(readonly_dir)
        finally:
            # 恢復權限以便清理
            if os.name == 'nt':
                os.chmod(readonly_dir, stat.S_IWRITE | stat.S_IREAD)
    
    def test_disk_space_error_simulation(self):
        """測試磁碟空間不足錯誤模擬"""
        source_file = os.path.join(self.input_dir, "test.jpg")
        target_file = os.path.join(self.output_dir, "test.jpg")
        create_test_image(source_file)
        os.makedirs(self.output_dir, exist_ok=True)
        
        with patch('shutil.move', side_effect=OSError("No space left on device")):
            result = safe_move_file(source_file, target_file)
            self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()
