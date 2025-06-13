"""
test_lpips_clustering.py
lpips_clustering 模組的單元測試

測試項目:
1. batch_generator 函數
2. safe_lpips_clustering 函數與錯誤處理
3. ensure_group_directory 函數與目錄創建
4. move_file_to_group 函數與檔案移動
5. process_lpips_clustering 主要處理流程
6. 權限錯誤、模型錯誤、圖像處理錯誤等異常處理
7. 整合測試與邊界條件測試
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch, Mock, MagicMock
from typing import Optional, List, Dict, Any
from PIL import Image

# 確保測試時能正確導入相關模組
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lpips_clustering import (
    batch_generator,
    safe_lpips_clustering,
    ensure_group_directory,
    move_file_to_group,
    process_lpips_clustering
)
from error_handler import DirectoryError, ImageProcessingError, ModelError, WaifucError


class TestLpipsClusteringBase(unittest.TestCase):
    """測試基礎類別"""
    
    def create_test_image_file(self, file_path: str, content: str = "test"):
        """創建測試圖像檔案"""
        try:
            # 創建一個簡單的 RGB 圖像
            img = Image.new('RGB', (100, 100), color='red')
            img.save(file_path)
        except Exception:
            # 如果 PIL 失敗，創建文字檔案
            with open(file_path, 'w') as f:
                f.write(content)


class TestBatchGenerator(TestLpipsClusteringBase):
    """測試 batch_generator 函數"""
    
    def test_batch_generator_normal(self):
        """測試正常的批次生成"""
        data = list(range(10))
        batches = list(batch_generator(data, 3))
        
        expected = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [9]
        ]
        self.assertEqual(batches, expected)
    
    def test_batch_generator_exact_division(self):
        """測試恰好整除的批次生成"""
        data = list(range(6))
        batches = list(batch_generator(data, 2))
        
        expected = [
            [0, 1],
            [2, 3],
            [4, 5]
        ]
        self.assertEqual(batches, expected)
    
    def test_batch_generator_empty_list(self):
        """測試空列表的批次生成"""
        batches = list(batch_generator([], 3))
        self.assertEqual(batches, [])
    
    def test_batch_generator_single_item(self):
        """測試單一項目的批次生成"""
        data = ['item1']
        batches = list(batch_generator(data, 3))
        self.assertEqual(batches, [['item1']])
    
    def test_batch_generator_large_batch_size(self):
        """測試批次大小大於資料量的情況"""
        data = [1, 2, 3]
        batches = list(batch_generator(data, 10))
        self.assertEqual(batches, [[1, 2, 3]])


class TestSafeLpipseClustering(TestLpipsClusteringBase):
    """測試 safe_lpips_clustering 函數"""
    
    @patch('lpips_clustering.lpips_clustering')
    def test_safe_lpips_clustering_success(self, mock_lpips):
        """測試成功的 LPIPS 聚類"""
        mock_lpips.return_value = [0, 1, 0, 2]
        file_paths = ['file1.jpg', 'file2.jpg', 'file3.jpg', 'file4.jpg']
        
        result = safe_lpips_clustering(file_paths)
        
        self.assertEqual(result, [0, 1, 0, 2])
        mock_lpips.assert_called_once_with(file_paths)
    
    @patch('lpips_clustering.lpips_clustering')
    def test_safe_lpips_clustering_model_error(self, mock_lpips):
        """測試 LPIPS 模型錯誤"""
        mock_lpips.side_effect = Exception("CUDA out of memory")
        file_paths = ['file1.jpg']
        
        with self.assertRaises(ModelError) as context:
            safe_lpips_clustering(file_paths)
        
        self.assertIn("LPIPS 聚類失敗", str(context.exception))
        self.assertIn("CUDA", str(context.exception))
    
    @patch('lpips_clustering.lpips_clustering')
    def test_safe_lpips_clustering_image_processing_error(self, mock_lpips):
        """測試圖像處理錯誤"""
        mock_lpips.side_effect = Exception("Invalid image format")
        file_paths = ['file1.jpg']
        
        with self.assertRaises(ImageProcessingError) as context:
            safe_lpips_clustering(file_paths)
        
        self.assertIn("LPIPS 聚類失敗", str(context.exception))
    
    @patch('lpips_clustering.lpips_clustering')
    def test_safe_lpips_clustering_empty_input(self, mock_lpips):
        """測試空輸入的 LPIPS 聚類"""
        mock_lpips.return_value = []
        
        result = safe_lpips_clustering([])
        
        self.assertEqual(result, [])
        mock_lpips.assert_called_once_with([])


class TestEnsureGroupDirectory(TestLpipsClusteringBase):
    """測試 ensure_group_directory 函數"""
    
    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        super().tearDown()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_ensure_group_directory_success(self):
        """測試成功創建群組目錄"""
        group_dir = ensure_group_directory(self.temp_dir, 1)
        expected_path = os.path.join(self.temp_dir, "group_1")
        
        self.assertEqual(group_dir, expected_path)
        self.assertTrue(os.path.exists(expected_path))
        self.assertTrue(os.path.isdir(expected_path))
    
    def test_ensure_group_directory_existing(self):
        """測試目錄已存在的情況"""
        # 先創建目錄
        existing_dir = os.path.join(self.temp_dir, "group_2")
        os.makedirs(existing_dir)
        
        group_dir = ensure_group_directory(self.temp_dir, 2)
        
        self.assertEqual(group_dir, existing_dir)
        self.assertTrue(os.path.exists(existing_dir))
    
    @patch('os.makedirs')
    def test_ensure_group_directory_permission_error(self, mock_makedirs):
        """測試權限錯誤"""
        mock_makedirs.side_effect = PermissionError("Permission denied")
        
        with self.assertRaises(DirectoryError) as context:
            ensure_group_directory(self.temp_dir, 3)
        
        self.assertIn("權限不足", str(context.exception))
        self.assertIn("group_3", str(context.exception))
    
    @patch('os.makedirs')
    def test_ensure_group_directory_other_error(self, mock_makedirs):
        """測試其他創建目錄錯誤"""
        mock_makedirs.side_effect = OSError("Disk full")
        
        with self.assertRaises(DirectoryError) as context:
            ensure_group_directory(self.temp_dir, 4)
        
        self.assertIn("創建群組目錄", str(context.exception))
        self.assertIn("Disk full", str(context.exception))


class TestMoveFileToGroup(TestLpipsClusteringBase):
    """測試 move_file_to_group 函數"""
    
    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()
        self.group_dir = os.path.join(self.temp_dir, "group_1")
        os.makedirs(self.group_dir)
        
        # 創建測試檔案
        self.test_file = os.path.join(self.temp_dir, "test_image.jpg")
        with open(self.test_file, 'w') as f:
            f.write("test content")
    
    def tearDown(self):
        super().tearDown()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_move_file_to_group_success(self):
        """測試成功移動檔案"""
        result = move_file_to_group(self.test_file, self.group_dir)
        
        self.assertTrue(result)
        self.assertFalse(os.path.exists(self.test_file))
        
        moved_file = os.path.join(self.group_dir, "test_image.jpg")
        self.assertTrue(os.path.exists(moved_file))
    
    def test_move_file_to_group_source_not_exist(self):
        """測試來源檔案不存在"""
        non_existent_file = os.path.join(self.temp_dir, "non_existent.jpg")
        
        result = move_file_to_group(non_existent_file, self.group_dir)
        
        self.assertFalse(result)
    
    def test_move_file_to_group_target_not_exist(self):
        """測試目標目錄不存在"""
        non_existent_dir = os.path.join(self.temp_dir, "non_existent_group")
        
        result = move_file_to_group(self.test_file, non_existent_dir)
        
        self.assertFalse(result)
        # 原檔案應該仍然存在
        self.assertTrue(os.path.exists(self.test_file))


class TestProcessLpipsClustering(TestLpipsClusteringBase):
    """測試 process_lpips_clustering 主要處理流程"""
    
    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.temp_dir, "lpips_output")
        
        # 創建測試圖像檔案
        self.test_files = []
        for i in range(5):
            file_path = os.path.join(self.temp_dir, f"test_image_{i}.jpg")
            with open(file_path, 'w') as f:
                f.write(f"test image content {i}")
            self.test_files.append(file_path)
    
    def tearDown(self):
        super().tearDown()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('lpips_clustering.lpips_clustering')
    def test_process_lpips_clustering_success(self, mock_lpips):
        """測試成功的 LPIPS 聚類處理"""
        # 模擬 LPIPS 結果：前3個檔案在同一群組，後2個在不同群組
        mock_lpips.return_value = [0, 0, 0, 1, 2]
        
        result_dir = process_lpips_clustering(self.test_files, self.output_dir, batch_size=10)
        
        self.assertEqual(result_dir, self.output_dir)
        self.assertTrue(os.path.exists(self.output_dir))
        
        # 檢查是否創建了群組目錄 (只有重複的會被移動)
        group_0_dir = os.path.join(self.output_dir, "group_0")
        self.assertTrue(os.path.exists(group_0_dir))
        
        # 檢查檔案移動 (第2、3個檔案應該被移動到 group_0)
        moved_files = os.listdir(group_0_dir)
        self.assertEqual(len(moved_files), 2)  # 第2、3個重複檔案
    
    @patch('lpips_clustering.lpips_clustering')
    def test_process_lpips_clustering_no_duplicates(self, mock_lpips):
        """測試沒有重複的情況"""
        # 所有檔案都在不同群組
        mock_lpips.return_value = [0, 1, 2, 3, 4]
        
        result_dir = process_lpips_clustering(self.test_files, self.output_dir)
        
        self.assertEqual(result_dir, self.output_dir)
        self.assertTrue(os.path.exists(self.output_dir))
        
        # 沒有檔案應該被移動
        subdirs = [d for d in os.listdir(self.output_dir) if os.path.isdir(os.path.join(self.output_dir, d))]
        self.assertEqual(len(subdirs), 0)
    
    @patch('lpips_clustering.lpips_clustering')
    def test_process_lpips_clustering_with_noise(self, mock_lpips):
        """測試包含噪音樣本 (-1) 的情況"""
        mock_lpips.return_value = [0, -1, 0, -1, 1]
        
        result_dir = process_lpips_clustering(self.test_files, self.output_dir)
        
        self.assertEqual(result_dir, self.output_dir)
        
        # 檢查群組目錄 (只有 group_0 會被創建，因為有重複)
        group_dirs = [d for d in os.listdir(self.output_dir) 
                     if d.startswith("group_") and os.path.isdir(os.path.join(self.output_dir, d))]
        self.assertEqual(len(group_dirs), 1)
        self.assertIn("group_0", group_dirs)
    
    def test_process_lpips_clustering_output_dir_permission_error(self):
        """測試輸出目錄權限錯誤"""
        # 使用不存在的根目錄路徑來模擬權限錯誤
        if os.name == 'nt':  # Windows
            invalid_dir = "C:\\Windows\\System32\\invalid_lpips_output"
        else:  # Unix-like
            invalid_dir = "/root/invalid_lpips_output"
        
        with self.assertRaises(DirectoryError) as context:
            process_lpips_clustering(self.test_files, invalid_dir)
        
        self.assertIn("權限不足", str(context.exception))
    
    @patch('lpips_clustering.lpips_clustering')
    def test_process_lpips_clustering_batch_processing(self, mock_lpips):
        """測試批次處理"""
        # 創建更多測試檔案
        extra_files = []
        for i in range(5, 10):
            file_path = os.path.join(self.temp_dir, f"test_image_{i}.jpg")
            with open(file_path, 'w') as f:
                f.write(f"test image content {i}")
            extra_files.append(file_path)
        
        all_files = self.test_files + extra_files
        
        # 模擬批次處理結果
        mock_lpips.side_effect = [
            [0, 0, 1, 1, 2],  # 第一批次
            [0, 1, 2, 3, 4]   # 第二批次
        ]
        
        result_dir = process_lpips_clustering(all_files, self.output_dir, batch_size=5)
        
        self.assertEqual(result_dir, self.output_dir)
        self.assertTrue(os.path.exists(self.output_dir))
        
        # 應該有批次處理的效果
        self.assertEqual(mock_lpips.call_count, 2)
    
    @patch('lpips_clustering.lpips_clustering')
    def test_process_lpips_clustering_lpips_failure(self, mock_lpips):
        """測試 LPIPS 處理失敗的情況"""
        mock_lpips.side_effect = Exception("LPIPS processing failed")
        
        # 處理應該完成，但沒有檔案被移動
        result_dir = process_lpips_clustering(self.test_files, self.output_dir)
        
        self.assertEqual(result_dir, self.output_dir)
        self.assertTrue(os.path.exists(self.output_dir))
        
        # 應該沒有群組目錄被創建
        subdirs = [d for d in os.listdir(self.output_dir) if os.path.isdir(os.path.join(self.output_dir, d))]
        self.assertEqual(len(subdirs), 0)
    
    def test_process_lpips_clustering_empty_file_list(self):
        """測試空檔案列表"""
        result_dir = process_lpips_clustering([], self.output_dir)
        
        self.assertEqual(result_dir, self.output_dir)
        self.assertTrue(os.path.exists(self.output_dir))


class TestLpipsClusteringIntegration(TestLpipsClusteringBase):
    """LPIPS 聚類模組整合測試"""
    
    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.temp_dir, "integration_output")
        
        # 創建一個完整的測試環境
        self.test_images = []
        for i in range(8):
            img_path = os.path.join(self.temp_dir, f"image_{i:02d}.jpg")
            self.create_test_image_file(img_path, f"Image {i} content")
            self.test_images.append(img_path)
    
    def tearDown(self):
        super().tearDown()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('lpips_clustering.lpips_clustering')
    def test_full_workflow_integration(self, mock_lpips):
        """測試完整工作流程整合"""
        # 模擬複雜的聚類結果
        mock_lpips.return_value = [0, 1, 0, 2, 1, 3, 0, 2]
        
        result_dir = process_lpips_clustering(
            self.test_images, 
            self.output_dir, 
            batch_size=4
        )
        
        # 驗證結果
        self.assertEqual(result_dir, self.output_dir)
        self.assertTrue(os.path.exists(self.output_dir))
        
        # 檢查群組目錄
        expected_groups = ["group_0", "group_1", "group_2"]
        for group in expected_groups:
            group_path = os.path.join(self.output_dir, group)
            self.assertTrue(os.path.exists(group_path), f"群組目錄 {group} 應該存在")
        
        # 檢查檔案移動的數量
        total_moved = 0
        for group in expected_groups:
            group_path = os.path.join(self.output_dir, group)
            if os.path.exists(group_path):
                moved_files = os.listdir(group_path)
                total_moved += len(moved_files)
        
        # 根據聚類結果，應該有5個檔案被移動 (重複檔案)
        self.assertGreater(total_moved, 0)
        self.assertLessEqual(total_moved, len(self.test_images))


if __name__ == '__main__':
    unittest.main()
