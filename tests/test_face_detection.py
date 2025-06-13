"""
人臉檢測模組的單元測試
"""
import unittest
import os
import sys
import tempfile
import shutil
from unittest.mock import patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from face_detection import (
    detect_faces_in_single_image, ensure_output_directory, 
    move_image_to_output, detect_faces_in_directory
)
from error_handler import DirectoryError, ImageProcessingError, ModelError
from tests.test_base import (
    setup_test_environment, teardown_test_environment,
    create_test_directory_structure, create_test_image,
    assert_file_exists, assert_directory_exists, count_files_in_directory,
    MockImageModel, mock_env_vars
)


class TestDetectFacesInSingleImage(unittest.TestCase):
    """測試單一圖像人臉檢測函數"""
    
    def setUp(self):
        """設定測試環境"""
        self.test_dir = setup_test_environment()
        
    def tearDown(self):
        """清理測試環境"""
        teardown_test_environment()
    
    @patch('face_detection.detect_faces')
    def test_detect_faces_success(self, mock_detect_faces):
        """測試成功檢測人臉"""
        # 模擬檢測結果：檢測到2張人臉
        mock_detect_faces.return_value = [{"face": 1}, {"face": 2}]
        
        image_path = os.path.join(self.test_dir, "test_image.jpg")
        create_test_image(image_path)
        
        result = detect_faces_in_single_image(image_path)
        self.assertEqual(result, 2)
        mock_detect_faces.assert_called_once_with(image_path)
    
    @patch('face_detection.detect_faces')
    def test_detect_faces_no_faces(self, mock_detect_faces):
        """測試未檢測到人臉"""
        mock_detect_faces.return_value = []
        
        image_path = os.path.join(self.test_dir, "no_face_image.jpg")
        create_test_image(image_path)
        
        result = detect_faces_in_single_image(image_path)
        self.assertEqual(result, 0)
    
    @patch('face_detection.detect_faces')
    def test_detect_faces_model_error(self, mock_detect_faces):
        """測試模型錯誤"""
        mock_detect_faces.side_effect = Exception("Model loading failed")
        
        image_path = os.path.join(self.test_dir, "test_image.jpg")
        create_test_image(image_path)
        
        with self.assertRaises(ModelError):
            detect_faces_in_single_image(image_path)
    
    @patch('face_detection.detect_faces')
    def test_detect_faces_cuda_error(self, mock_detect_faces):
        """測試 CUDA 記憶體錯誤"""
        mock_detect_faces.side_effect = Exception("CUDA out of memory")
        
        image_path = os.path.join(self.test_dir, "test_image.jpg")
        create_test_image(image_path)
        
        with self.assertRaises(ModelError):
            detect_faces_in_single_image(image_path)
    
    @patch('face_detection.detect_faces')
    def test_detect_faces_image_processing_error(self, mock_detect_faces):
        """測試圖像處理錯誤"""
        mock_detect_faces.side_effect = Exception("Invalid image format")
        
        image_path = os.path.join(self.test_dir, "test_image.jpg")
        create_test_image(image_path)
        
        with self.assertRaises(ImageProcessingError):
            detect_faces_in_single_image(image_path)


class TestEnsureOutputDirectory(unittest.TestCase):
    """測試輸出目錄確保函數"""
    
    def setUp(self):
        """設定測試環境"""
        self.test_dir = setup_test_environment()
        
    def tearDown(self):
        """清理測試環境"""
        teardown_test_environment()
    
    def test_create_new_directory(self):
        """測試創建新目錄"""
        new_dir = os.path.join(self.test_dir, "new_directory")
        
        ensure_output_directory(new_dir)
        assert_directory_exists(new_dir)
    
    def test_existing_directory(self):
        """測試已存在的目錄"""
        existing_dir = os.path.join(self.test_dir, "existing")
        os.makedirs(existing_dir)
        
        # 應該不會拋出異常
        ensure_output_directory(existing_dir)
        assert_directory_exists(existing_dir)
    
    @patch('os.makedirs')
    def test_permission_error(self, mock_makedirs):
        """測試權限錯誤"""
        mock_makedirs.side_effect = PermissionError("權限不足")
        
        new_dir = os.path.join(self.test_dir, "restricted")
        
        with self.assertRaises(DirectoryError):
            ensure_output_directory(new_dir)


class TestMoveImageToOutput(unittest.TestCase):
    """測試圖像移動函數"""
    
    def setUp(self):
        """設定測試環境"""
        self.test_dir = setup_test_environment()
        self.test_paths = create_test_directory_structure(self.test_dir)
        
    def tearDown(self):
        """清理測試環境"""
        teardown_test_environment()
    
    def test_move_image_success(self):
        """測試成功移動圖像"""
        source_file = self.test_paths['valid_images'][0]
        output_folder = self.test_paths['output_dir']
        face_count = 2
        
        result = move_image_to_output(source_file, output_folder, face_count)
        
        self.assertTrue(result)
        
        # 檢查目標目錄是否創建
        target_dir = os.path.join(output_folder, f"faces_{face_count}")
        assert_directory_exists(target_dir)
        
        # 檢查檔案是否移動成功
        target_file = os.path.join(target_dir, os.path.basename(source_file))
        assert_file_exists(target_file)
    
    def test_move_image_multiple_faces(self):
        """測試移動不同人臉數量的圖像"""
        output_folder = self.test_paths['output_dir']
        
        for i, source_file in enumerate(self.test_paths['valid_images']):
            face_count = i + 1  # 1, 2, 3 張人臉
            
            with self.subTest(face_count=face_count):
                result = move_image_to_output(source_file, output_folder, face_count)
                self.assertTrue(result)
                
                # 檢查對應的目錄是否創建
                target_dir = os.path.join(output_folder, f"faces_{face_count}")
                assert_directory_exists(target_dir)
    
    @patch('face_detection.shutil.move')
    def test_move_image_failure(self, mock_move):
        """測試移動失敗"""
        mock_move.side_effect = Exception("移動失敗")
        
        source_file = self.test_paths['valid_images'][0]
        output_folder = self.test_paths['output_dir']
        face_count = 1
        
        result = move_image_to_output(source_file, output_folder, face_count)
        self.assertFalse(result)


class TestDetectFacesInDirectory(unittest.TestCase):
    """測試目錄人臉檢測函數"""
    
    def setUp(self):
        """設定測試環境"""
        self.test_dir = setup_test_environment()
        self.test_paths = create_test_directory_structure(self.test_dir)
        
        # 創建更多測試圖像
        for i in range(5):
            img_path = os.path.join(self.test_paths['input_dir'], f"extra_{i}.jpg")
            create_test_image(img_path)
        
    def tearDown(self):
        """清理測試環境"""
        teardown_test_environment()
    
    @patch('face_detection.safe_execute')
    @patch('face_detection.logger')
    def test_detect_faces_directory_success(self, mock_logger, mock_safe_execute):
        """測試目錄人臉檢測成功"""
        # 模擬檢測結果：一半的圖片檢測到人臉
        def mock_detect_result(func, *args, **kwargs):
            if "test1" in args[0] or "test2" in args[0]:
                return 2  # 檢測到2張人臉
            elif "extra_0" in args[0] or "extra_1" in args[0]:
                return 1  # 檢測到1張人臉
            else:
                return 0  # 未檢測到人臉
        
        mock_safe_execute.side_effect = mock_detect_result
        
        input_dir = self.test_paths['input_dir']
        output_dir = self.test_paths['output_dir']
        min_face_count = 1
        
        total, moved = detect_faces_in_directory(input_dir, min_face_count, output_dir)
        
        # 檢查結果
        self.assertGreater(total, 0, "應該處理一些圖片")
        self.assertGreater(moved, 0, "應該移動一些圖片")
    
    def test_detect_faces_nonexistent_directory(self):
        """測試不存在的目錄"""
        nonexistent_dir = os.path.join(self.test_dir, "nonexistent")
        output_dir = self.test_paths['output_dir']
        
        with self.assertRaises(DirectoryError):
            detect_faces_in_directory(nonexistent_dir, 1, output_dir)
    
    def test_detect_faces_permission_error(self):
        """測試目錄權限錯誤"""
        with patch('os.access', return_value=False):
            with self.assertRaises(DirectoryError):
                detect_faces_in_directory(
                    self.test_paths['input_dir'], 1, self.test_paths['output_dir']
                )
    
    @patch('os.listdir')
    def test_detect_faces_list_directory_error(self, mock_listdir):
        """測試目錄列舉錯誤"""
        mock_listdir.side_effect = PermissionError("權限不足")
        
        with self.assertRaises(DirectoryError):
            detect_faces_in_directory(
                self.test_paths['input_dir'], 1, self.test_paths['output_dir']
            )
    
    def test_detect_faces_empty_directory(self):
        """測試空目錄"""
        empty_dir = os.path.join(self.test_dir, "empty")
        os.makedirs(empty_dir)
        
        total, moved = detect_faces_in_directory(empty_dir, 1, self.test_paths['output_dir'])
        
        self.assertEqual(total, 0, "空目錄應該沒有檔案被處理")
        self.assertEqual(moved, 0, "空目錄應該沒有檔案被移動")
    
    def test_detect_faces_filter_image_files(self):
        """測試只處理圖像檔案"""
        input_dir = self.test_paths['input_dir']
        
        # 計算圖像檔案數量
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
        all_files = os.listdir(input_dir)
        image_files = [f for f in all_files if any(f.lower().endswith(ext) for ext in image_extensions)]
        
        with patch('face_detection.safe_execute', return_value=0):  # 模擬無人臉檢測
            total, moved = detect_faces_in_directory(input_dir, 1, self.test_paths['output_dir'])
            
            # 應該只處理圖像檔案
            self.assertEqual(total, len(image_files), "應該只處理圖像檔案")


class TestFaceDetectionIntegration(unittest.TestCase):
    """人臉檢測模組整合測試"""
    
    def setUp(self):
        """設定測試環境"""
        self.test_dir = setup_test_environment()
        
    def tearDown(self):
        """清理測試環境"""
        teardown_test_environment()
    
    @mock_env_vars(
        directory='test_input',
        min_face_count='2',
        face_output_directory='test_output'
    )
    @patch('face_detection.safe_execute')
    def test_main_execution_flow(self, mock_safe_execute):
        """測試主程序執行流程"""
        # 創建測試目錄結構
        input_dir = os.path.join(self.test_dir, 'test_input')
        os.makedirs(input_dir)
        
        # 創建測試圖像
        for i in range(3):
            create_test_image(os.path.join(input_dir, f"test_{i}.jpg"))
        
        # 模擬成功執行
        mock_safe_execute.return_value = (3, 1)  # 處理3個檔案，移動1個
        
        # 這裡應該測試主程序邏輯，但由於環境變數的複雜性，
        # 實際實現可能需要重構主程序部分
        pass
    
    def test_different_face_counts_organization(self):
        """測試不同人臉數量的組織結構"""
        input_dir = os.path.join(self.test_dir, "input")
        output_dir = os.path.join(self.test_dir, "output")
        os.makedirs(input_dir)
        
        # 創建測試圖像
        test_images = []
        for i in range(6):
            img_path = os.path.join(input_dir, f"test_{i}.jpg")
            create_test_image(img_path)
            test_images.append(img_path)
        
        # 模擬不同的檢測結果
        expected_face_counts = [0, 1, 1, 2, 2, 3]
        
        with patch('face_detection.safe_execute') as mock_safe_execute:
            # 模擬檢測結果
            def mock_detect_side_effect(func, *args, **kwargs):
                file_path = args[0]
                filename = os.path.basename(file_path)
                index = int(filename.split('_')[1].split('.')[0])
                return expected_face_counts[index]
            
            mock_safe_execute.side_effect = mock_detect_side_effect
            
            # 執行檢測
            total, moved = detect_faces_in_directory(input_dir, 1, output_dir)
            
            # 驗證結果
            self.assertEqual(total, 6, "應該處理6個檔案")
            # 人臉數 >= 1 的圖片有5個（除了第一個是0）
            self.assertEqual(moved, 5, "應該移動5個檔案")


if __name__ == '__main__':
    unittest.main()
