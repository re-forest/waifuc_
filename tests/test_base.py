"""
測試基礎設定和工具函數
"""
import os
import tempfile
import shutil
from pathlib import Path
from PIL import Image
from typing import Optional, Dict, List, Any
from unittest.mock import patch, MagicMock

# 測試資料目錄
TEST_DATA_DIR = Path(__file__).parent / "test_data"
TEMP_TEST_DIR = None


class TestConfig:
    """測試配置類別"""
    
    # 測試圖片規格
    TEST_IMAGE_SIZE = (256, 256)
    TEST_IMAGE_MODE = "RGB"
    TEST_IMAGE_COLOR = (255, 0, 0)  # 紅色
    
    # 測試檔案名稱
    VALID_IMAGE_NAMES = ["test1.jpg", "test2.png", "test3.jpeg"]
    INVALID_IMAGE_NAMES = ["corrupted.jpg", "empty.png"]
    NON_IMAGE_NAMES = ["text.txt", "data.json"]


def setup_test_environment():
    """設定測試環境"""
    global TEMP_TEST_DIR
    TEMP_TEST_DIR = tempfile.mkdtemp(prefix="waifuc_test_")
    
    # 創建測試資料目錄
    test_data_path = Path(TEMP_TEST_DIR) / "test_data"
    test_data_path.mkdir(exist_ok=True)
    
    return TEMP_TEST_DIR


def teardown_test_environment():
    """清理測試環境"""
    global TEMP_TEST_DIR
    if TEMP_TEST_DIR and os.path.exists(TEMP_TEST_DIR):
        shutil.rmtree(TEMP_TEST_DIR)
        TEMP_TEST_DIR = None


def create_test_image(path: str, size: Optional[tuple] = None, mode: Optional[str] = None, color: Optional[tuple] = None):
    """
    創建測試用圖片檔案
    
    Args:
        path (str): 圖片檔案路徑
        size (tuple): 圖片尺寸，預設 (256, 256)
        mode (str): 圖片模式，預設 "RGB"
        color (tuple): 圖片顏色，預設 (255, 0, 0)
    """
    size = size or TestConfig.TEST_IMAGE_SIZE
    mode = mode or TestConfig.TEST_IMAGE_MODE
    color = color or TestConfig.TEST_IMAGE_COLOR
    
    # 確保目錄存在
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # 創建圖片
    img = Image.new(mode, size, color)
    img.save(path)


def create_corrupted_image(path: str):
    """
    創建損壞的測試圖片檔案
    
    Args:
        path (str): 圖片檔案路徑
    """
    # 確保目錄存在
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # 寫入無效的圖片資料
    with open(path, 'wb') as f:
        f.write(b'invalid image data')


def create_test_directory_structure(base_path: str):
    """
    創建標準的測試目錄結構
    
    Args:
        base_path (str): 基礎路徑
        
    Returns:
        dict: 包含各種測試檔案路徑的字典
    """
    paths = {
        'input_dir': os.path.join(base_path, 'input'),
        'output_dir': os.path.join(base_path, 'output'),
        'valid_images': [],
        'corrupted_images': [],
        'non_images': []
    }
    
    # 創建目錄
    os.makedirs(paths['input_dir'], exist_ok=True)
    os.makedirs(paths['output_dir'], exist_ok=True)
    
    # 創建有效圖片
    for name in TestConfig.VALID_IMAGE_NAMES:
        img_path = os.path.join(paths['input_dir'], name)
        create_test_image(img_path)
        paths['valid_images'].append(img_path)
    
    # 創建損壞圖片
    for name in TestConfig.INVALID_IMAGE_NAMES:
        img_path = os.path.join(paths['input_dir'], name)
        create_corrupted_image(img_path)
        paths['corrupted_images'].append(img_path)
    
    # 創建非圖片檔案
    for name in TestConfig.NON_IMAGE_NAMES:
        file_path = os.path.join(paths['input_dir'], name)
        with open(file_path, 'w') as f:
            f.write("test content")
        paths['non_images'].append(file_path)
    
    return paths


def mock_env_vars(**kwargs):
    """
    Mock 環境變數的便利函數
    
    Args:
        **kwargs: 要設定的環境變數
        
    Returns:
        patch: unittest.mock.patch 物件
    """
    env_vars = {
        'directory': '',
        'min_face_count': '1',
        'face_output_directory': 'face_out',
        'lpips_output_directory': 'lpips_output',
        'lpips_batch_size': '100',
        'output_directory': 'output',
        'num_threads': '2',
        'custom_character_tag': '',
        'custom_artist_name': '',
        'enable_wildcard': 'false',        
        'LOG_LEVEL': 'INFO',
        'LOG_TO_FILE': 'true'
    }
    env_vars.update(kwargs)
    
    return patch.dict(os.environ, env_vars)


class MockImageModel:
    """模擬圖像處理模型的 Mock 類別"""
    
    @staticmethod
    def process_image(image_path):
        """模擬圖像處理"""
        return {"success": True, "result": "mock_result"}
    
    @staticmethod
    def detect_faces(image_path):
        """模擬人臉檢測"""
        # 根據檔案名稱返回不同的人臉數量
        if "face1" in image_path:
            return [{"face": 1}]
        elif "face2" in image_path:
            return [{"face": 1}, {"face": 2}]
        else:
            return []
    
    @staticmethod
    def tag_image(image_path):
        """模擬圖像標記"""
        return ("rating", {"tag1": 0.9, "tag2": 0.8}, {"char1": 0.7})


def assert_file_exists(file_path: str, message: Optional[str] = None):
    """斷言檔案存在"""
    msg = message or f"檔案應該存在: {file_path}"
    assert os.path.exists(file_path), msg


def assert_file_not_exists(file_path: str, message: Optional[str] = None):
    """斷言檔案不存在"""
    msg = message or f"檔案不應該存在: {file_path}"
    assert not os.path.exists(file_path), msg


def assert_directory_exists(dir_path: str, message: Optional[str] = None):
    """斷言目錄存在"""
    msg = message or f"目錄應該存在: {dir_path}"
    assert os.path.isdir(dir_path), msg


def count_files_in_directory(dir_path: str, extension: Optional[str] = None):
    """計算目錄中的檔案數量"""
    if not os.path.exists(dir_path):
        return 0
    
    files = os.listdir(dir_path)
    if extension:
        files = [f for f in files if f.lower().endswith(extension.lower())]
    
    return len([f for f in files if os.path.isfile(os.path.join(dir_path, f))])
