# config/settings.py
import os

# 專案根目錄 (waifuc_)
# __file__ 是 config/settings.py 的路徑
# os.path.dirname(__file__) 是 config/
# os.path.dirname(os.path.dirname(__file__)) 是 waifuc_/ (專案根目錄)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 基本路徑配置
INPUT_DIR = os.path.join(BASE_DIR, 'input_images')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output_images')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
CUSTOM_CSS = os.path.join(STATIC_DIR, 'custom.css')
MODELS_DIR = os.path.join(BASE_DIR, 'models') # 存放模型檔案的建議目錄

# 確保常用目錄存在
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True) # 確保模型目錄存在

# 圖像處理預設值
DEFAULT_TARGET_SIZE = (1024, 1024)
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
# ... 其他全域配置 ...

# Gradio 應用設定
GRADIO_SERVER_NAME = "0.0.0.0"
GRADIO_SERVER_PORT = 7888
GRADIO_SHARE = False # 開發時建議 False，部署時可考慮 True
GRADIO_TITLE = "圖像處理工具 WaifuC"
GRADIO_THEME = "soft" # 例如： 'default', 'huggingface', 'soft', 'glass', 'mono'

# 日誌設定
LOG_LEVEL = "INFO" # 可選: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
LOG_ROTATION_MAX_BYTES = 10*1024*1024 # 例如 10MB
LOG_ROTATION_BACKUP_COUNT = 5

# 模型路徑 (請根據實際情況調整，或留空由各服務自行處理)
# 範例：
# FACE_DETECTION_MODEL_PATH = os.path.join(MODELS_DIR, 'face_detection_model.pth')
# UPSCALE_MODEL_PATH = os.path.join(MODELS_DIR, 'upscale_model.pth')
# TAGGING_MODEL_PATH = os.path.join(MODELS_DIR, 'tagging_model.pth')

# 可用處理步驟及其標籤 (用於UI顯示和內部邏輯)
AVAILABLE_STEPS = {
    "validate": "圖片驗證",
    "face_detect": "人臉偵測",
    "cluster": "LPIPS 聚類",
    "crop": "圖片裁切",
    "tag": "圖像標記",
    "upscale": "圖像放大"
}

# Tagging settings
TAG_MODEL_NAME = "EVA02_Large" # 模型名稱，例如 "wd14-vit-v2", "wd14-convnext-v2", "wd14-swinv2-v2", "mld-caformer", "mld-tresnet", "wd-v1-4-moat-tagger-v2", "wd-v1-4-vit-tagger-v2", "wd-v1-4-convnext-tagger-v2", "wd-v1-4-swin-tagger-v2", "Meta-CAFL", "Z3D-E", "ViT-bigG-14" , "EVA02_Large"
TAG_GENERAL_THRESHOLD = 0.35 # 通用標籤閾值
TAG_CHARACTER_THRESHOLD = 0.85 # 角色標籤閾值 (imgutils 內部使用，此處為參考)
TAG_CUSTOM_CHARACTER_TAG = "" # 自訂角色標籤，例如 "1girl, solo"
TAG_CUSTOM_ARTIST_NAME = ""   # 自訂繪師名稱
TAG_ENABLE_WILDCARD = False   # 是否啟用 wildcard 功能
TAG_WILDCARD_TEMPLATE = "an anime girl in {artist} style" # Wildcard 模板
TAG_EXCLUDED_TAGS = ["questionable", "general"] # 需要排除的標籤列表
TAG_PREPEND_TAGS = "" # 需要加到最前面的標籤，例如 "masterpiece, best quality"
TAG_APPEND_TAGS = "" # 需要加到最後面的標籤

# Upscaling settings
UPSCALE_MODEL_NAME = "HGSR-MHR-anime-aug_X4_320" # 預設放大模型
UPSCALE_TARGET_WIDTH = 2048 # 預設目標寬度
UPSCALE_TARGET_HEIGHT = 2048 # 預設目標高度
UPSCALE_PRESERVE_ASPECT_RATIO = True # 是否保持原始寬高比
UPSCALE_CENTER_CROP_AFTER_UPSCALE = True # 放大後是否置中裁切到目標尺寸
UPSCALE_TILE_SIZE = 512 # 分塊大小
UPSCALE_TILE_OVERLAP = 64 # 分塊重疊
UPSCALE_BATCH_SIZE = 1 # 批次大小
UPSCALE_MIN_SIZE_THRESHOLD = None # 最小尺寸閾值，小於此值才放大 (e.g., 1024, 如果寬或高小於1024則放大)
UPSCALE_OUTPUT_SUBDIR = "upscaled" # 放大圖片存放的子目錄名稱 (如果沒有覆寫原檔)
UPSCALE_OVERWRITE_ORIGINAL = False # 是否覆寫原始檔案

# File Utils settings
GRADIO_TEMP_DIR = os.path.join(BASE_DIR, 'temp_previews') # Gradio 預覽圖片的臨時目錄
TEMP_PROCESSING_DIR = os.path.join(BASE_DIR, 'temp_processing') # 處理過程中的臨時檔案目錄
URL_DOWNLOAD_TIMEOUT = 30 # URL 下載超時時間 (秒)

# 啟用/停用各處理步驟的開關
ENABLE_VALIDATION = True
ENABLE_FACE_DETECTION = True
ENABLE_LPIPS_CLUSTERING = True # 啟用以測試改進後的聚類效果
ENABLE_CROP = True
ENABLE_TAGGING = True
ENABLE_UPSCALE = True

# 確保這些目錄也存在
os.makedirs(GRADIO_TEMP_DIR, exist_ok=True)
os.makedirs(TEMP_PROCESSING_DIR, exist_ok=True)

# Face detection settings
FACE_DETECTION_CONFIDENCE_THRESHOLD = 0.3  # 降低閾值以偵測更多臉部
FACE_DETECTION_MODEL = "default"  # 可能需要嘗試不同模型

# LPIPS clustering settings
LPIPS_CLUSTERING_THRESHOLD = 0.3  # 進一步降低閾值以獲得更細緻的聚類
LPIPS_MIN_SAMPLES = 2  # 最小樣本數
LPIPS_BATCH_SIZE = 100  # 批次大小

# Crop settings
CROP_ENABLE_HEAD = True
CROP_ENABLE_HALFBODY = True  
CROP_ENABLE_PERSON = True
CROP_MIN_SIZE = 64  # 最小裁切尺寸

# Validation settings
VALIDATION_QUARANTINE_INVALID = True  # 是否隔離無效圖片
VALIDATION_QUARANTINE_DIR = None  # 隔離目錄，None則自動創建

# Face detection settings for training
FACE_DETECTION_AUTO_CLASSIFY = True  # 是否自動按人臉數量分類
FACE_DETECTION_TARGET_FACE_COUNT = 1  # 訓練需要的人臉數量 (0=無人臉, 1=單人, 2=雙人, -1=不限制)
FACE_DETECTION_FILTER_MODE = "keep_target"  # 過濾模式: "keep_target"(保留目標), "exclude_target"(排除目標), "classify_all"(只分類不過濾)
FACE_DETECTION_EXCLUDED_DIR = "excluded_faces"  # 不符合要求的圖片存放目錄
FACE_DETECTION_TRAINING_DIR = "training_faces"  # 符合訓練要求的圖片存放目錄

# LPIPS clustering settings
LPIPS_AUTO_ELIMINATE_DUPLICATES = True  # 是否自動淘汰重複圖片
LPIPS_ELIMINATED_DIR = None  # 淘汰圖片存放目錄，None則自動創建

# Tag service settings
TAG_AUTO_SAVE_TO_FILE = True  # 是否自動保存標籤到文件
TAG_OUTPUT_DIR = None  # 標籤文件輸出目錄，None則與圖片同目錄
