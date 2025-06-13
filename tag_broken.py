import os
import onnxruntime
from dotenv import load_dotenv
from imgutils.tagging import get_wd14_tags, tags_to_text
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from logger_config import get_logger
from error_handler import safe_execute, DirectoryError, ImageProcessingError, ModelError, WaifucError

# 設定日誌記錄器
logger = get_logger('tag')

# 支援的影像格式
SUPPORTED_EXT = [".jpg", ".png", ".jpeg", ".bmp", ".jfif", ".webp"]

def safe_process_single_image(image_path):
    """
    安全處理單張圖片，生成標籤並儲存到對應的 .txt 檔案中。

    Args:
        image_path (str): 圖片的完整路徑。
        
    Returns:
        dict: 包含處理結果的字典
        
    Raises:
        ImageProcessingError: 當圖片處理失敗時
        ModelError: 當模型執行失敗時
    """
    try:
        # 檢查檔案是否存在
        if not os.path.isfile(image_path):
            raise ImageProcessingError(f"圖片檔案不存在: {image_path}", image_path)
          # 檢查檔案是否可讀取
        if not os.access(image_path, os.R_OK):
            raise ImageProcessingError(f"無法讀取圖片檔案: {image_path}，請檢查權限", image_path)
        
        try:
            # 使用模型生成圖片的標籤
            rating, features, chars = get_wd14_tags(image_path, model_name="EVA02_Large")        except Exception as e:
            error_msg = f"標記模型處理失敗: {str(e)}"
            if ("model" in str(e).lower() or "download" in str(e).lower() or 
                "onnx" in str(e).lower() or "cuda" in str(e).lower()):
                raise ModelError(error_msg, "wd14_tagger")
            else:
                raise ImageProcessingError(error_msg, image_path)
        
        # 從環境變數中讀取設定
        custom_character_tag = os.getenv("custom_character_tag", "").strip()
        custom_artist_name = os.getenv("custom_artist_name", "").strip()
        enable_wildcard = os.getenv("enable_wildcard", "false").lower() == "true"
        threshold = float(os.getenv("tag_threshold", "0.3"))  # 添加閾值支持
        
        # 初始化標籤部分
        parts = []
        
        # 1. 處理角色標籤 (chars)
        if len(chars) != 0:
            parts.append("".join(chars))
        elif custom_character_tag:
            parts.append(custom_character_tag)
            
        # 2. 處理繪師名稱 (artist_name)
        if custom_artist_name:
            parts.append(custom_artist_name)        # 3. 處理特徵標籤 (features)
        # 過濾低於閾值的標籤
        filtered_features = {k: v for k, v in features.items() if v >= threshold}
        feature_tags = tags_to_text(filtered_features)
        
        # 4. 組合所有標籤
        parts.append(feature_tags)
        text_output = ",".join(parts)
            
        # 檢查 text_output 若有 "1girl," 則移至字串開頭
        if "1girl," in text_output:
            text_output = "1girl," + text_output.replace("1girl,", "")
          # 儲存標籤到對應的 .txt 檔案
        txt_filename = os.path.splitext(image_path)[0] + ".txt"
        
        try:
            # 判斷是否需要添加 wildcard 行
            if enable_wildcard and custom_artist_name:
                wildcard_line = f"an anime girl in {custom_artist_name} style"
                with open(txt_filename, 'w', encoding='utf-8') as txt_file:
                    txt_file.write(text_output + "\n" + wildcard_line)
            else:
                with open(txt_filename, 'w', encoding='utf-8') as txt_file:
                    txt_file.write(text_output)
        except PermissionError:
            raise ImageProcessingError(f"權限不足，無法寫入標籤檔案: {txt_filename}", image_path)
        except Exception as e:
            raise ImageProcessingError(f"寫入標籤檔案失敗: {str(e)}", image_path)        
        logger.debug(f"成功處理圖片: {image_path}")
        return {
            "success": True, 
            "image_path": image_path,
            "tag_text": text_output,
            "features_count": len(features),
            "characters_count": len(chars),
            "tags_count": len(features) + len(chars), 
            "txt_file": txt_filename
        }
        
    except (ImageProcessingError, ModelError):
        raise
    except Exception as e:
        raise ImageProcessingError(f"處理圖片時發生未預期的錯誤: {str(e)}", image_path)

def process_image(image_path):
    """
    處理單張圖片，生成標籤並儲存到對應的 .txt 檔案中。
    使用安全包裝函數來處理錯誤

    Args:
        image_path (str): 圖片的完整路徑。
        
    Returns:
        str: 處理完成的圖片路徑
    """
    result = safe_execute(
        safe_process_single_image,
        image_path,
        logger=logger,
        default_return=None,
        error_msg_prefix=f"處理圖片 {os.path.basename(image_path)} 時"
    )
    
    if result and result.get("success"):
        logger.debug(f"圖片 {image_path} 標記完成，產生 {result.get('tags_count', 0)} 個標籤")
    
    return image_path

def tag_image(image_path):
    """
    遍歷目錄中的所有圖片，為每張圖片生成標籤。

    Args:
        image_path (str): 包含圖片的目錄路徑。
        
    Returns:
        int: 處理的圖片數量
        
    Raises:
        DirectoryError: 當目錄不存在或無法存取時
        WaifucError: 當處理過程中發生其他錯誤時
    """
    logger.info(f"開始圖片標記處理: {image_path}")
    
    # 載入環境變數
    load_dotenv()
    
    # 驗證目錄
    if not os.path.isdir(image_path):
        raise DirectoryError(f"目錄 '{image_path}' 不存在或不是有效目錄", image_path)
    
    if not os.access(image_path, os.R_OK):
        raise DirectoryError(f"無法讀取目錄 '{image_path}'，請檢查權限", image_path)
    
    # 定義影像列表
    image_list = []

    # 遍歷目錄和所有子目錄，收集支援的影像格式
    try:
        for root, dirs, files in os.walk(image_path):
            for file in files:
                ext = os.path.splitext(file)[-1].lower()
                if ext in SUPPORTED_EXT:
                    full_file_path = os.path.join(root, file)
                    image_list.append(full_file_path)
    except PermissionError:
        raise DirectoryError(f"權限不足，無法列舉目錄 '{image_path}' 中的檔案", image_path)
    except Exception as e:
        raise WaifucError(f"列舉目錄檔案時發生錯誤: {str(e)}")

    logger.info(f"找到 {len(image_list)} 張圖片需要處理標籤")
    print(f"找到 {len(image_list)} 張圖片需要處理標籤。")
    
    if not image_list:
        logger.warning("未找到任何支援的圖片檔案")
        return 0
    
    # 從環境變數中讀取線程數，並進行驗證
    num_threads_str = os.getenv("num_threads", "2")  
    if not num_threads_str.isdigit():
        logger.warning(f"環境變數 'num_threads' 值 '{num_threads_str}' 無效，使用預設值 2")
        print(f"警告: 環境變數 'num_threads' 值 '{num_threads_str}' 無效，使用預設值 2")
        num_threads = 2
    else:
        num_threads = int(num_threads_str)
        
    logger.info(f"使用 {num_threads} 個線程進行並行處理")

    # 使用 ThreadPoolExecutor 進行多線程處理，並顯示進度條
    try:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(tqdm(executor.map(process_image, image_list), 
                              total=len(image_list), 
                              desc="標記圖片進度"))
    except Exception as e:
        raise WaifucError(f"多線程處理時發生錯誤: {str(e)}")
        
    logger.info(f"圖片標記完成，共處理 {len(image_list)} 張圖片")
    return len(image_list)
        
if __name__ == "__main__":
    # 載入環境變數
    load_dotenv()
    
    # 顯示 onnxruntime 版本
    logger.info(f"使用 onnxruntime 版本: {onnxruntime.__version__}")
    print(f"使用 onnxruntime 版本: {onnxruntime.__version__}")
    
    # 從環境變數中讀取輸出目錄
    directory = os.getenv("output_directory")
    if not directory:
        logger.error("未設定 output_directory 環境變數")
        print("錯誤: 未設定 output_directory 環境變數")
        exit(1)
    
    # 使用 safe_execute 安全執行標記處理
    processed_count = safe_execute(
        tag_image,
        directory,
        logger=logger,
        default_return=0,
        error_msg_prefix="執行圖片標記時"
    )
    
    if processed_count is not None and processed_count > 0:
        logger.info(f"標記完成，共處理 {processed_count} 張圖片")
        print(f"標記完成，共處理 {processed_count} 張圖片")
    else:
        logger.error("圖片標記處理失敗")
        print("圖片標記處理失敗")

