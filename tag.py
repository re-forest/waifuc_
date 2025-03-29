import os
import onnxruntime
from dotenv import load_dotenv
from imgutils.tagging import get_wd14_tags, tags_to_text
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# 支援的影像格式
SUPPORTED_EXT = [".jpg", ".png", ".jpeg", ".bmp", ".jfif", ".webp"]

def process_image(image_path):
    """
    處理單張圖片，生成標籤並儲存到對應的 .txt 檔案中。

    Args:
        image_path (str): 圖片的完整路徑。
        
    Returns:
        str: 處理完成的圖片路徑
    """
    try:
        # 使用模型生成圖片的標籤
        rating, features, chars = get_wd14_tags(image_path, model_name="EVA02_Large")
        
        # 從環境變數中讀取自訂角色標籤，預設為空字串，並去除多餘空白
        custom_character_tag = os.getenv("custom_character_tag", "").strip()
        # 從環境變數中讀取自訂繪師名稱，預設為空字串，並去除多餘空白
        custom_artist_name = os.getenv("custom_artist_name", "").strip()
        
        # 初始化標籤部分
        parts = []
        
        # 1. 處理角色標籤 (chars)
        if len(chars) != 0:
            parts.append("".join(chars))
        elif custom_character_tag:
            parts.append(custom_character_tag)
            
        # 2. 處理繪師名稱 (artist_name)
        if custom_artist_name:
            parts.append(custom_artist_name)
            
        # 3. 處理特徵標籤 (features)
        feature_tags = tags_to_text(features)
        
        # 4. 組合所有標籤
        parts.append(feature_tags)
        text_output = ",".join(parts)
            
        # 檢查 text_output 若有 "1girl," 則移至字串開頭
        if "1girl," in text_output:
            text_output = "1girl," + text_output.replace("1girl,", "")
        
        # 儲存標籤到對應的 .txt 檔案
        txt_filename = os.path.splitext(image_path)[0] + ".txt"
        with open(txt_filename, 'w') as txt_file:
            txt_file.write(text_output)
    except Exception as e:
        print(f"處理 {image_path} 時發生錯誤: {str(e)}")
    
    return image_path

def tag_image(image_path):
    """
    遍歷目錄中的所有圖片，為每張圖片生成標籤。

    Args:
        image_path (str): 包含圖片的目錄路徑。
        
    Returns:
        int: 處理的圖片數量
    """
    # 載入環境變數
    load_dotenv()
    
    # 定義影像列表
    image_list = []

    # 遍歷目錄和所有子目錄，收集支援的影像格式
    for root, dirs, files in os.walk(image_path):
        for file in files:
            ext = os.path.splitext(file)[-1].lower()
            if ext in SUPPORTED_EXT:
                full_file_path = os.path.join(root, file)
                image_list.append(full_file_path)

    print(f"找到 {len(image_list)} 張圖片需要處理標籤。")
    
    # 從環境變數中讀取線程數，並進行驗證
    num_threads_str = os.getenv("num_threads", "2")  
    if not num_threads_str.isdigit():
        print(f"警告: 環境變數 'num_threads' 值 '{num_threads_str}' 無效，使用預設值 2")
        num_threads = 2
    else:
        num_threads = int(num_threads_str)

    # 使用 ThreadPoolExecutor 進行多線程處理，並顯示進度條
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(tqdm(executor.map(process_image, image_list), 
                          total=len(image_list), 
                          desc="標記圖片進度"))
        
    return len(image_list)
        
if __name__ == "__main__":
    # 載入環境變數
    load_dotenv()
    
    # 顯示 onnxruntime 版本
    print(f"使用 onnxruntime 版本: {onnxruntime.__version__}")
    
    # 從環境變數中讀取輸出目錄
    directory = os.getenv("output_directory")
    if not directory or not os.path.isdir(directory):
        print(f"錯誤: 目錄 '{directory}' 不存在或不是有效目錄")
        exit(1)
        
    # 開始標記圖片
    processed_count = tag_image(directory)
    print(f"標記完成，共處理 {processed_count} 張圖片")

