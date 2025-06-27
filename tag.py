import os
import onnxruntime
from pathlib import Path
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
        # 從環境變數中讀取wildcard啟用狀態
        enable_wildcard = os.getenv("enable_wildcard", "false").lower() == "true"
        
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
            
        # 檢查 text_output 若有 "1girl,"或 "1boy,"則移至字串開頭
        tags_list = [tag.strip() for tag in text_output.split(",") if tag.strip()]
        priority_tags = ["1girl", "1boy"]
        
        # 找到第一個優先標籤並移至開頭
        for priority_tag in priority_tags:
            if priority_tag in tags_list:
                tags_list.remove(priority_tag)
                tags_list.insert(0, priority_tag)
                break
                
        text_output = ",".join(tags_list)
        
        # 儲存標籤到對應的 .txt 檔案
        txt_filename = os.path.splitext(image_path)[0] + ".txt"
        
        # 判斷是否需要添加 wildcard 行
        if enable_wildcard and custom_artist_name:
            wildcard_line = f"an anime girl in {custom_artist_name} style"
            with open(txt_filename, 'w') as txt_file:
                txt_file.write(text_output + "\n" + wildcard_line)
        else:
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

def tag_images_with_summary(image_path_or_directory):
    """
    為圖像生成標籤並返回詳細摘要
    
    Parameters:
    - image_path_or_directory: 圖像文件路徑或包含圖像的目錄路徑
    
    Returns:
    - dict: 包含處理結果摘要的字典
    """
    logs = []
    try:
        # 載入環境變數
        load_dotenv()
        
        # 驗證輸入路徑
        if not image_path_or_directory or not os.path.exists(image_path_or_directory):
            return {
                "success": False,
                "error": f"路徑不存在: {image_path_or_directory}",
                "logs": [f"路徑不存在: {image_path_or_directory}"],
                "input_path": image_path_or_directory,
                "total_images": 0,
                "processed_images": 0,
                "failed_images": 0,
                "tags_generated": 0,
                "custom_character": None,
                "custom_artist": None,
                "wildcard_enabled": False
            }
        
        logs.append(f"開始處理: {image_path_or_directory}")
        
        # 讀取環境變數設定
        custom_character_tag = os.getenv("custom_character_tag", "").strip()
        custom_artist_name = os.getenv("custom_artist_name", "").strip()
        enable_wildcard = os.getenv("enable_wildcard", "false").lower() == "true"
        num_threads_str = os.getenv("num_threads", "2")
        
        # 驗證線程數設定
        if not num_threads_str.isdigit():
            logs.append(f"警告: 環境變數 'num_threads' 值 '{num_threads_str}' 無效，使用預設值 2")
            num_threads = 2
        else:
            num_threads = int(num_threads_str)
        
        logs.append(f"使用線程數: {num_threads}")
        if custom_character_tag:
            logs.append(f"自訂角色標籤: {custom_character_tag}")
        if custom_artist_name:
            logs.append(f"自訂繪師名稱: {custom_artist_name}")
        if enable_wildcard:
            logs.append("啟用 wildcard 功能")
        
        # 收集圖像文件
        image_list = []
        
        if os.path.isfile(image_path_or_directory):
            # 單個文件
            ext = os.path.splitext(image_path_or_directory)[-1].lower()
            if ext in SUPPORTED_EXT:
                image_list.append(image_path_or_directory)
            else:
                return {
                    "success": False,
                    "error": f"不支援的文件格式: {ext}",
                    "logs": logs + [f"不支援的文件格式: {ext}"],
                    "input_path": image_path_or_directory,
                    "total_images": 0,
                    "processed_images": 0,
                    "failed_images": 0,
                    "tags_generated": 0,
                    "custom_character": custom_character_tag,
                    "custom_artist": custom_artist_name,
                    "wildcard_enabled": enable_wildcard
                }
        else:
            # 目錄
            for root, dirs, files in os.walk(image_path_or_directory):
                for file in files:
                    ext = os.path.splitext(file)[-1].lower()
                    if ext in SUPPORTED_EXT:
                        full_file_path = os.path.join(root, file)
                        image_list.append(full_file_path)
        
        if not image_list:
            return {
                "success": True,
                "error": None,
                "logs": logs + [f"在 '{image_path_or_directory}' 中未找到支援的圖像文件"],
                "input_path": image_path_or_directory,
                "total_images": 0,
                "processed_images": 0,
                "failed_images": 0,
                "tags_generated": 0,
                "custom_character": custom_character_tag,
                "custom_artist": custom_artist_name,
                "wildcard_enabled": enable_wildcard
            }
        
        logs.append(f"找到 {len(image_list)} 張圖片需要處理標籤")
        
        # 處理圖像並統計結果
        processed_count = 0
        failed_count = 0
        tags_generated_count = 0
        
        # 使用 ThreadPoolExecutor 進行多線程處理
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(tqdm(executor.map(process_image, image_list), 
                              total=len(image_list), 
                              desc="標記圖片進度"))
        
        # 統計處理結果
        for result in results:
            if result:  # process_image 返回路徑表示成功
                processed_count += 1
                # 檢查是否生成了標籤文件
                txt_filename = os.path.splitext(result)[0] + ".txt"
                if os.path.exists(txt_filename):
                    tags_generated_count += 1
            else:
                failed_count += 1
        
        logs.append(f"處理完成: 總圖片 {len(image_list)}, 成功處理 {processed_count}, 失敗 {failed_count}")
        logs.append(f"生成標籤文件 {tags_generated_count} 個")
        
        return {
            "success": True,
            "error": None,
            "logs": logs,
            "input_path": image_path_or_directory,
            "total_images": len(image_list),
            "processed_images": processed_count,
            "failed_images": failed_count,
            "tags_generated": tags_generated_count,
            "custom_character": custom_character_tag,
            "custom_artist": custom_artist_name,
            "wildcard_enabled": enable_wildcard
        }
        
    except Exception as e:
        error_msg = f"圖像標籤處理失敗: {str(e)}"
        logs.append(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "logs": logs,
            "input_path": image_path_or_directory,
            "total_images": 0,
            "processed_images": 0,
            "failed_images": 0,
            "tags_generated": 0,
            "custom_character": None,
            "custom_artist": None,
            "wildcard_enabled": False
        }
        
if __name__ == "__main__":    # 載入環境變數
    load_dotenv()
    
    # 顯示 onnxruntime 版本
    print(f"使用 onnxruntime 版本: {onnxruntime.__version__}")
    
    # 從環境變數中讀取輸出目錄並轉為 Path 物件
    directory_str = os.getenv("output_directory")
    if not directory_str:
        print("錯誤: 未設定 output_directory 環境變數")
        exit(1)
        
    directory = Path(directory_str)
    if not directory.exists() or not directory.is_dir():
        print(f"錯誤: 目錄 '{directory}' 不存在或不是有效目錄")
        exit(1)
        
    # 開始標記圖片
    processed_count = tag_image(str(directory))
    print(f"標記完成，共處理 {processed_count} 張圖片")

