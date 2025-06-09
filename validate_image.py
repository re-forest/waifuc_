import os
from PIL import Image
from tqdm import tqdm

# 支援的影像格式
SUPPORTED_EXT = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".webp",
]

def validate_and_remove_invalid_images(directory_path):
    """
    驗證目錄中所有圖片的完整性，並刪除無效或損壞的圖片
    
    Args:
        directory_path (str): 圖片目錄路徑
    
    Returns:
        tuple: 包含處理的總檔案數和刪除的檔案數
    """
    if not os.path.isdir(directory_path):
        print(f"錯誤: 目錄 '{directory_path}' 不存在或不是有效目錄")
        return 0, 0
        
    files = os.listdir(directory_path)
    removed_count = 0

    for filename in tqdm(files, desc="驗證圖片進度"):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            ext = os.path.splitext(filename)[-1].lower()
            if ext not in SUPPORTED_EXT:
                continue
            try:
                # 嘗試開啟圖像文件
                with Image.open(file_path) as img:
                    img.load()
                # 不輸出每張圖片的驗證結果
            except Exception as e:
                # 刪除損壞或不完整的圖像文件，並輸出刪除訊息
                os.remove(file_path)
                removed_count += 1
                print(f"已移除 '{filename}': {str(e)}")
    
    print(f"圖片驗證完成，共處理 {len(files)} 張圖片，移除 {removed_count} 張無效圖片")
    return len(files), removed_count

if __name__ == "__main__":
    from dotenv import load_dotenv
    
    # 載入環境變數
    load_dotenv()
    
    # 從環境變數獲取目錄
    directory_path = os.getenv("directory")
    if not directory_path:
        print("錯誤: 未設定 directory 環境變數")
        exit(1)
        
    # 執行驗證
    total, removed = validate_and_remove_invalid_images(directory_path)
    print(f"驗證完成: 共處理 {total} 個檔案，移除 {removed} 個無效檔案")
