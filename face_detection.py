import os
import shutil
from tqdm import tqdm
from imgutils.detect import detect_faces

def detect_faces_in_directory(directory, min_face_count=1, output_base_folder="face_out"):
    """
    檢測目錄中所有圖片的人臉，並將符合條件的圖片移動到指定資料夾。
    
    Args:
        directory (str): 圖片目錄路徑
        min_face_count (int): 最小人臉數量閾值，默認為1
        output_base_folder (str): 輸出基礎資料夾名稱，默認為 'face_out'
    
    Returns:
        tuple: (處理的檔案總數, 移動的檔案數)
    """
    if not os.path.isdir(directory):
        print(f"錯誤: 目錄 '{directory}' 不存在或不是有效目錄")
        return 0, 0
        
    # 取得資料夾內所有檔案路徑
    file_paths = [os.path.join(directory, f) for f in os.listdir(directory) 
                 if os.path.isfile(os.path.join(directory, f))]
    
    # 預先建立可能需要的輸出資料夾
    # 確保基礎輸出資料夾存在
    if not os.path.exists(output_base_folder):
        os.makedirs(output_base_folder)
    
    moved_count = 0
    # 顯示進度條
    with tqdm(total=len(file_paths), desc="人臉檢測進度") as pbar:
        for file_path in file_paths:
            try:
                result = detect_faces(file_path)
                # 計算圖片中臉的數量
                face_count = len(result)
                # 如果圖片中臉的數量大於等於 min_face_count，處理圖片
                if face_count >= min_face_count:
                    # 根據人臉數量創建資料夾
                    output_folder = os.path.join(output_base_folder, f"faces_{face_count}")
                    # 確保對應人臉數量的資料夾存在
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)
                    # 移動文件到資料夾中
                    shutil.move(file_path, os.path.join(output_folder, os.path.basename(file_path)))
                    moved_count += 1
            except Exception as e:
                print(f"處理 {file_path} 時發生錯誤: {str(e)}")
            
            # 更新進度條
            pbar.update(1)
    
    print(f"人臉檢測完成，共處理 {len(file_paths)} 張圖片，移動了 {moved_count} 張圖片")
    return len(file_paths), moved_count

if __name__ == "__main__":
    from dotenv import load_dotenv
    
    # 載入環境變數
    load_dotenv()
    
    # 從環境變數獲取設定
    directory = os.getenv("directory")
    min_face_count = int(os.getenv("min_face_count", 1))
    output_base_folder = os.getenv("face_output_directory", "face_out")
    
    if not directory:
        print("錯誤: 未設定 directory 環境變數")
        exit(1)
        
    # 執行人臉檢測
    total, moved = detect_faces_in_directory(directory, min_face_count, output_base_folder)
    print(f"已完成 {total} 張圖片的人臉檢測，移動了 {moved} 張圖片")
