import os
import shutil
from tqdm import tqdm
from imgutils.detect import detect_faces
from logger_config import get_logger
from error_handler import safe_execute, DirectoryError, ImageProcessingError, ModelError, WaifucError

# 設定日誌記錄器
logger = get_logger('face_detection')

def detect_faces_in_single_image(file_path):
    """
    檢測單一圖片中的人臉數量
    
    Args:
        file_path (str): 圖片檔案路徑
    
    Returns:
        int: 檢測到的人臉數量，錯誤時返回 -1
      Raises:
        ImageProcessingError: 當圖片處理失敗時
        ModelError: 當人臉檢測模型執行失敗時
    """
    try:
        result = detect_faces(file_path)
        face_count = len(result)
        logger.debug(f"圖片 {file_path} 檢測到 {face_count} 張人臉")
        return face_count
    except Exception as e:
        error_msg = f"人臉檢測失敗: {str(e)}"
        if any(keyword in str(e).lower() for keyword in ["model", "load", "cuda", "memory", "device"]):
            raise ModelError(error_msg, "face_detection")
        else:
            raise ImageProcessingError(error_msg, file_path)

def ensure_output_directory(output_path):
    """
    確保輸出目錄存在
    
    Args:
        output_path (str): 目錄路徑
    
    Raises:
        DirectoryError: 當無法創建目錄時
    """
    try:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            logger.debug(f"創建目錄: {output_path}")
    except PermissionError:
        raise DirectoryError(f"權限不足，無法創建目錄 '{output_path}'", output_path)
    except Exception as e:
        raise DirectoryError(f"創建目錄 '{output_path}' 時發生錯誤: {str(e)}", output_path)

def move_image_to_output(file_path, output_folder, face_count):
    """
    將圖片移動到對應的輸出目錄
    
    Args:
        file_path (str): 原始檔案路徑
        output_folder (str): 輸出基礎目錄（已包含來源目錄名稱）
        face_count (int): 人臉數量
    
    Returns:
        bool: 移動是否成功
    """
    try:
        # 根據實際人臉數量創建資料夾（不限制最大值）
        target_folder = os.path.join(output_folder, f"faces_{face_count}")
        ensure_output_directory(target_folder)
        
        # 移動檔案
        target_path = os.path.join(target_folder, os.path.basename(file_path))
        shutil.move(file_path, target_path)
        logger.debug(f"移動圖片 {file_path} 到 {target_path}")
        return True
    except Exception as e:
        logger.error(f"移動圖片 {file_path} 失敗: {str(e)}")
        return False

def detect_faces_in_directory(directory, min_face_count=1, output_base_folder="face_out"):
    """
    檢測目錄中所有圖片的人臉，並將符合條件的圖片移動到指定資料夾。
    
    Args:
        directory (str): 圖片目錄路徑
        min_face_count (int): 最小人臉數量閾值，默認為1
        output_base_folder (str): 輸出基礎資料夾名稱，默認為 'face_out'
    
    Returns:
        tuple: (處理的檔案總數, 移動的檔案數)
    
    Raises:
        DirectoryError: 當目錄不存在或無法存取時
        WaifucError: 當處理過程中發生其他錯誤時    """
    logger.info(f"開始人臉檢測: 目錄={directory}, 最小人臉數={min_face_count}, 輸出目錄={output_base_folder}")
    
    # 驗證輸入目錄
    if not os.path.isdir(directory):
        raise DirectoryError(f"目錄 '{directory}' 不存在或不是有效目錄", directory)
    
    if not os.access(directory, os.R_OK):
        raise DirectoryError(f"無法讀取目錄 '{directory}'，請檢查權限", directory)
    
    # 創建基於來源目錄名稱的輸出目錄
    source_dir_name = os.path.basename(directory.rstrip(os.sep))
    source_output_folder = os.path.join(output_base_folder, source_dir_name)
    
    # 確保輸出目錄存在
    ensure_output_directory(source_output_folder)
    source_dir_name = os.path.basename(directory.rstrip(os.sep))
    source_output_folder = os.path.join(output_base_folder, source_dir_name)
    
    # 確保輸出目錄存在
    ensure_output_directory(source_output_folder)
    
    # 取得所有圖片檔案
    try:
        all_files = os.listdir(directory)
        file_paths = []
        
        # 支援的圖片格式
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
        
        for f in all_files:
            file_path = os.path.join(directory, f)
            if os.path.isfile(file_path):
                ext = os.path.splitext(f)[-1].lower()
                if ext in image_extensions:
                    file_paths.append(file_path)
        
        logger.info(f"找到 {len(file_paths)} 個圖片檔案")
        
    except PermissionError:
        raise DirectoryError(f"權限不足，無法列舉目錄 '{directory}' 中的檔案", directory)
    except Exception as e:
        raise WaifucError(f"列舉目錄檔案時發生錯誤: {str(e)}")
    
    moved_count = 0
    
    # 檢測人臉並移動符合條件的圖片
    with tqdm(total=len(file_paths), desc="人臉檢測進度") as pbar:
        for file_path in file_paths:
            # 使用 safe_execute 安全檢測人臉
            face_count = safe_execute(
                detect_faces_in_single_image,
                file_path,
                logger=logger,
                default_return=-1,
                error_msg_prefix=f"檢測圖片 {os.path.basename(file_path)} 的人臉時"
            )            # 如果檢測成功且符合條件，移動圖片
            if face_count is not None and face_count >= 0 and face_count >= min_face_count:
                move_success = safe_execute(
                    move_image_to_output,
                    file_path,
                    source_output_folder,  # 使用基於來源的輸出目錄
                    face_count,
                    logger=logger,
                    default_return=False,
                    error_msg_prefix=f"移動圖片 {os.path.basename(file_path)} 時"
                )
                
                if move_success:
                    moved_count += 1
            
            pbar.update(1)
    
    logger.info(f"人臉檢測完成，共處理 {len(file_paths)} 張圖片，移動了 {moved_count} 張圖片")
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
        logger.error("未設定 directory 環境變數")
        print("錯誤: 未設定 directory 環境變數")
        exit(1)
    
    # 使用 safe_execute 安全執行人臉檢測
    result = safe_execute(
        detect_faces_in_directory,
        directory,
        min_face_count,
        output_base_folder,
        logger=logger,
        default_return=(0, 0),
        error_msg_prefix="執行人臉檢測時"
    )
    
    if result:
        total, moved = result
        logger.info(f"已完成 {total} 張圖片的人臉檢測，移動了 {moved} 張圖片")
        print(f"已完成 {total} 張圖片的人臉檢測，移動了 {moved} 張圖片")
    else:
        logger.error("人臉檢測失敗")
        print("人臉檢測失敗")
