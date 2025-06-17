# utils/file_utils.py
import os
import shutil
import uuid
from urllib.parse import urlparse
import requests
from PIL import Image

from config import settings # 假設 settings 包含 TEMP_DIR 或類似配置
# from utils.logger_config import setup_logging # 如果需要獨立日誌

# 可以在此處獲取一個 logger 實例，如果 file_utils 需要獨立日誌記錄
# logger = setup_logging(__name__, settings.LOG_DIR, settings.LOG_LEVEL)

def prepare_preview_image(pil_image_or_path, logger, temp_dir=None):
    """
    準備圖片用於 Gradio 預覽。
    如果輸入是 PIL Image 物件，則將其儲存到臨時目錄並返回路徑。
    如果輸入是路徑，則檢查其安全性 (是否在允許範圍內)，如果不在或為了確保權限，則複製到臨時目錄。
    返回一個 Gradio 可以安全顯示的圖片路徑。

    Args:
        pil_image_or_path (Image.Image | str): PIL Image 物件或圖片檔案路徑。
        logger: 日誌記錄器實例。
        temp_dir (str, optional): 臨時預覽目錄。預設從 settings 獲取。

    Returns:
        str | None: 可供 Gradio 預覽的圖片路徑，或在失敗時返回 None。
    """
    if temp_dir is None:
        temp_dir = getattr(settings, 'GRADIO_TEMP_DIR', os.path.join(settings.BASE_DIR, 'temp_previews'))
    os.makedirs(temp_dir, exist_ok=True) # 確保臨時目錄存在

    logger.info(f"[FileUtils] Preparing preview for: {type(pil_image_or_path)}")
    
    if isinstance(pil_image_or_path, str): # 輸入是路徑
        try:
            if not os.path.exists(pil_image_or_path):
                logger.error(f"[FileUtils] Image path does not exist: {pil_image_or_path}")
                return None
            
            # 檢查路徑是否已經在 Gradio 安全的目錄下
            # 這裡的 is_path_safe_for_gradio 假設 GRADIO_TEMP_DIR 本身是安全的
            # 如果圖片已經在 temp_dir (GRADIO_TEMP_DIR) 內，則無需複製
            if os.path.abspath(os.path.dirname(pil_image_or_path)) == os.path.abspath(temp_dir):
                 logger.info(f"[FileUtils] Image {pil_image_or_path} is already in Gradio temp dir. No copy needed.")
                 return pil_image_or_path

            # 為了 Gradio 權限和避免直接暴露原始檔案系統結構，通常複製到專用臨時目錄是個好做法
            # 即使原路徑理論上可被 Gradio 存取，複製可以提供一層隔離
            unique_filename = f"{uuid.uuid4()}{os.path.splitext(pil_image_or_path)[1]}"
            preview_path = os.path.join(temp_dir, unique_filename)
            shutil.copy(pil_image_or_path, preview_path)
            logger.info(f"[FileUtils] Copied image from {pil_image_or_path} to preview path {preview_path}")
            return preview_path
        except Exception as e:
            logger.error(f"[FileUtils] Error processing image path {pil_image_or_path} for preview: {e}", exc_info=True)
            return None
    elif isinstance(pil_image_or_path, Image.Image): # 輸入是 PIL Image
        try:
            unique_filename = f"{uuid.uuid4()}.png" # 預設儲存為 PNG
            preview_path = os.path.join(temp_dir, unique_filename)
            pil_image_or_path.save(preview_path)
            logger.info(f"[FileUtils] Saved PIL image to preview path {preview_path}")
            return preview_path
        except Exception as e:
            logger.error(f"[FileUtils] Error saving PIL image to temp dir for preview: {e}", exc_info=True)
            return None
    else:
        logger.warning(f"[FileUtils] Invalid input type for prepare_preview_image: {type(pil_image_or_path)}")
        return None

def save_processed_image(pil_image, original_filename_or_base, output_dir, logger, new_suffix="_processed"):
    """
    儲存處理後的 PIL Image 物件到指定輸出目錄。

    Args:
        pil_image (Image.Image): 要儲存的 PIL Image 物件。
        original_filename_or_base (str): 原始檔名或用於生成新檔名的基礎字串。
        output_dir (str): 儲存的目標目錄。
        logger: 日誌記錄器實例。
        new_suffix (str, optional): 加到原始檔名（去除副檔名後）之後的新後綴。

    Returns:
        str | None: 儲存後的檔案路徑，或在失敗時返回 None。    """
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        base, ext = os.path.splitext(original_filename_or_base)
        if not ext: # 如果原始輸入沒有副檔名 (例如只是一個 base name)
            ext = ".png" # 預設為 .png
        
        # 檢查並修正不支持的檔案擴展名
        supported_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}
        if ext.lower() not in supported_extensions:
            logger.warning(f"[FileUtils] Unsupported file extension '{ext}', using .png instead")
            ext = ".png"
        
        # 移除路徑部分，只取檔名
        base = os.path.basename(base)

        # 避免潛在的路徑遍歷字元 (雖然 os.path.basename 應該已經處理了大部分)
        base = base.replace("..", "_").replace("/", "_").replace("\\", "_")

        new_filename = f"{base}{new_suffix}{ext}"
        save_path = os.path.join(output_dir, new_filename)
        
        # 確保檔名不衝突，如果衝突則添加數字後綴
        counter = 1
        temp_save_path = save_path
        while os.path.exists(temp_save_path):
            temp_save_path = os.path.join(output_dir, f"{base}{new_suffix}_{counter}{ext}")
            counter += 1
        save_path = temp_save_path

        pil_image.save(save_path)
        logger.info(f"[FileUtils] Saved processed image to {save_path}")
        return save_path
    except Exception as e:
        logger.error(f"[FileUtils] Error saving processed image {original_filename_or_base} to {output_dir}: {e}", exc_info=True)
        return None

def handle_input_path(input_path, logger, temp_dir=None):
    """
    處理輸入路徑，可以是本地檔案或 URL。
    如果是 URL，下載圖片到臨時目錄。
    如果是本地路徑，檢查其有效性。
    返回 PIL Image 物件。

    Args:
        input_path (str): 圖片的本地路徑或 URL。
        logger: 日誌記錄器實例。
        temp_dir (str, optional): 下載 URL 或處理檔案時的臨時目錄。預設從 settings 獲取。

    Returns:
        Image.Image | None: PIL Image 物件，或在失敗時返回 None。
    """
    if temp_dir is None:
        temp_dir = getattr(settings, 'TEMP_DIR', os.path.join(settings.BASE_DIR, 'temp_processing'))
        os.makedirs(temp_dir, exist_ok=True)

    logger.info(f"[FileUtils] Handling input path: {input_path}")
    
    try:
        # 檢查是否為 URL
        parsed_url = urlparse(input_path)
        if parsed_url.scheme in ['http', 'https']:
            logger.info(f"[FileUtils] Input is a URL: {input_path}. Downloading...")
            response = requests.get(input_path, stream=True, timeout=settings.URL_DOWNLOAD_TIMEOUT if hasattr(settings, 'URL_DOWNLOAD_TIMEOUT') else 10)
            response.raise_for_status() # 如果請求失敗則拋出 HTTPError
            
            # 從 URL 或 Content-Disposition 推斷檔名和副檔名
            content_disposition = response.headers.get('content-disposition')
            filename = None
            if content_disposition:
                import cgi
                value, params = cgi.parse_header(content_disposition)
                if 'filename' in params:
                    filename = params['filename']
            
            if not filename: # 如果無法從 header 取得檔名
                filename_from_url = os.path.basename(parsed_url.path)
                if not filename_from_url or '.' not in filename_from_url: # 如果 URL path 也沒有檔名或副檔名
                    # 嘗試從 content-type 推斷副檔名
                    content_type = response.headers.get('content-type')
                    ext = '.jpg' # 預設
                    if content_type:
                        import mimetypes
                        ext_from_mime = mimetypes.guess_extension(content_type.split(';')[0].strip())
                        if ext_from_mime:
                            ext = ext_from_mime
                    filename = f"{uuid.uuid4()}{ext}"
                else:
                    filename = filename_from_url

            temp_file_path = os.path.join(temp_dir, f"download_{uuid.uuid4()}_{os.path.basename(filename)}")
            
            with open(temp_file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"[FileUtils] Downloaded URL content to temporary file: {temp_file_path}")
            input_path_to_open = temp_file_path # 後續使用下載的檔案路徑
        elif os.path.isfile(input_path):
            logger.info(f"[FileUtils] Input is a local file path: {input_path}")
            # TODO: 可以增加對本地路徑的白名單或安全檢查
            # 例如： if not is_path_allowed(input_path, settings.ALLOWED_INPUT_PATHS):
            #           logger.error(f"Path not allowed: {input_path}")
            #           return None
            input_path_to_open = input_path
        else:
            logger.error(f"[FileUtils] Input path is not a valid file or URL: {input_path}")
            return None

        # 打開圖片
        img = Image.open(input_path_to_open)
        # 確保圖片被完全載入，避免 "cannot identify image file" 等問題，尤其對於網路下載的檔案
        img.load() 
        logger.info(f"[FileUtils] Successfully opened image from: {input_path_to_open}")
        
        # 如果是從 URL 下載的臨時檔案，可以考慮在返回 Image 物件後刪除，
        # 但如果後續操作仍需此路徑，則不應刪除。
        # 目前設計是返回 PIL Image，所以臨時檔案可以刪除。
        if parsed_url.scheme in ['http', 'https'] and os.path.exists(input_path_to_open):
             try:
                 # os.remove(input_path_to_open)
                 # logger.debug(f"[FileUtils] Removed temporary downloaded file: {input_path_to_open}")
                 # 暫時不刪除，因為 prepare_preview_image 可能會再次用到這個路徑
                 # 如果 prepare_preview_image 總是複製，那這裡就可以刪除
                 pass
             except Exception as e_remove:
                 logger.warning(f"[FileUtils] Could not remove temporary downloaded file {input_path_to_open}: {e_remove}")
        
        return img

    except requests.exceptions.RequestException as e:
        logger.error(f"[FileUtils] Failed to download image from URL {input_path}: {e}", exc_info=True)
        return None
    except FileNotFoundError:
        logger.error(f"[FileUtils] Image file not found at path: {input_path}", exc_info=True)
        return None
    except IOError as e: # PIL.UnidentifiedImageError 繼承自 IOError
        logger.error(f"[FileUtils] Could not open or read image file at {input_path_to_open if 'input_path_to_open' in locals() else input_path}. Error: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"[FileUtils] An unexpected error occurred while handling input path {input_path}: {e}", exc_info=True)
        return None

# 可以在 settings.py 中定義 ALLOWED_PREVIEW_PATHS
# 例如： ALLOWED_PREVIEW_PATHS = [os.path.abspath(settings.STATIC_DIR), os.path.abspath(settings.GRADIO_TEMP_DIR)]
def is_path_safe_for_gradio(path, allowed_paths=None):
    """
    檢查給定路徑是否在 Gradio 允許的預覽路徑下。
    """
    if allowed_paths is None:
        # 預設允許 GRADIO_TEMP_DIR 和 STATIC_DIR
        gradio_temp = getattr(settings, 'GRADIO_TEMP_DIR', os.path.join(settings.BASE_DIR, 'temp_previews'))
        static_dir = getattr(settings, 'STATIC_DIR', os.path.join(settings.BASE_DIR, 'static'))
        allowed_paths = [os.path.abspath(gradio_temp), os.path.abspath(static_dir)]
        
    abs_path = os.path.abspath(path)
    for allowed in allowed_paths:
        if abs_path.startswith(allowed):
            return True
    return False

def scan_directory_for_images(directory_path, recursive=True, supported_extensions=None):
    """
    掃描目錄中的所有圖片文件，支援遞歸掃描子目錄。
    
    Args:
        directory_path (str): 要掃描的目錄路徑
        recursive (bool): 是否遞歸掃描子目錄
        supported_extensions (list): 支援的圖片副檔名列表
        
    Returns:
        list: 找到的圖片文件路徑列表
    """
    if supported_extensions is None:
        supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']
    
    image_files = []
    
    if not os.path.isdir(directory_path):
        return image_files
    
    try:
        if recursive:
            # 遞歸掃描所有子目錄
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in supported_extensions):
                        image_files.append(os.path.join(root, file))
        else:
            # 只掃描當前目錄
            for file in os.listdir(directory_path):
                file_path = os.path.join(directory_path, file)
                if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in supported_extensions):
                    image_files.append(file_path)
                    
        # 按文件名排序以確保處理順序一致
        image_files.sort()
        
    except Exception as e:
        print(f"Error scanning directory {directory_path}: {e}")
        
    return image_files

def create_output_structure(input_path, output_base_dir, preserve_structure=True):
    """
    基於輸入路徑創建輸出目錄結構。
    
    Args:
        input_path (str): 輸入文件路徑
        output_base_dir (str): 輸出基礎目錄
        preserve_structure (bool): 是否保持原有目錄結構
        
    Returns:
        str: 輸出文件應該保存的目錄路徑
    """
    if not preserve_structure:
        # 所有文件都輸出到基礎目錄
        return output_base_dir
    
    # 保持目錄結構
    input_dir = os.path.dirname(input_path)
    relative_dir = os.path.relpath(input_dir, start=os.path.dirname(input_path))
    
    if relative_dir == '.':
        return output_base_dir
    else:
        output_dir = os.path.join(output_base_dir, relative_dir)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

def get_relative_output_path(input_file_path, input_base_dir, output_base_dir):
    """
    根據輸入文件路徑和基礎目錄，計算相對應的輸出路徑。
    保持原有的目錄結構。
    
    Args:
        input_file_path (str): 輸入文件的完整路徑
        input_base_dir (str): 輸入文件的基礎目錄
        output_base_dir (str): 輸出文件的基礎目錄
        
    Returns:
        str: 輸出文件應該保存的目錄路徑
    """
    # 計算相對於輸入基礎目錄的相對路徑
    try:
        relative_path = os.path.relpath(os.path.dirname(input_file_path), input_base_dir)
        
        if relative_path == '.':
            # 文件就在基礎目錄中
            output_dir = output_base_dir
        else:
            # 文件在子目錄中，保持相同的子目錄結構
            output_dir = os.path.join(output_base_dir, relative_path)
            
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
        
    except Exception as e:
        print(f"Error calculating output path for {input_file_path}: {e}")
        return output_base_dir

def safe_move_file(source_path, target_path, logger, overwrite=True):
    """
    安全地移動檔案，處理目標檔案已存在的情況。
    
    Args:
        source_path (str): 來源檔案路徑
        target_path (str): 目標檔案路徑
        logger: 日誌記錄器
        overwrite (bool): 是否覆蓋已存在的檔案
        
    Returns:
        str: 最終的檔案路徑，如果失敗則返回 None
    """
    try:
        # 確保目標目錄存在
        target_dir = os.path.dirname(target_path)
        os.makedirs(target_dir, exist_ok=True)
        
        # 如果目標檔案已存在
        if os.path.exists(target_path):
            if overwrite:
                # 覆蓋模式：先刪除目標檔案
                try:
                    os.remove(target_path)
                    logger.info(f"[FileUtils] Removed existing file: {target_path}")
                except Exception as e:
                    logger.warning(f"[FileUtils] Failed to remove existing file {target_path}: {e}")
            else:
                # 不覆蓋模式：生成新的檔名
                base, ext = os.path.splitext(target_path)
                counter = 1
                while os.path.exists(target_path):
                    target_path = f"{base}_{counter}{ext}"
                    counter += 1
                logger.info(f"[FileUtils] Target file exists, using new name: {target_path}")
        
        # 執行移動操作
        if os.path.exists(source_path):
            os.rename(source_path, target_path)
            logger.info(f"[FileUtils] Successfully moved file: {source_path} -> {target_path}")
            return target_path
        else:
            logger.error(f"[FileUtils] Source file not found: {source_path}")
            return None
            
    except Exception as e:
        logger.error(f"[FileUtils] Error moving file from {source_path} to {target_path}: {e}", exc_info=True)
        return None
