import os
import shutil
from imgutils.metrics import lpips_clustering
from tqdm import tqdm
from logger_config import get_logger
from error_handler import safe_execute, DirectoryError, ImageProcessingError, ModelError, WaifucError

# 設定日誌記錄器
logger = get_logger('lpips_clustering')

def batch_generator(lst, batch_size):
    """生成器，對文件路徑列表進行批次處理"""
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

def safe_lpips_clustering(file_paths_batch):
    """
    安全執行 LPIPS 聚類
    
    Args:
        file_paths_batch: 圖像檔案路徑列表
    
    Returns:
        list: LPIPS 聚類結果，失敗時返回空列表
    
    Raises:
        ModelError: 當 LPIPS 模型執行失敗時
        ImageProcessingError: 當圖像處理失敗時
    """
    try:
        result = lpips_clustering(file_paths_batch)
        logger.debug(f"LPIPS 聚類完成，處理了 {len(file_paths_batch)} 個檔案")
        return result
    except Exception as e:
        error_msg = f"LPIPS 聚類失敗: {str(e)}"
        if "model" in str(e).lower() or "cuda" in str(e).lower() or "memory" in str(e).lower():
            raise ModelError(error_msg, "lpips")
        else:
            raise ImageProcessingError(error_msg)

def ensure_group_directory(output_directory, group_id):
    """
    確保群組目錄存在
    
    Args:
        output_directory (str): 基礎輸出目錄
        group_id (int): 群組 ID
    
    Returns:
        str: 群組目錄路徑
    
    Raises:
        DirectoryError: 當無法創建目錄時
    """
    group_directory = os.path.join(output_directory, f"group_{group_id}")
    try:
        os.makedirs(group_directory, exist_ok=True)
        return group_directory
    except PermissionError:
        raise DirectoryError(f"權限不足，無法創建群組目錄 '{group_directory}'", group_directory)
    except Exception as e:
        raise DirectoryError(f"創建群組目錄 '{group_directory}' 時發生錯誤: {str(e)}", group_directory)

def move_file_to_group(source_file, group_directory):
    """
    將檔案移動到群組目錄
    
    Args:
        source_file (str): 來源檔案路徑
        group_directory (str): 目標群組目錄
    
    Returns:
        bool: 移動是否成功
    """
    try:
        destination_file = os.path.join(group_directory, os.path.basename(source_file))
        shutil.move(source_file, destination_file)
        logger.debug(f"移動檔案 {source_file} 到 {destination_file}")
        return True
    except Exception as e:
        logger.error(f"移動檔案 {source_file} 失敗: {str(e)}")
        return False

def process_lpips_clustering(file_paths, output_directory="lpips_output", batch_size=100):
    """
    根據LPIPS聚類結果處理圖像文件
    
    Parameters:
    - file_paths: 圖像文件路徑列表
    - output_directory: 輸出目錄
    - batch_size: 批處理大小
    
    Returns:
    - output_directory: 聚類結果的輸出目錄
    
    Raises:
        DirectoryError: 當無法創建輸出目錄時
        WaifucError: 當處理過程中發生其他錯誤時
    """
    logger.info(f"開始 LPIPS 聚類: 檔案數={len(file_paths)}, 輸出目錄={output_directory}, 批次大小={batch_size}")
    
    # 確保輸出目錄存在
    try:
        os.makedirs(output_directory, exist_ok=True)
    except PermissionError:
        raise DirectoryError(f"權限不足，無法創建輸出目錄 '{output_directory}'", output_directory)
    except Exception as e:
        raise DirectoryError(f"創建輸出目錄 '{output_directory}' 時發生錯誤: {str(e)}", output_directory)
    
    lpips_counts = {}
    total_moved = 0
    
    # 添加進度條以跟踪處理過程
    with tqdm(total=len(file_paths), desc="LPIPS聚類進度") as pbar:
        for batch in batch_generator(file_paths, batch_size):
            # 使用 safe_execute 安全執行 LPIPS 聚類
            lpips_batch = safe_execute(
                safe_lpips_clustering,
                batch,
                logger=logger,
                default_return=[],
                error_msg_prefix=f"處理批次 ({len(batch)} 個檔案) 時"
            )
            
            if not lpips_batch:
                logger.warning(f"批次處理失敗，跳過 {len(batch)} 個檔案")
                pbar.update(len(batch))
                continue
            
            # 處理批次結果
            for source_file, lpips_result in zip(batch, lpips_batch):
                try:
                    # 確保 lpips_result 不是 -1（噪音樣本）
                    if lpips_result != -1:
                        # 使用 get 方法獲取 lpips_result 對應的計數
                        lpips_count = lpips_counts.get(lpips_result, 0)

                        # 如果 lpips_result 已經出現過1次或更多，移動檔案
                        if lpips_count > 0:
                            # 創建群組目錄
                            group_directory = safe_execute(
                                ensure_group_directory,
                                output_directory,
                                lpips_result,
                                logger=logger,
                                default_return=None,
                                error_msg_prefix=f"創建群組目錄 group_{lpips_result} 時"
                            )
                            
                            if group_directory:
                                # 移動檔案
                                move_success = safe_execute(
                                    move_file_to_group,
                                    source_file,
                                    group_directory,
                                    logger=logger,
                                    default_return=False,
                                    error_msg_prefix=f"移動檔案 {os.path.basename(source_file)} 時"
                                )
                                
                                if move_success:
                                    total_moved += 1

                        # 更新 lpips_counts 字典中 lpips_result 的計數
                        lpips_counts[lpips_result] = lpips_count + 1
                    
                except Exception as e:
                    logger.error(f"處理檔案 {source_file} 時發生未預期的錯誤: {str(e)}")
                
                # 更新進度條
                pbar.update(1)

    logger.info(f"LPIPS 聚類完成，共移動 {total_moved} 個檔案到群組目錄")
    print("檔案已根據 LPIPS 結果分類並放入各自的資料夾中。")
    return output_directory

if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    
    # 先讀取.env檔案的設定
    load_dotenv()
    
    # 從環境變數獲取預設值
    default_directory = os.getenv("directory", "")
    default_output = os.getenv("lpips_output_directory", "lpips_output")
    default_batch_size = int(os.getenv("lpips_batch_size", 100))
    
    parser = argparse.ArgumentParser(description='使用LPIPS進行圖像聚類')
    parser.add_argument('directory', type=str, nargs='?', default=default_directory, 
                       help='圖像文件目錄 (預設: 從.env讀取)')
    parser.add_argument('--output', type=str, default=default_output, 
                       help=f'輸出目錄 (預設: {default_output})')
    parser.add_argument('--batch-size', type=int, default=default_batch_size, 
                       help=f'處理批次大小 (預設: {default_batch_size})')
    
    args = parser.parse_args()
    
    # 確認目錄存在
    if not args.directory:
        logger.error("未指定輸入目錄")
        print("錯誤: 未指定輸入目錄")
        exit(1)
    
    if not os.path.isdir(args.directory):
        logger.error(f"指定的目錄 '{args.directory}' 不存在或不是有效目錄")
        print(f"錯誤: 指定的目錄 '{args.directory}' 不存在或不是有效目錄")
        exit(1)
    
    # 取得資料夾內所有檔案路徑，只處理圖片檔案
    try:
        all_files = os.listdir(args.directory)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
        file_paths = []
        
        for f in all_files:
            file_path = os.path.join(args.directory, f)
            if os.path.isfile(file_path):
                ext = os.path.splitext(f)[-1].lower()
                if ext in image_extensions:
                    file_paths.append(file_path)
        
    except PermissionError:
        logger.error(f"權限不足，無法讀取目錄 '{args.directory}'")
        print(f"錯誤: 權限不足，無法讀取目錄 '{args.directory}'")
        exit(1)
    except Exception as e:
        logger.error(f"讀取目錄時發生錯誤: {str(e)}")
        print(f"錯誤: 讀取目錄時發生錯誤: {str(e)}")
        exit(1)
    
    if not file_paths:
        logger.warning(f"在目錄 '{args.directory}' 中未找到圖片檔案")
        print(f"警告: 在目錄 '{args.directory}' 中未找到圖片檔案")
        exit(0)
    
    logger.info(f"找到 {len(file_paths)} 個圖片檔案，開始進行LPIPS聚類...")
    print(f"找到 {len(file_paths)} 個圖片檔案，開始進行LPIPS聚類...")
    print(f"輸出目錄: {args.output}, 批次大小: {args.batch_size}")
    
    # 使用 safe_execute 安全執行LPIPS聚類處理
    output_dir = safe_execute(
        process_lpips_clustering,
        file_paths,
        args.output,
        args.batch_size,
        logger=logger,
        default_return=None,
        error_msg_prefix="執行 LPIPS 聚類時"
    )
    
    if output_dir:
        logger.info(f"處理完成，結果已儲存至: {output_dir}")
        print(f"處理完成，結果已儲存至: {output_dir}")
    else:
        logger.error("LPIPS 聚類處理失敗")
        print("LPIPS 聚類處理失敗")
