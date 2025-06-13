import argparse
import os
import shutil
from waifuc.action import ThreeStageSplitAction
from waifuc.export import SaveExporter
from waifuc.source import LocalSource
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog
from logger_config import get_logger
from error_handler import safe_execute, DirectoryError, ImageProcessingError, ModelError, WaifucError

# 設定日誌記錄器
logger = get_logger('crop')

def safe_waifuc_process(input_path, output_path):
    """
    安全執行 waifuc 圖片處理
    
    Args:
        input_path (str): 輸入目錄路徑
        output_path (str): 輸出目錄路徑
    
    Raises:
        DirectoryError: 當目錄操作失敗時
        ImageProcessingError: 當圖片處理失敗時
        ModelError: 當模型執行失敗時
    """
    try:
        # 確保輸出目錄存在
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
        
        source = LocalSource(input_path)
        source.attach(
            ThreeStageSplitAction(),
        ).export(SaveExporter(output_path))
        
        logger.debug(f"waifuc 處理完成: {input_path} -> {output_path}")
        
    except PermissionError:
        raise DirectoryError(f"權限不足，無法存取輸出目錄 '{output_path}'", output_path)
    except FileNotFoundError:
        raise DirectoryError(f"輸入目錄 '{input_path}' 不存在", input_path)
    except Exception as e:
        error_msg = f"waifuc 處理失敗: {str(e)}"
        if "model" in str(e).lower() or "download" in str(e).lower():
            raise ModelError(error_msg, "waifuc_crop")
        elif "image" in str(e).lower() or "format" in str(e).lower():
            raise ImageProcessingError(error_msg)
        else:
            raise WaifucError(error_msg)

def safe_move_file(source_path, target_path):
    """
    安全移動檔案
    
    Args:
        source_path (str): 來源檔案路徑
        target_path (str): 目標檔案路徑
    
    Returns:
        bool: 移動是否成功
    """
    try:
        shutil.move(source_path, target_path)
        logger.debug(f"移動檔案: {source_path} -> {target_path}")
        return True
    except PermissionError:
        logger.error(f"權限不足，無法移動檔案 {source_path}")
        return False
    except FileNotFoundError:
        logger.error(f"檔案不存在: {source_path}")
        return False
    except Exception as e:
        logger.error(f"移動檔案 {source_path} 時發生錯誤: {str(e)}")
        return False

# 處理單一資料夾的圖片
def process_single_folder(input_path, output_path):
    """
    處理單一資料夾的圖片
    
    Args:
        input_path (str): 輸入目錄路徑
        output_path (str): 輸出目錄路徑
    
    Raises:
        DirectoryError: 當目錄操作失敗時
        WaifucError: 當處理失敗時
    """
    logger.info(f"開始處理資料夾：{input_path} -> {output_path}")
    
    # 驗證輸入目錄
    if not os.path.isdir(input_path):
        raise DirectoryError(f"輸入目錄 '{input_path}' 不存在或不是有效目錄", input_path)
    
    if not os.access(input_path, os.R_OK):
        raise DirectoryError(f"無法讀取輸入目錄 '{input_path}'，請檢查權限", input_path)
    
    # 使用 safe_execute 安全執行 waifuc 處理
    safe_execute(
        safe_waifuc_process,
        input_path,
        output_path,
        logger=logger,
        error_msg_prefix=f"處理資料夾 {input_path} 時"
    )
    
    # 執行檔案分類
    safe_execute(
        classify_files_in_directory,
        output_path,
        logger=logger,
        error_msg_prefix=f"分類檔案 {output_path} 時"
    )

def classify_files_in_directory(directory):
    """
    在目錄中分類檔案到不同的子目錄
    
    Args:
        directory (str): 要分類的目錄路徑
    
    Raises:
        DirectoryError: 當目錄操作失敗時
        WaifucError: 當分類過程失敗時
    """
    logger.info(f"開始分類檔案: {directory}")
    
    # 用於映射檔案類型到其子目錄
    file_to_folder_map = {
        "_person1_head": "head",
        "_person1_halfbody": "halfbody",
        "_person1": "person",
    }
    
    # 確保目錄存在
    if not os.path.exists(directory):
        raise DirectoryError(f"目錄不存在 {directory}", directory)
    
    if not os.access(directory, os.R_OK):
        raise DirectoryError(f"無法讀取目錄 '{directory}'，請檢查權限", directory)
    
    try:
        # 獲取所有檔案
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    except PermissionError:
        raise DirectoryError(f"權限不足，無法列舉目錄 '{directory}' 中的檔案", directory)
    except Exception as e:
        raise WaifucError(f"列舉目錄檔案時發生錯誤: {str(e)}")

    # 先創建所有需要的子目錄
    for folder in file_to_folder_map.values():
        target_directory = os.path.join(directory, folder)
        try:
            os.makedirs(target_directory, exist_ok=True)
        except PermissionError:
            raise DirectoryError(f"權限不足，無法創建目錄 '{target_directory}'", target_directory)
        except Exception as e:
            raise DirectoryError(f"創建目錄 '{target_directory}' 時發生錯誤: {str(e)}", target_directory)
    
    moved_count = 0
    
    for file in tqdm(files, desc="分類檔案進度"):
        # 根據檔案名稱決定將其移動到哪個子目錄
        for key, folder in file_to_folder_map.items():
            if key in file:
                source_path = os.path.join(directory, file)
                target_directory = os.path.join(directory, folder)
                target_path = os.path.join(target_directory, file)
                
                # 使用 safe_execute 安全移動檔案
                move_success = safe_execute(
                    safe_move_file,
                    source_path,
                    target_path,
                    logger=logger,
                    default_return=False,
                    error_msg_prefix=f"移動檔案 {file} 時"
                )
                
                if move_success:
                    moved_count += 1
                
                break  # 一旦找到匹配項，就退出內部迴圈
    
    logger.info(f"檔案分類完成，共移動 {moved_count} 個檔案")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="處理圖片的腳本")
    parser.add_argument("--input_path", required=False, help="包含圖片的資料夾")
    parser.add_argument("--output_path", required=False, help="儲存輸出圖片的資料夾")
    parser.add_argument("--include_subfolders", action='store_true', help="是否包括子資料夾")
    
    args = parser.parse_args()

    # 若未帶入路徑參數，彈出視窗選擇資料夾
    try:
        if args.input_path is None:
            root = tk.Tk()
            root.withdraw()
            args.input_path = filedialog.askdirectory(title="請選擇輸入資料夾")
        if args.output_path is None:
            root = tk.Tk()
            root.withdraw()
            args.output_path = filedialog.askdirectory(title="請選擇輸出資料夾")
    except Exception as e:
        logger.error(f"GUI 選擇目錄時發生錯誤: {str(e)}")
        print(f"錯誤: GUI 選擇目錄時發生錯誤: {str(e)}")
        exit(1)

    if not args.input_path or not args.output_path:
        logger.error("未選擇輸入或輸出目錄")
        print("錯誤: 未選擇輸入或輸出目錄")
        exit(1)

    # 確保輸出資料夾存在
    try:
        os.makedirs(args.output_path, exist_ok=True)
    except PermissionError:
        logger.error(f"權限不足，無法創建輸出目錄 '{args.output_path}'")
        print(f"錯誤: 權限不足，無法創建輸出目錄 '{args.output_path}'")
        exit(1)
    except Exception as e:
        logger.error(f"創建輸出目錄時發生錯誤: {str(e)}")
        print(f"錯誤: 創建輸出目錄時發生錯誤: {str(e)}")
        exit(1)

    # 使用 safe_execute 安全處理主資料夾
    logger.info(f"開始處理主資料夾: {args.input_path}")
    main_result = safe_execute(
        process_single_folder,
        args.input_path,
        args.output_path,
        logger=logger,
        default_return=False,
        error_msg_prefix="處理主資料夾時"
    )

    if not main_result:
        logger.error("主資料夾處理失敗")
        print("主資料夾處理失敗")

    # 如果需要包括子資料夾
    logger.info(f"是否包括子資料夾：{args.include_subfolders}")
    print(f"是否包括子資料夾：{args.include_subfolders}")
    
    if args.include_subfolders:
        try:
            for subdir, _, _ in os.walk(args.input_path):
                rel_path = os.path.relpath(subdir, args.input_path)  # 獲取相對路徑
                if rel_path == '.':  # 跳過根目錄
                    continue
                
                new_input_path = os.path.join(args.input_path, rel_path)
                new_output_path = os.path.join(args.output_path, rel_path)
                
                # 安全創建輸出子目錄
                create_result = safe_execute(
                    os.makedirs,
                    new_output_path,
                    exist_ok=True,
                    logger=logger,
                    default_return=False,
                    error_msg_prefix=f"創建子目錄 {new_output_path} 時"
                )
                
                if create_result is False:
                    logger.warning(f"跳過子目錄 {rel_path}，創建失敗")
                    continue
                
                # 安全處理子資料夾
                sub_result = safe_execute(
                    process_single_folder,
                    new_input_path,
                    new_output_path,
                    logger=logger,
                    default_return=False,
                    error_msg_prefix=f"處理子資料夾 {rel_path} 時"
                )
                
                if not sub_result:
                    logger.warning(f"子資料夾 {rel_path} 處理失敗")
                    
        except Exception as e:
            logger.error(f"處理子資料夾時發生未預期的錯誤: {str(e)}")
            print(f"處理子資料夾時發生錯誤: {str(e)}")

    logger.info("圖片裁切與分類處理完成")
    print("圖片裁切與分類處理完成")
