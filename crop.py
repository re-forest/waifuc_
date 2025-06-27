import argparse
import os
import shutil
from waifuc.action import ThreeStageSplitAction
from waifuc.export import SaveExporter
from waifuc.source import LocalSource
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog

# 處理單一資料夾的圖片
def process_single_folder(input_path, output_path):
    print(f"目前正在處理：{input_path}")  # Debug用，顯示目前正在處理的資料夾
    
    source = LocalSource(input_path)
    source.attach(
        ThreeStageSplitAction(),
    ).export(SaveExporter(output_path))
    
    classify_files_in_directory(output_path)

def classify_files_in_directory(directory):
    # 用於映射檔案類型到其子目錄
    file_to_folder_map = {
        "_person1_head": "head",
        "_person1_halfbody": "halfbody",
        "_person1": "person",
    }
    
    # 確保目錄存在
    if not os.path.exists(directory):
        print(f"錯誤: 目錄不存在 {directory}")
        return
        
    # 獲取所有檔案，過濾掉 JSON 檔案
    all_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    files = [f for f in all_files if not f.lower().endswith('.json')]
    
    # 刪除所有 JSON 檔案
    json_files = [f for f in all_files if f.lower().endswith('.json')]
    for json_file in json_files:
        try:
            os.remove(os.path.join(directory, json_file))
            print(f"已刪除 JSON 檔案: {json_file}")
        except Exception as e:
            print(f"刪除 JSON 檔案 {json_file} 時發生錯誤: {str(e)}")

    # 先創建所有需要的子目錄，包括 other
    all_folders = list(file_to_folder_map.values()) + ["other"]
    for folder in all_folders:
        target_directory = os.path.join(directory, folder)
        os.makedirs(target_directory, exist_ok=True)    
    for file in tqdm(files, desc="分類檔案進度"):
        # 標記是否找到匹配的分類
        classified = False
        
        # 根據檔案名稱決定將其移動到哪個子目錄
        for key, folder in file_to_folder_map.items():
            if key in file:
                target_directory = os.path.join(directory, folder)
                
                try:
                    # 移動檔案到相應子目錄
                    shutil.move(os.path.join(directory, file), os.path.join(target_directory, file))
                    classified = True
                except Exception as e:
                    print(f"移動檔案 {file} 時發生錯誤: {str(e)}")
                break  # 一旦找到匹配項，就退出內部迴圈
        
        # 如果沒有找到匹配的分類，移動到 other 資料夾
        if not classified:
            other_directory = os.path.join(directory, "other")
            try:
                shutil.move(os.path.join(directory, file), os.path.join(other_directory, file))
                print(f"檔案 {file} 移動到 other 資料夾")
            except Exception as e:
                print(f"移動檔案 {file} 到 other 資料夾時發生錯誤: {str(e)}")

def process_cropping_and_classification(input_dir, output_dir):
    """
    處理圖片裁剪和分類並返回詳細摘要
    
    Parameters:
    - input_dir: 輸入目錄路徑
    - output_dir: 輸出目錄路徑
    
    Returns:
    - dict: 包含處理結果摘要的字典
    """
    logs = []
    try:
        # 驗證輸入目錄
        if not input_dir or not os.path.exists(input_dir):
            return {
                "success": False,
                "error": f"輸入目錄不存在: {input_dir}",
                "logs": [f"輸入目錄不存在: {input_dir}"],
                "input_directory": input_dir,
                "output_directory": output_dir,
                "total_files": 0,
                "processed_files": 0,
                "categories": {}
            }
        
        # 確保輸出目錄存在
        if not output_dir:
            return {
                "success": False,
                "error": "未指定輸出目錄",
                "logs": ["未指定輸出目錄"],
                "input_directory": input_dir,
                "output_directory": output_dir,
                "total_files": 0,
                "processed_files": 0,
                "categories": {}
            }
        
        os.makedirs(output_dir, exist_ok=True)
        logs.append(f"開始處理目錄: {input_dir}")
        logs.append(f"輸出目錄: {output_dir}")
        
        # 統計輸入文件
        input_files = []
        for file in os.listdir(input_dir):
            if os.path.isfile(os.path.join(input_dir, file)):
                input_files.append(file)
        
        logs.append(f"找到 {len(input_files)} 個輸入文件")
        
        # 執行處理
        process_single_folder(input_dir, output_dir)
        
        # 統計處理結果
        categories = {
            "head": 0,
            "halfbody": 0,
            "person": 0,
            "other": 0
        }
        
        total_processed = 0
          # 檢查輸出目錄中的分類結果
        if os.path.exists(output_dir):
            # 統計各類別的文件數，包括 other
            for category in ["head", "halfbody", "person", "other"]:
                category_dir = os.path.join(output_dir, category)
                if os.path.exists(category_dir):
                    category_files = [f for f in os.listdir(category_dir) if os.path.isfile(os.path.join(category_dir, f))]
                    categories[category] = len(category_files)
                    total_processed += len(category_files)
                    logs.append(f"{category} 類別: {len(category_files)} 個文件")
            
            # 統計直接在輸出目錄中的文件（應該沒有，但為了安全起見）
            root_files = []
            for file in os.listdir(output_dir):
                file_path = os.path.join(output_dir, file)
                if os.path.isfile(file_path):
                    root_files.append(file)
            
            if root_files:
                logs.append(f"根目錄中剩餘文件: {len(root_files)} 個")
                # 這些文件沒有被正確分類，加到 other 統計中
                categories["other"] += len(root_files)
        
        logs.append(f"處理完成，共處理 {total_processed} 個文件")
        
        return {
            "success": True,
            "error": None,
            "logs": logs,
            "input_directory": input_dir,
            "output_directory": output_dir,
            "total_files": len(input_files),
            "processed_files": total_processed,
            "categories": categories
        }
        
    except Exception as e:
        error_msg = f"裁剪和分類處理失敗: {str(e)}"
        logs.append(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "logs": logs,
            "input_directory": input_dir,
            "output_directory": output_dir,
            "total_files": 0,
            "processed_files": 0,
            "categories": {}
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="處理圖片的腳本")
    parser.add_argument("--input_path", required=False, help="包含圖片的資料夾")
    parser.add_argument("--output_path", required=False, help="儲存輸出圖片的資料夾")
    parser.add_argument("--include_subfolders", action='store_true', help="是否包括子資料夾")
    
    args = parser.parse_args()

    # 若未帶入路徑參數，彈出視窗選擇資料夾
    if args.input_path is None:
        root = tk.Tk()
        root.withdraw()
        args.input_path = filedialog.askdirectory(title="請選擇輸入資料夾")
    if args.output_path is None:
        root = tk.Tk()
        root.withdraw()
        args.output_path = filedialog.askdirectory(title="請選擇輸出資料夾")

    # 確保輸出資料夾存在
    os.makedirs(args.output_path, exist_ok=True)

    # 處理主資料夾
    process_single_folder(args.input_path, args.output_path)

    # 如果需要包括子資料夾
    print(f"是否包括子資料夾：{args.include_subfolders}")
    if args.include_subfolders:
        for subdir, _, _ in os.walk(args.input_path):
            rel_path = os.path.relpath(subdir, args.input_path)  # 獲取相對路徑
            if rel_path == '.':  # 跳過根目錄
                continue
            new_input_path = os.path.join(args.input_path, rel_path)
            new_output_path = os.path.join(args.output_path, rel_path)
            os.makedirs(new_output_path, exist_ok=True)  # 創建輸出的子資料夾
            
            # 處理子資料夾
            process_single_folder(new_input_path, new_output_path)
