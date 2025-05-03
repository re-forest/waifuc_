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
        
    # 獲取所有檔案
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    # 先創建所有需要的子目錄
    for folder in file_to_folder_map.values():
        target_directory = os.path.join(directory, folder)
        os.makedirs(target_directory, exist_ok=True)
    
    for file in tqdm(files, desc="分類檔案進度"):
        # 根據檔案名稱決定將其移動到哪個子目錄
        for key, folder in file_to_folder_map.items():
            if key in file:
                target_directory = os.path.join(directory, folder)
                
                try:
                    # 移動檔案到相應子目錄
                    shutil.move(os.path.join(directory, file), os.path.join(target_directory, file))
                except Exception as e:
                    print(f"移動檔案 {file} 時發生錯誤: {str(e)}")
                break  # 一旦找到匹配項，就退出內部迴圈


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
