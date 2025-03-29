import os
import sys
import shutil
from concurrent.futures import ProcessPoolExecutor
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from tqdm import tqdm

# 從本地模組導入功能
from validate_image import validate_and_remove_invalid_images
from face_detection import detect_faces_in_directory
from lpips_clustering import process_lpips_clustering
from crop import classify_files_in_directory, process_single_folder
from tag import tag_image
from upscale import upscale_images_in_directory

def main():
    """主程式執行流程"""
    # 從.env檔案中讀取設定
    load_dotenv()
    
    # 檔案路徑
    directory = os.getenv("directory")
    if not directory or not os.path.isdir(directory):
        print(f"錯誤: 目錄 '{directory}' 不存在或不是有效目錄")
        return 1

    print(f"開始處理目錄: {directory}")
    
    # 驗證圖片完整性
    if os.getenv("enable_validation", "true").lower() == "true":
        print("------- 開始驗證圖片完整性 -------")
        validate_and_remove_invalid_images(directory)
    
    # 人臉檢測
    if os.getenv("enable_face_detection", "true").lower() == "true":
        print("------- 開始人臉檢測 -------")
        min_face_count = int(os.getenv("min_face_count", 1))  # 預設最小人臉數量為 1
        face_output_directory = os.getenv("face_output_directory", "face_out")  # 從環境變數讀取輸出目錄
        detect_faces_in_directory(directory, min_face_count, face_output_directory)
    
    # 重新取得資料夾內所有檔案路徑 
    file_paths = [os.path.join(directory, f) for f in os.listdir(directory) 
                 if os.path.isfile(os.path.join(directory, f))]
    
    # 圖片去重複
    if os.getenv("enable_lpips_clustering", "true").lower() == "true":
        print("------- 開始圖片去重複 -------")
        lpips_output_directory = os.getenv("lpips_output_directory", "lpips_output")
        lpips_batch_size = int(os.getenv("lpips_batch_size", 100))
        
        print(f"圖片去重複設定 - 輸出目錄: {lpips_output_directory}, 批次大小: {lpips_batch_size}")
        lpips_result_dir = process_lpips_clustering(file_paths, lpips_output_directory, lpips_batch_size)
        print(f"圖片去重複處理完成，結果保存在: {lpips_result_dir}")
    
    # 裁切檔案
    if os.getenv("enable_cropping", "true").lower() == "true":
        print("------- 開始裁切圖片 -------")
        output_directory = os.getenv("output_directory")
        if not output_directory:
            print("警告: 未設定 output_directory 環境變數，使用預設值")
            output_directory = os.path.join(os.path.dirname(directory), "cropped")
        
        process_single_folder(directory, output_directory)
    
    # 檔案分類
    if os.getenv("enable_classification", "true").lower() == "true":
        print("------- 開始分類圖片 -------")
        output_directory = os.getenv("output_directory")
        if not output_directory:
            print("警告: 未設定 output_directory 環境變數，跳過分類步驟")
        else:
            classify_files_in_directory(output_directory)

    # 放大圖片
    if os.getenv("enable_upscaling", "true").lower() == "true":
        print("------- 開始放大圖片 -------")
        output_directory = os.getenv("output_directory")
        if not output_directory:
            print("警告: 未設定 output_directory 環境變數，跳過放大步驟")
        else:
            target_width = int(os.getenv("upscale_target_width", 1024))
            target_height = int(os.getenv("upscale_target_height", 1024))
            upscale_model = os.getenv("upscale_model", "HGSR-MHR-anime-aug_X4_320")
            min_size = int(os.getenv("upscale_min_size", 800)) if os.getenv("upscale_min_size") else None
            
            print(f"放大設定 - 目標尺寸: {target_width}x{target_height}, 模型: {upscale_model}")
            if min_size:
                print(f"只處理小於 {min_size}x{min_size} 的圖片")
                
            upscale_images_in_directory(
                output_directory,
                target_width=target_width,
                target_height=target_height,
                model=upscale_model,
                overwrite=True,
                min_size=min_size,
                recursive=True
            )
    
    # 標記圖片
    if os.getenv("enable_tagging", "true").lower() == "true":
        print("------- 開始標記圖片 -------")
        output_directory = os.getenv("output_directory")
        if not output_directory:
            print("警告: 未設定 output_directory 環境變數，跳過標記步驟")
        else:
            tag_image(output_directory)
    
    print("所有處理已完成!")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n程式被使用者中斷")
        sys.exit(1)
    except Exception as e:
        print(f"\n執行過程中發生錯誤: {str(e)}")
        sys.exit(1)
