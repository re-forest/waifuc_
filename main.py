import os
import sys
import shutil
from pathlib import Path
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
from transparency import scan_directory, convert_transparent_to_white

def run_full_pipeline(config: dict) -> dict:
    """執行完整的圖片處理流程
    
    Args:
        config: 包含所有設定的字典
        
    Returns:
        dict: 包含執行日誌和結果摘要的字典
    """
    import io
    import contextlib
    import csv
    
    # 捕捉所有輸出到字串
    log_buffer = io.StringIO()
    results = {
        'success': True,
        'logs': '',
        'error': None,
        'summary': {}
    }
    
    try:
        with contextlib.redirect_stdout(log_buffer), contextlib.redirect_stderr(log_buffer):
            directory_str = config.get("directory")
            if not directory_str:
                raise ValueError("錯誤: 未設定目錄路徑")

            directory = Path(directory_str)
            if not directory.exists():
                raise ValueError(f"錯誤: 目錄 '{directory}' 不存在")

            if not directory.is_dir():
                raise ValueError(f"錯誤: '{directory}' 不是有效目錄")

            print(f"開始處理目錄: {directory}")

            # 驗證圖片完整性
            if config.get("enable_validation", True):
                print("------- 開始驗證圖片完整性 -------")
                validate_and_remove_invalid_images(str(directory))
            
            # 透明通道處理
            if config.get("enable_transparency_processing", True):
                print("------- 開始處理透明通道 -------")
                scan_results = scan_directory(str(directory))
                transparent_count = sum(1 for r in scan_results if r['has_transparency'])
                total_count = len(scan_results)
                
                print(f"圖片總數: {total_count}")
                print(f"包含透明層的圖片: {transparent_count}")
                
                if transparent_count > 0:
                    print(f"發現 {transparent_count} 張含有透明通道的圖片，開始轉換...")
                    converted_count = 0
                    
                    for result in scan_results:
                        if result['has_transparency']:
                            if convert_transparent_to_white(result['file_path']):
                                converted_count += 1
                                result['converted_to_white'] = True
                            else:
                                result['converted_to_white'] = False
                        else:
                            result['converted_to_white'] = False
                    
                    print(f"已成功轉換 {converted_count} 張圖片的透明背景為白色")
                    
                    # 將結果保存到CSV文件
                    csv_path = directory / "transparency_results.csv"
                    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                        fieldnames = ['file_path', 'has_transparency', 'converted_to_white']
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        for result in scan_results:
                            writer.writerow(result)
                    print(f"處理結果已保存至: {csv_path}")
                else:
                    print("未發現含有透明通道的圖片")
            
            # 人臉檢測
            if config.get("enable_face_detection", True):
                print("------- 開始人臉檢測 -------")
                min_face_count = config.get("min_face_count", 1)
                face_output_directory = Path(config.get("face_output_directory", "face_out"))
                detect_faces_in_directory(str(directory), min_face_count, str(face_output_directory))
            
            # 重新取得資料夾內所有檔案路徑
            file_paths = [str(file_path) for file_path in directory.iterdir() if file_path.is_file()]
            
            # 圖片去重複
            if config.get("enable_lpips_clustering", True):
                print("------- 開始圖片去重複 -------")
                lpips_output_directory = Path(config.get("lpips_output_directory", "lpips_output"))
                lpips_batch_size = config.get("lpips_batch_size", 100)
                
                print(f"圖片去重複設定 - 輸出目錄: {lpips_output_directory}, 批次大小: {lpips_batch_size}")
                lpips_result_dir = process_lpips_clustering(file_paths, str(lpips_output_directory), lpips_batch_size)
                print(f"圖片去重複處理完成，結果保存在: {lpips_result_dir}")
            
            # 裁切檔案
            if config.get("enable_cropping", True):
                print("------- 開始裁切圖片 -------")
                output_directory_str = config.get("output_directory")
                if not output_directory_str:
                    print("警告: 未設定 output_directory 環境變數，使用預設值")
                    output_directory = directory.parent / "cropped"
                else:
                    output_directory = Path(output_directory_str)
                process_single_folder(str(directory), str(output_directory))
            
            # 檔案分類
            if config.get("enable_classification", True):
                print("------- 開始分類圖片 -------")
                output_directory_str = config.get("output_directory")
                if not output_directory_str:
                    print("警告: 未設定 output_directory 環境變數，跳過分類步驟")
                else:
                    output_directory = Path(output_directory_str)
                    classify_files_in_directory(str(output_directory))
            
            # 放大圖片
            if config.get("enable_upscaling", False):
                print("------- 開始放大圖片 -------")
                output_directory_str = config.get("output_directory")
                if not output_directory_str:
                    print("警告: 未設定 output_directory 環境變數，跳過放大步驟")
                else:
                    output_directory = Path(output_directory_str)
                    target_width = config.get("upscale_target_width", 1024)
                    target_height = config.get("upscale_target_height", 1024)
                    upscale_model = config.get("upscale_model", "HGSR-MHR-anime-aug_X4_320")
                    min_size = config.get("upscale_min_size", 800)
                    
                    print(f"放大設定 - 目標尺寸: {target_width}x{target_height}, 模型: {upscale_model}")
                    if min_size:
                        print(f"只處理小於 {min_size}x{min_size} 的圖片")
                        
                    upscale_images_in_directory(
                        str(output_directory),
                        target_width=target_width,
                        target_height=target_height,
                        model=upscale_model,
                        overwrite=True,
                        min_size=min_size,
                        recursive=True
                    )
            
            # 標記圖片
            if config.get("enable_tagging", True):
                print("------- 開始標記圖片 -------")
                output_directory_str = config.get("output_directory")
                if not output_directory_str:
                    print("警告: 未設定 output_directory 環境變數，跳過標記步驟")
                else:
                    output_directory = Path(output_directory_str)
                    tag_image(str(output_directory))
            
            print("所有處理已完成!")
            
    except Exception as e:
        results['success'] = False
        results['error'] = str(e)
        print(f"執行過程中發生錯誤: {str(e)}")
    
    finally:
        results['logs'] = log_buffer.getvalue()
    
    return results


def main():
    """主程式執行流程"""
    load_dotenv()

    # 從環境變數建立設定字典
    config = {
        "directory": os.getenv("directory"),
        "enable_validation": os.getenv("enable_validation", "true").lower() == "true",
        "enable_transparency_processing": os.getenv("enable_transparency_processing", "true").lower() == "true",
        "enable_face_detection": os.getenv("enable_face_detection", "true").lower() == "true",
        "min_face_count": int(os.getenv("min_face_count", 1)),
        "face_output_directory": os.getenv("face_output_directory", "face_out"),
        "enable_lpips_clustering": os.getenv("enable_lpips_clustering", "true").lower() == "true",
        "lpips_output_directory": os.getenv("lpips_output_directory", "lpips_output"),
        "lpips_batch_size": int(os.getenv("lpips_batch_size", 100)),
        "enable_cropping": os.getenv("enable_cropping", "true").lower() == "true",
        "enable_classification": os.getenv("enable_classification", "true").lower() == "true",
        "enable_upscaling": os.getenv("enable_upscaling", "false").lower() == "true",
        "upscale_target_width": int(os.getenv("upscale_target_width", 1024)),
        "upscale_target_height": int(os.getenv("upscale_target_height", 1024)),
        "upscale_model": os.getenv("upscale_model", "HGSR-MHR-anime-aug_X4_320"),
        "upscale_min_size": int(os.getenv("upscale_min_size", 800)) if os.getenv("upscale_min_size") else None,
        "enable_tagging": os.getenv("enable_tagging", "true").lower() == "true",
        "output_directory": os.getenv("output_directory")
    }

    # 執行完整流程
    result = run_full_pipeline(config)
    
    if result['success']:
        print(result['logs'])
        return 0
    else:
        print(result['logs'])
        if result['error']:
            print(f"錯誤: {result['error']}")
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n程式被使用者中斷")
        sys.exit(1)
    except Exception as e:
        print(f"\n執行過程中發生錯誤: {str(e)}")
        sys.exit(1)
