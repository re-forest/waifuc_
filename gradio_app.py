import os
import json
import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askdirectory
from dotenv import load_dotenv
import gradio as gr
import time
import asyncio
from tqdm import tqdm

# 配置 Gradio 以減少資源載入問題
os.environ['GRADIO_ANALYTICS_ENABLED'] = '0'
os.environ['GRADIO_DEBUG'] = '0'

# 導入日誌系統
from logger_config import get_logger

from validate_image import validate_and_remove_invalid_images
from face_detection import detect_faces_in_directory
from lpips_clustering import process_lpips_clustering
from crop import process_single_folder, classify_files_in_directory
from tag import tag_image
from upscale import upscale_images_in_directory

# 載入環境變數
load_dotenv()

# 初始化日誌記錄器
logger = get_logger('gradio_app')

# 從環境變數讀取預設值，如果沒有則使用合理預設值
DEFAULT_VALUES = {
    'directory': os.getenv('directory', ''),
    'pipeline_main_output_root': os.getenv('pipeline_main_output_root', ''),
    'face_output_directory': os.getenv('face_output_directory', 'face_out'),
    'min_face_count': int(os.getenv('min_face_count', 1)),
    'lpips_output_directory': os.getenv('lpips_output_directory', 'lpips_output'),
    'lpips_batch_size': int(os.getenv('lpips_batch_size', 100)),
    'output_directory': os.getenv('output_directory', 'output'),
    'upscale_target_width': int(os.getenv('upscale_target_width', 1024)),
    'upscale_target_height': int(os.getenv('upscale_target_height', 1024)),
    'upscale_model': os.getenv('upscale_model', 'HGSR-MHR-anime-aug_X4_320'),
    'upscale_min_size': int(os.getenv('upscale_min_size', 800)),
    'enable_validation': os.getenv('enable_validation', 'true').lower() == 'true',
    'enable_face_detection': os.getenv('enable_face_detection', 'true').lower() == 'true',
    'enable_lpips_clustering': os.getenv('enable_lpips_clustering', 'true').lower() == 'true',
    'enable_cropping': os.getenv('enable_cropping', 'true').lower() == 'true',
    'enable_classification': os.getenv('enable_classification', 'true').lower() == 'true',
    'enable_upscaling': os.getenv('enable_upscaling', 'true').lower() == 'true',
    'enable_tagging': os.getenv('enable_tagging', 'true').lower() == 'true',
}

# 設置額外的環境變數來抑制不必要的資源請求
os.environ['GRADIO_SHARE'] = '0'
os.environ['GRADIO_QUIET'] = '1'


def run_validation(directory, progress=gr.Progress()):
    """驗證圖像完整性並顯示進度"""
    try:
        logger.info(f"開始圖像驗證 - 目錄: {directory}")
        progress(0, desc="開始驗證圖像...")
        processed, removed = validate_and_remove_invalid_images(directory)
        progress(1, desc="驗證完成")
        result = f"已處理 {processed} 張圖片，刪除 {removed} 張無效圖片"
        logger.info(f"圖像驗證完成 - {result}")
        return result
    except Exception as e:
        error_msg = f"圖像驗證過程中發生錯誤: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return error_msg


def run_face_detection(directory, min_face, output_dir, progress=gr.Progress()):
    """人臉檢測並顯示進度"""
    try:
        logger.info(f"開始人臉檢測 - 目錄: {directory}, 最小人臉數: {min_face}, 輸出: {output_dir}")
        progress(0, desc="開始人臉檢測...")
        time.sleep(0.1)  # 小延遲讓進度條可見
        processed, moved = detect_faces_in_directory(directory, min_face, output_dir)
        progress(1, desc="人臉檢測完成")
          # 獲取預覽圖像（從基於來源的子目錄中）
        preview_images = []
        source_dir_name = os.path.basename(directory.rstrip(os.sep))
        source_output_dir = os.path.join(output_dir, source_dir_name)
        
        if os.path.exists(source_output_dir):
            # 從各個 faces_* 子目錄中收集預覽圖像
            for item in os.listdir(source_output_dir):
                if item.startswith("faces_"):
                    subdir_path = os.path.join(source_output_dir, item)
                    if os.path.isdir(subdir_path):
                        image_files = [f for f in os.listdir(subdir_path) 
                                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))][:2]
                        preview_images.extend([os.path.join(subdir_path, f) for f in image_files])
                        if len(preview_images) >= 4:
                            break
        
        result = f"已檢測 {processed} 張圖片，移動 {moved} 張圖片"
        logger.info(f"人臉檢測完成 - {result}")
        return result, preview_images
    except Exception as e:
        error_msg = f"人臉檢測過程中發生錯誤: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return error_msg, []


def run_lpips(directory, output_dir, batch_size, progress=gr.Progress()):
    """LPIPS去重並顯示進度"""
    try:
        if not os.path.isdir(directory):
            error_msg = "指定目錄不存在"
            logger.warning(f"LPIPS 去重失敗 - {error_msg}: {directory}")
            return error_msg
        
        logger.info(f"開始 LPIPS 去重 - 目錄: {directory}, 輸出: {output_dir}, 批次大小: {batch_size}")
        progress(0, desc="準備處理 LPIPS 去重...")
        
        # 智能檢測目錄結構並收集圖片檔案
        file_paths = []
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
        
        # 檢查是否為 face_out 相關的目錄結構
        if os.path.basename(directory) == "face_out" or "face_out" in directory:
            # 如果是 face_out 目錄，智能處理子目錄結構
            for root, dirs, files in os.walk(directory):
                # 優先處理 faces_* 目錄中的檔案
                for f in files:
                    if any(f.lower().endswith(ext) for ext in image_extensions):
                        file_paths.append(os.path.join(root, f))
        else:
            # 一般目錄，遞歸收集所有圖片檔案
            for root, dirs, files in os.walk(directory):
                for f in files:
                    if any(f.lower().endswith(ext) for ext in image_extensions):
                        file_paths.append(os.path.join(root, f))
        
        logger.debug(f"發現 {len(file_paths)} 個檔案")
        progress(0.2, desc=f"發現 {len(file_paths)} 張圖片，開始處理...")
        
        # 為輸出目錄添加來源標識（如果適用）
        if "face_out" in directory:
            # 嘗試提取來源目錄名稱
            parts = directory.split(os.sep)
            if "face_out" in parts:
                face_out_index = parts.index("face_out")
                if face_out_index + 1 < len(parts):
                    source_name = parts[face_out_index + 1]
                    output_dir = f"{output_dir}_{source_name}"
        
        result_dir = process_lpips_clustering(file_paths, output_dir, batch_size)
        progress(1, desc="LPIPS 去重完成")
        result = f"處理完成，結果位於 {result_dir}"
        logger.info(f"LPIPS 去重完成 - {result}")
        return result
    except Exception as e:
        error_msg = f"LPIPS 去重過程中發生錯誤: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return error_msg


def run_crop(input_path, output_path, progress=gr.Progress()):
    """圖像裁切並顯示進度"""
    try:
        logger.info(f"開始圖像裁切 - 輸入: {input_path}, 輸出: {output_path}")
        progress(0, desc="開始圖像裁切...")
        process_single_folder(input_path, output_path)
        progress(1, desc="圖像裁切完成")
        
        # 獲取裁切後的預覽圖像
        preview_images = []
        if os.path.exists(output_path):
            for subdir in ['head', 'halfbody', 'fullbody']:  # 常見的裁切類別
                subdir_path = os.path.join(output_path, subdir)
                if os.path.exists(subdir_path):
                    image_files = [f for f in os.listdir(subdir_path) 
                                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))][:2]
                    preview_images.extend([os.path.join(subdir_path, f) for f in image_files])
                    if len(preview_images) >= 4:  # 最多顯示4張
                        break
        
        logger.info("圖像裁切完成")
        return "裁切完成", preview_images[:4]
    except Exception as e:
        error_msg = f"圖像裁切過程中發生錯誤: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return error_msg, []


def run_classify(directory, progress=gr.Progress()):
    """圖像分類並顯示進度"""
    progress(0, desc="開始圖像分類...")
    classify_files_in_directory(directory)
    progress(1, desc="圖像分類完成")
    return "分類完成"


def run_tag(directory, progress=gr.Progress()):
    """圖像標記並顯示進度"""
    progress(0, desc="開始圖像標記...")
    count = tag_image(directory)
    progress(1, desc="圖像標記完成")
    return f"標記完成，共處理 {count} 張圖片"


def run_upscale(directory, width, height, model, min_size, overwrite, recursive, progress=gr.Progress()):
    """圖像放大並顯示進度"""
    progress(0, desc="開始圖像放大處理...")
    time.sleep(0.1)  # 小延遲讓進度條可見
    total, upscaled = upscale_images_in_directory(
        directory,
        target_width=width,
        target_height=height,
        model=model,
        min_size=min_size,
        overwrite=overwrite,
        recursive=recursive,
    )
    progress(1, desc="圖像放大完成")
    
    # 獲取放大後的預覽圖像
    preview_images = []
    if os.path.exists(directory):
        if recursive:
            for root, dirs, files in os.walk(directory):
                image_files = [f for f in files 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))][:4]
                preview_images.extend([os.path.join(root, f) for f in image_files])
                if len(preview_images) >= 4:
                    break
        else:
            image_files = [f for f in os.listdir(directory) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))][:4]
            preview_images = [os.path.join(directory, f) for f in image_files]
    
    return f"處理 {total} 張圖片，放大 {upscaled} 張", preview_images[:4]


def run_integrated_pipeline(
    directory, 
    pipeline_main_output_root,
    enable_validation, enable_face_detection, enable_lpips, enable_cropping, 
    enable_classification, enable_upscaling, enable_tagging,
    face_output_dir, min_face_count, lpips_output_dir, lpips_batch_size,
    crop_output_dir, upscale_width, upscale_height, upscale_model, upscale_min_size,
    progress=gr.Progress()
):
    """執行整合式處理流程"""
    import time
    start_time = time.time()
    
    results = []
    step = 0
    total_steps = sum([enable_validation, enable_face_detection, enable_lpips, 
                      enable_cropping, enable_classification, enable_upscaling, enable_tagging])
    
    current_dir = directory
    source_dir_name = os.path.basename(directory.rstrip(os.sep))
    
    # 決定基礎輸出路徑
    if pipeline_main_output_root and pipeline_main_output_root.strip():
        base_output_path = os.path.join(pipeline_main_output_root.strip(), source_dir_name)
        logger.info(f"整合流程將使用主要輸出根目錄: {pipeline_main_output_root.strip()}")
        logger.info(f"所有步驟的輸出將位於: {base_output_path} 下")
    else:
        # 如果未提供主要輸出根目錄，則各步驟輸出路徑基於其獨立設定和來源資料夾名稱
        base_output_path = None
        logger.info("未指定主要輸出根目錄，各步驟將使用獨立的輸出目錄設定。")
    
    try:
        # 驗證配置
        is_valid, error_msg = validate_pipeline_config(pipeline_main_output_root, directory)
        if not is_valid:
            logger.error(f"配置驗證失敗: {error_msg}")
            return f"❌ 配置錯誤: {error_msg}", []
        
        # 確保基礎輸出路徑存在 (如果設定了 pipeline_main_output_root)
        if base_output_path:
            os.makedirs(base_output_path, exist_ok=True)
          # 1. 圖片驗證
        if enable_validation:
            step += 1
            status_msg = get_processing_status_message("圖片驗證", step, total_steps)
            progress(step/total_steps, desc=status_msg)
            processed, removed = validate_and_remove_invalid_images(current_dir)
            results.append(f"✅ 圖片驗證: 處理 {processed} 張，刪除 {removed} 張無效圖片")
        
        # 2. 人臉檢測
        if enable_face_detection:
            step += 1
            status_msg = get_processing_status_message("人臉檢測", step, total_steps, f"最小人臉數: {min_face_count}")
            progress(step/total_steps, desc=status_msg)
            actual_face_output_dir = get_step_output_path(base_output_path, face_output_dir, source_dir_name, "人臉檢測")
            
            processed, moved = detect_faces_in_directory(current_dir, min_face_count, actual_face_output_dir)
            results.append(f"✅ 人臉檢測: 檢測 {processed} 張，移動 {moved} 張符合條件的圖片到 {actual_face_output_dir}")
            
            # 更新當前工作目錄為實際的人臉檢測輸出目錄 (包含 source_dir_name 的那層)
            # detect_faces_in_directory 內部會創建 source_dir_name 子目錄
            current_dir = os.path.join(actual_face_output_dir, source_dir_name)
          # 3. LPIPS 去重
        if enable_lpips:
            step += 1
            status_msg = get_processing_status_message("LPIPS去重", step, total_steps, f"批次大小: {lpips_batch_size}")
            progress(step/total_steps, desc=status_msg)            # 根據最小人臉數量收集符合條件的圖片
            file_paths = []
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
            # 查找所有 faces_* 目錄，只處理符合最小人臉數量要求的
            if os.path.exists(current_dir):
                for item in os.listdir(current_dir):
                    if item.startswith("faces_"):
                        try:
                            face_count = int(item.split("_")[1])
                            # 只處理符合最小人臉數量要求的目錄
                            if face_count >= min_face_count:
                                dir_path = os.path.join(current_dir, item)
                                if os.path.isdir(dir_path):
                                    for f in os.listdir(dir_path):
                                        if any(f.lower().endswith(ext) for ext in image_extensions):
                                            file_paths.append(os.path.join(dir_path, f))
                        except (ValueError, IndexError):
                            continue
            actual_lpips_output_dir = get_step_output_path(base_output_path, lpips_output_dir, source_dir_name, "LPIPS去重")

            if not file_paths:
                results.append(f"⚠️ LPIPS 去重: 在 {current_dir} 中未找到符合條件的圖片進行處理。")
                logger.warning(f"LPIPS: No files found in {current_dir} for min_face_count {min_face_count}")
            else:
                result_dir = process_lpips_clustering(file_paths, actual_lpips_output_dir, lpips_batch_size)
                results.append(f"✅ LPIPS 去重: 處理完成，結果保存在 {result_dir}")
                current_dir = result_dir
            step += 1
        
        # 4. 圖像裁切
        if enable_cropping:
            step += 1
            status_msg = get_processing_status_message("圖像裁切", step, total_steps)
            progress(step/total_steps, desc=status_msg)
            actual_crop_output_dir = get_step_output_path(base_output_path, crop_output_dir, source_dir_name, "圖像裁切")
            process_single_folder(current_dir, actual_crop_output_dir)
            results.append(f"✅ 圖像裁切: 完成，結果保存在 {actual_crop_output_dir}")
            current_dir = actual_crop_output_dir  # 更新當前工作目錄
            step += 1
        
        # 5. 圖像分類
        if enable_classification:
            progress(step/total_steps, desc="正在進行圖像分類...")
            classify_files_in_directory(current_dir)
            results.append(f"✅ 圖像分類: 完成")
            step += 1
        
        # 6. 圖像放大
        if enable_upscaling:
            progress(step/total_steps, desc="正在進行圖像放大...")
            total, upscaled = upscale_images_in_directory(
                current_dir,
                target_width=upscale_width,
                target_height=upscale_height,
                model=upscale_model,
                min_size=upscale_min_size,
                overwrite=True,
                recursive=True
            )
            results.append(f"✅ 圖像放大: 處理 {total} 張，放大 {upscaled} 張")
            step += 1
        
        # 7. 圖像標記
        if enable_tagging:
            progress(step/total_steps, desc="正在進行圖像標記...")
            count = tag_image(current_dir)
            results.append(f"✅ 圖像標記: 完成，共處理 {count} 張圖片")
            step += 1
        
        progress(1, desc="整合處理流程完成!")
        
        # 計算總處理時間
        total_time = time.time() - start_time
        
        # 獲取最終結果預覽
        preview_images = []
        if os.path.exists(current_dir):
            for root, dirs, files in os.walk(current_dir):
                image_files = [f for f in files 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))][:6]
                preview_images.extend([os.path.join(root, f) for f in image_files])
                if len(preview_images) >= 6:
                    break
        
        # 使用新的格式化函數
        final_result = format_pipeline_result(results, current_dir, total_time)
        return final_result, preview_images[:6]
        
    except Exception as e:
        error_msg = f"❌ 處理過程中發生錯誤: {str(e)}"
        results.append(error_msg)
        logger.error(error_msg)  # 記錄錯誤日誌
        return "\n".join(results), []


async def save_pipeline_config(*args):
    """儲存當前流程配置到 JSON 檔案"""
    try:
        file_path = await tk_asksaveasfilename_async(
            title="儲存整合流程配置",
            defaultext=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not file_path:
            return "儲存已取消"
        
        # 收集所有配置參數 (按照 UI inputs 的順序)
        config_data = {
            'pipeline_dir': args[0] if len(args) > 0 else '',
            'pipeline_main_output_root': args[1] if len(args) > 1 else '',
            'enable_validation': args[2] if len(args) > 2 else True,
            'enable_face_detection': args[3] if len(args) > 3 else True,
            'enable_lpips': args[4] if len(args) > 4 else True,
            'enable_cropping': args[5] if len(args) > 5 else True,
            'enable_classification': args[6] if len(args) > 6 else True,
            'enable_upscaling': args[7] if len(args) > 7 else True,
            'enable_tagging': args[8] if len(args) > 8 else True,
            'pipeline_face_output': args[9] if len(args) > 9 else 'face_out',
            'pipeline_min_face': args[10] if len(args) > 10 else 1,
            'pipeline_lpips_output': args[11] if len(args) > 11 else 'lpips_output',
            'pipeline_lpips_batch': args[12] if len(args) > 12 else 100,
            'pipeline_crop_output': args[13] if len(args) > 13 else 'output',
            'pipeline_upscale_w': args[14] if len(args) > 14 else 1024,
            'pipeline_upscale_h': args[15] if len(args) > 15 else 1024,
            'pipeline_upscale_model': args[16] if len(args) > 16 else 'HGSR-MHR-anime-aug_X4_320',
            'pipeline_upscale_min': args[17] if len(args) > 17 else 800,
        }
        
        # 儲存配置到檔案
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"流程配置已儲存到: {file_path}")
        return f"✅ 配置已成功儲存到: {os.path.basename(file_path)}"
        
    except Exception as e:
        error_msg = f"❌ 儲存配置失敗: {str(e)}"
        logger.error(f"儲存配置錯誤: {e}", exc_info=True)
        return error_msg


async def load_pipeline_config():
    """從 JSON 檔案載入流程配置"""
    try:
        file_path = await tk_askopenfilename_async(
            title="載入整合流程配置",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not file_path:
            # 返回空更新，保持現有值
            return tuple([gr.update() for _ in range(18)] + ["載入已取消"])
        
        # 讀取配置檔案
        with open(file_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        # 按照 UI 元件的順序返回配置值
        updates = (
            gr.update(value=config_data.get('pipeline_dir', '')),
            gr.update(value=config_data.get('pipeline_main_output_root', '')),
            gr.update(value=config_data.get('enable_validation', True)),
            gr.update(value=config_data.get('enable_face_detection', True)),
            gr.update(value=config_data.get('enable_lpips', True)),
            gr.update(value=config_data.get('enable_cropping', True)),
            gr.update(value=config_data.get('enable_classification', True)),
            gr.update(value=config_data.get('enable_upscaling', True)),
            gr.update(value=config_data.get('enable_tagging', True)),
            gr.update(value=config_data.get('pipeline_face_output', 'face_out')),
            gr.update(value=config_data.get('pipeline_min_face', 1)),
            gr.update(value=config_data.get('pipeline_lpips_output', 'lpips_output')),
            gr.update(value=config_data.get('pipeline_lpips_batch', 100)),
            gr.update(value=config_data.get('pipeline_crop_output', 'output')),
            gr.update(value=config_data.get('pipeline_upscale_w', 1024)),
            gr.update(value=config_data.get('pipeline_upscale_h', 1024)),
            gr.update(value=config_data.get('pipeline_upscale_model', 'HGSR-MHR-anime-aug_X4_320')),
            gr.update(value=config_data.get('pipeline_upscale_min', 800)),
            f"✅ 配置已成功載入: {os.path.basename(file_path)}"
        )
        
        logger.info(f"流程配置已載入: {file_path}")
        return updates
        
    except Exception as e:
        error_msg = f"❌ 載入配置失敗: {str(e)}"
        logger.error(f"載入配置錯誤: {e}", exc_info=True)
        # 返回空更新加錯誤訊息
        return tuple([gr.update() for _ in range(18)] + [error_msg])


def create_default_config_templates():
    """創建預設配置模板"""
    templates = {
        "標準處理流程": {
            'pipeline_dir': '',
            'pipeline_main_output_root': '',
            'enable_validation': True,
            'enable_face_detection': True,
            'enable_lpips': True,
            'enable_cropping': True,
            'enable_classification': True,
            'enable_upscaling': False,
            'enable_tagging': True,
            'pipeline_face_output': 'face_detected',
            'pipeline_min_face': 1,
            'pipeline_lpips_output': 'lpips_unique',
            'pipeline_lpips_batch': 100,
            'pipeline_crop_output': 'cropped',
            'pipeline_upscale_w': 1024,
            'pipeline_upscale_h': 1024,
            'pipeline_upscale_model': 'HGSR-MHR-anime-aug_X4_320',
            'pipeline_upscale_min': 800,
        },
        "快速處理": {
            'pipeline_dir': '',
            'pipeline_main_output_root': '',
            'enable_validation': True,
            'enable_face_detection': True,
            'enable_lpips': False,
            'enable_cropping': True,
            'enable_classification': False,
            'enable_upscaling': False,
            'enable_tagging': False,
            'pipeline_face_output': 'face_detected',
            'pipeline_min_face': 1,
            'pipeline_lpips_output': 'lpips_unique',
            'pipeline_lpips_batch': 50,
            'pipeline_crop_output': 'cropped',
            'pipeline_upscale_w': 512,
            'pipeline_upscale_h': 512,
            'pipeline_upscale_model': 'HGSR-MHR-anime-aug_X4_320',
            'pipeline_upscale_min': 400,
        },
        "高品質處理": {
            'pipeline_dir': '',
            'pipeline_main_output_root': '',
            'enable_validation': True,
            'enable_face_detection': True,
            'enable_lpips': True,
            'enable_cropping': True,
            'enable_classification': True,
            'enable_upscaling': True,
            'enable_tagging': True,
            'pipeline_face_output': 'face_detected',
            'pipeline_min_face': 2,
            'pipeline_lpips_output': 'lpips_unique',
            'pipeline_lpips_batch': 50,
            'pipeline_crop_output': 'cropped',
            'pipeline_upscale_w': 2048,
            'pipeline_upscale_h': 2048,
            'pipeline_upscale_model': 'HGSR-MHR-anime-aug_X4_320',
            'pipeline_upscale_min': 1024,
        }
    }
    return templates


def load_template_config(template_name):
    """載入指定的預設配置模板"""
    templates = create_default_config_templates()
    
    if template_name not in templates:
        return tuple([gr.update() for _ in range(18)] + [f"❌ 未找到模板: {template_name}"])
    
    config_data = templates[template_name]
    
    # 按照 UI 元件的順序返回配置值
    updates = (
        gr.update(value=config_data.get('pipeline_dir', '')),
        gr.update(value=config_data.get('pipeline_main_output_root', '')),
        gr.update(value=config_data.get('enable_validation', True)),
        gr.update(value=config_data.get('enable_face_detection', True)),
        gr.update(value=config_data.get('enable_lpips', True)),
        gr.update(value=config_data.get('enable_cropping', True)),
        gr.update(value=config_data.get('enable_classification', True)),
        gr.update(value=config_data.get('enable_upscaling', True)),
        gr.update(value=config_data.get('enable_tagging', True)),
        gr.update(value=config_data.get('pipeline_face_output', 'face_out')),
        gr.update(value=config_data.get('pipeline_min_face', 1)),
        gr.update(value=config_data.get('pipeline_lpips_output', 'lpips_output')),
        gr.update(value=config_data.get('pipeline_lpips_batch', 100)),
        gr.update(value=config_data.get('pipeline_crop_output', 'output')),
        gr.update(value=config_data.get('pipeline_upscale_w', 1024)),
        gr.update(value=config_data.get('pipeline_upscale_h', 1024)),
        gr.update(value=config_data.get('pipeline_upscale_model', 'HGSR-MHR-anime-aug_X4_320')),
        gr.update(value=config_data.get('pipeline_upscale_min', 800)),
        f"✅ 已載入預設配置: {template_name}"
    )
    
    logger.info(f"載入預設配置模板: {template_name}")
    return updates


def get_step_output_path(base_output_path, step_output_dir, source_dir_name, step_name):
    """
    根據是否設定主要輸出根目錄，決定各步驟的實際輸出路徑
    
    Args:
        base_output_path (str or None): 主要輸出根目錄，如果未設定則為 None
        step_output_dir (str): 步驟指定的輸出目錄名稱
        source_dir_name (str): 來源資料夾名稱
        step_name (str): 步驟名稱，用於日誌記錄
    
    Returns:
        str: 實際的輸出路徑
    """
    if base_output_path:
        # 如果有主要輸出根目錄，輸出路徑是 {base_output_path}/{step_output_dir}/
        actual_output_dir = os.path.join(base_output_path, step_output_dir)
        logger.info(f"{step_name}: 使用主要輸出根目錄結構 -> {actual_output_dir}")
    else:
        # 否則，輸出路徑是 {step_output_dir}_{source_dir_name}
        actual_output_dir = f"{step_output_dir}_{source_dir_name}"
        logger.info(f"{step_name}: 使用獨立輸出目錄結構 -> {actual_output_dir}")
    
    # 確保目錄存在
    os.makedirs(actual_output_dir, exist_ok=True)
    return actual_output_dir


def validate_pipeline_config(pipeline_main_output_root, directory):
    """
    驗證整合流程配置的有效性
    
    Args:
        pipeline_main_output_root (str): 主要輸出根目錄
        directory (str): 輸入目錄
    
    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        # 檢查輸入目錄
        if not directory or not directory.strip():
            return False, "請指定輸入圖片資料夾"
        
        input_path = directory.strip()
        if not os.path.exists(input_path):
            return False, f"輸入目錄不存在: {input_path}"
        
        if not os.path.isdir(input_path):
            return False, f"指定的路徑不是目錄: {input_path}"
        
        # 檢查主要輸出根目錄（如果有指定）
        if pipeline_main_output_root and pipeline_main_output_root.strip():
            output_root = pipeline_main_output_root.strip()
            parent_dir = os.path.dirname(output_root)
            
            # 如果父目錄不存在，嘗試創建
            if not os.path.exists(parent_dir):
                try:
                    os.makedirs(parent_dir, exist_ok=True)
                except PermissionError:
                    return False, f"無權限創建輸出目錄: {parent_dir}"
                except Exception as e:
                    return False, f"創建輸出目錄失敗: {str(e)}"
        
        return True, "配置有效"
        
    except Exception as e:
        return False, f"配置驗證時發生錯誤: {str(e)}"


def preview_output_structure(pipeline_main_output_root, source_dir, enabled_steps):
    """
    預覽輸出目錄結構
    
    Args:
        pipeline_main_output_root (str): 主要輸出根目錄
        source_dir (str): 來源目錄
        enabled_steps (dict): 啟用的處理步驟
    
    Returns:
        str: 輸出結構預覽
    """
    if not source_dir or not source_dir.strip():
        return "請先指定輸入圖片資料夾"
    
    source_dir_name = os.path.basename(source_dir.strip().rstrip(os.sep))
    
    preview_lines = ["📁 輸出目錄結構預覽:", ""]
    
    if pipeline_main_output_root and pipeline_main_output_root.strip():
        base_path = pipeline_main_output_root.strip()
        preview_lines.append(f"📂 {base_path}/")
        preview_lines.append(f"  └── 📂 {source_dir_name}/")
        
        step_dirs = []
        if enabled_steps.get('face_detection'):
            step_dirs.append("face_detected/")
        if enabled_steps.get('lpips'):
            step_dirs.append("lpips_unique/")
        if enabled_steps.get('cropping'):
            step_dirs.append("cropped/")
        
        for i, step_dir in enumerate(step_dirs):
            if i == len(step_dirs) - 1:
                preview_lines.append(f"      └── 📂 {step_dir}")
            else:
                preview_lines.append(f"      ├── 📂 {step_dir}")
    else:
        preview_lines.append("📂 專案目錄/")
        step_dirs = []
        if enabled_steps.get('face_detection'):
            step_dirs.append(f"face_detected_{source_dir_name}/")
        if enabled_steps.get('lpips'):
            step_dirs.append(f"lpips_unique_{source_dir_name}/")
        if enabled_steps.get('cropping'):
            step_dirs.append(f"cropped_{source_dir_name}/")
        
        for step_dir in step_dirs:
            preview_lines.append(f"  ├── 📂 {step_dir}")
    
    if not any(enabled_steps.values()):
        preview_lines.append("  (未選擇任何處理步驟)")
    
    return "\n".join(preview_lines)

def validate_path_realtime(path, path_type="input"):
    """
    即時驗證路徑的有效性並提供狀態反饋
    
    Args:
        path (str): 要驗證的路徑
        path_type (str): 路徑類型 ('input' 或 'output')
    
    Returns:
        str: 狀態訊息
    """
    if not path or not path.strip():
        if path_type == "input":
            return "⚠️ 請輸入路徑"
        else:
            return "✅ 使用預設設定"
    
    path = path.strip()
    
    try:
        if path_type == "input":
            if os.path.exists(path):
                if os.path.isdir(path):
                    # 檢查目錄中是否有圖像檔案
                    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
                    image_count = 0
                    for root, dirs, files in os.walk(path):
                        for file in files:
                            if any(file.lower().endswith(ext) for ext in image_extensions):
                                image_count += 1
                                if image_count >= 10:  # 只需檢查前10個檔案
                                    break
                        if image_count >= 10:
                            break
                    
                    if image_count > 0:
                        return f"✅ 有效 ({image_count}+ 張圖片)"
                    else:
                        return "⚠️ 無圖像檔案"
                else:
                    return "❌ 不是目錄"
            else:
                return "❌ 路徑不存在"
        
        else:  # output path
            if os.path.exists(path):
                if os.path.isdir(path):
                    if os.access(path, os.W_OK):
                        return "✅ 可寫入"
                    else:
                        return "❌ 無寫入權限"
                else:
                    return "❌ 不是目錄"
            else:
                # 檢查是否可以創建目錄
                parent_dir = os.path.dirname(path)
                if parent_dir and os.path.exists(parent_dir):
                    if os.access(parent_dir, os.W_OK):
                        return "✅ 可創建"
                    else:
                        return "❌ 無創建權限"
                else:
                    return "⚠️ 父目錄不存在"
    
    except Exception as e:
        return f"❌ 驗證錯誤: {str(e)[:20]}"


def tk_window_askdirectory(init_dir=None) -> str:
    """使用 tkinter 選擇資料夾"""
    window = tk.Tk()
    window.wm_attributes('-topmost', 1)
    window.withdraw()
    if init_dir is None:
        init_dir = os.getcwd()
    directory = askdirectory(initialdir=init_dir)
    window.destroy()  # 確保視窗被正確關閉
    return directory


async def tk_askdirectory_async(init_dir=None) -> str:
    """異步選擇資料夾"""
    directory = await asyncio.to_thread(tk_window_askdirectory, init_dir)
    return directory


async def browse_input_folder():
    """異步瀏覽輸入資料夾"""
    try:
        directory = await tk_askdirectory_async()
        if directory:
            logger.info(f"使用者選擇了輸入資料夾: {directory}")
            return directory
        else:
            return ""
    except Exception as e:
        error_msg = f"瀏覽資料夾時發生錯誤: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return ""


async def browse_output_folder():
    """異步瀏覽輸出資料夾"""
    try:
        directory = await tk_askdirectory_async()
        if directory:
            logger.info(f"使用者選擇了輸出資料夾: {directory}")
            return directory
        else:
            return ""
    except Exception as e:
        error_msg = f"瀏覽輸出資料夾時發生錯誤: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return ""


def format_pipeline_result(results, current_dir, total_time=None):
    """
    格式化整合處理流程的結果顯示
    
    Args:
        results (list): 各步驟的處理結果
        current_dir (str): 最終輸出目錄
        total_time (float): 總處理時間（秒）
    
    Returns:
        str: 格式化的結果字串
    """
    if not results:
        return "❌ 未執行任何處理步驟"
    
    # 標題
    result_lines = [
        "🎉 整合處理流程完成！",
        "=" * 50,
        ""
    ]
    
    # 各步驟結果
    result_lines.append("📋 處理步驟結果：")
    for i, result in enumerate(results, 1):
        result_lines.append(f"  {i}. {result}")
    
    result_lines.append("")
    
    # 最終輸出位置
    result_lines.append("📁 最終輸出位置：")
    result_lines.append(f"  {current_dir}")
    
    # 處理時間（如果提供）
    if total_time:
        result_lines.append("")
        result_lines.append(f"⏱️ 總處理時間: {total_time:.2f} 秒")
    
    # 使用提示
    result_lines.extend([
        "",
        "💡 提示：",
        "  • 可在右側預覽區查看處理結果",
        "  • 建議檢查輸出目錄中的檔案",
        "  • 可使用「儲存配置」保存此次處理設定"
    ])
    
    return "\n".join(result_lines)


def get_processing_status_message(step_name, current_step, total_steps, additional_info=""):
    """
    生成標準化的處理狀態訊息
    
    Args:
        step_name (str): 步驟名稱
        current_step (int): 當前步驟編號
        total_steps (int): 總步驟數
        additional_info (str): 額外資訊
    
    Returns:
        str: 格式化的狀態訊息
    """
    progress_bar = "█" * current_step + "░" * (total_steps - current_step)
    percentage = (current_step / total_steps) * 100
    
    base_msg = f"[{progress_bar}] {percentage:.0f}% - 正在執行: {step_name}"
    
    if additional_info:
        return f"{base_msg} ({additional_info})"
    
    return base_msg


def build_interface():
    # 配置 Gradio 主題和資源設定
    with gr.Blocks(
        title="Waifuc 圖像處理系統",
        css="""
        /* 自定義樣式，避免依賴外部字體和資源 */
        * {
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif !important;
        }
        body {
            font-family: system-ui, -apple-system, sans-serif !important;
            background-color: #f8f9fa;
        }
        .gradio-container {
            max-width: 1200px !important;
            font-family: system-ui, -apple-system, sans-serif !important;
        }
        /* 禁用外部字體載入 */
        @font-face { 
            font-display: none !important; 
        }
        /* 隱藏錯誤消息 */
        .error-message, .font-error {
            display: none !important;
        }
        """,
        head="""
        <meta name="robots" content="noindex, nofollow">
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
        /* 防止載入外部字體資源 */
        @import url() { display: none !important; }
        /* 禁用所有外部字體請求 */
        @font-face {
            font-family: 'ui-sans-serif';
            src: local('Arial'), local('Helvetica');
            font-display: swap;
        }
        @font-face {
            font-family: 'system-ui';
            src: local('Arial'), local('Helvetica');  
            font-display: swap;
        }        </style>
        """,
        analytics_enabled=False
    ) as demo:
        gr.Markdown("# 🎨 Waifuc 圖像處理介面")
        gr.Markdown("專為動漫風格圖像預處理設計的自動化工具")
          # 全域快速設定區域
        with gr.Accordion("🌐 全域路徑快速設定", open=False):
            gr.Markdown("💡 **提示**: 已實作資料夾瀏覽功能，每個 Tab 中的路徑輸入框旁都有 📁 瀏覽按鈕")
            gr.Markdown("🔧 **功能**: 使用各 Tab 中的瀏覽按鈕可快速選擇資料夾路徑，支援輸入和輸出目錄選擇")
        
        with gr.Tab("🚀 整合處理流程"):
            gr.Markdown("### 一鍵執行完整的圖像預處理流程")
            
            with gr.Row():
                with gr.Column():                    # 基本設定                    gr.Markdown("#### 📁 基本設定")
                    with gr.Row():
                        pipeline_dir = gr.Textbox(
                            label="主要圖片資料夾 (輸入)", 
                            value=DEFAULT_VALUES['directory'],
                            placeholder="請輸入包含原始圖像的資料夾路徑",
                            scale=3
                        )
                        pipeline_dir_browse = gr.Button(
                            "📁 瀏覽",
                            variant="secondary",
                            size="sm",
                            scale=1
                        )
                        pipeline_dir_status = gr.Textbox(
                            label="路徑狀態",
                            value="",
                            interactive=False,
                            scale=1
                        )
                    
                    with gr.Row():
                        # 新增：主要輸出根目錄選項
                        pipeline_main_output_root = gr.Textbox(
                            label="主要輸出根目錄 (可選)",
                            value=DEFAULT_VALUES['pipeline_main_output_root'],
                            placeholder="例如: D:\\processed_images 或 ./pipeline_outputs",
                            info="若指定此路徑，所有處理步驟的輸出將統一存放在此目錄下的子資料夾中",
                            scale=3
                        )
                        pipeline_output_browse = gr.Button(
                            "📁 瀏覽",
                            variant="secondary",
                            size="sm",
                            scale=1
                        )
                        pipeline_output_status = gr.Textbox(
                            label="輸出狀態",
                            value="",
                            interactive=False,
                            scale=1
                        )
                    
                    # 處理步驟選擇
                    gr.Markdown("#### ⚙️ 處理步驟選擇")
                    with gr.Row():
                        enable_validation = gr.Checkbox(
                            value=DEFAULT_VALUES['enable_validation'], 
                            label="圖片驗證"
                        )
                        enable_face_detection = gr.Checkbox(
                            value=DEFAULT_VALUES['enable_face_detection'], 
                            label="人臉檢測"
                        )
                        enable_lpips = gr.Checkbox(
                            value=DEFAULT_VALUES['enable_lpips_clustering'], 
                            label="LPIPS 去重"
                        )
                    with gr.Row():
                        enable_cropping = gr.Checkbox(
                            value=DEFAULT_VALUES['enable_cropping'], 
                            label="圖像裁切"
                        )
                        enable_classification = gr.Checkbox(
                            value=DEFAULT_VALUES['enable_classification'], 
                            label="圖像分類"
                        )
                        enable_upscaling = gr.Checkbox(
                            value=DEFAULT_VALUES['enable_upscaling'], 
                            label="圖像放大"
                        )
                        enable_tagging = gr.Checkbox(
                            value=DEFAULT_VALUES['enable_tagging'], 
                            label="圖像標記"                        )
                    
                    # 詳細參數設定
                    with gr.Accordion("🔧 詳細參數設定 (若未指定上方「主要輸出根目錄」，以下目錄將結合來源資料夾名稱；否則，它們將作為「主要輸出根目錄」下的子資料夾名稱)", open=False):
                        gr.Markdown("##### 人臉檢測參數")
                        pipeline_face_output = gr.Textbox(
                            value=DEFAULT_VALUES['face_output_directory'], 
                            label="人臉檢測輸出子目錄名",
                            info="例如: face_detected"
                        )
                        pipeline_min_face = gr.Number(
                            value=DEFAULT_VALUES['min_face_count'], 
                            label="最小人臉數量"
                        )
                        
                        gr.Markdown("##### LPIPS 去重參數")
                        pipeline_lpips_output = gr.Textbox(
                            value=DEFAULT_VALUES['lpips_output_directory'], 
                            label="LPIPS 輸出子目錄名",
                            info="例如: lpips_unique"
                        )
                        pipeline_lpips_batch = gr.Number(
                            value=DEFAULT_VALUES['lpips_batch_size'], 
                            label="LPIPS 批次大小"
                        )
                        
                        gr.Markdown("##### 裁切參數")
                        pipeline_crop_output = gr.Textbox(
                            value=DEFAULT_VALUES['output_directory'], 
                            label="裁切輸出子目錄名",
                            info="例如: cropped_images"
                        )
                        
                        gr.Markdown("##### 放大參數")
                        with gr.Row():
                            pipeline_upscale_w = gr.Number(
                                value=DEFAULT_VALUES['upscale_target_width'], 
                                label="目標寬度"
                            )
                            pipeline_upscale_h = gr.Number(
                                value=DEFAULT_VALUES['upscale_target_height'], 
                                label="目標高度"
                            )
                        pipeline_upscale_model = gr.Dropdown(
                            choices=[
                                "HGSR-MHR-anime-aug_X4_320",
                                "Real-ESRGAN_4x",
                                "Waifu2x"
                            ],
                            value=DEFAULT_VALUES['upscale_model'], 
                            label="放大模型"
                        )
                        pipeline_upscale_min = gr.Number(
                            value=DEFAULT_VALUES['upscale_min_size'], 
                            label="最小處理尺寸"
                        )
                    
                    # 輸出結構預覽
                    gr.Markdown("#### 📋 輸出結構預覽")
                    output_structure_preview = gr.Textbox(
                        label="預覽輸出目錄結構",
                        value="請先設定輸入路徑和處理步驟",
                        lines=8,
                        interactive=False
                    )
                    
                    # 配置管理按鈕
                    with gr.Row():
                        save_config_btn = gr.Button("💾 儲存配置", variant="secondary", size="sm")
                        load_config_btn = gr.Button("📂 載入配置", variant="secondary", size="sm")
                    
                    # 預設配置模板
                    gr.Markdown("#### 🎯 快速配置")
                    with gr.Row():
                        template_standard = gr.Button("標準處理", variant="secondary", size="sm")
                        template_fast = gr.Button("快速處理", variant="secondary", size="sm")
                        template_hq = gr.Button("高品質", variant="secondary", size="sm")
                    
                    config_status = gr.Textbox(
                        label="配置狀態",
                        value="",
                        lines=2,
                        interactive=False
                    )
                    
                    pipeline_btn = gr.Button("🚀 開始整合處理", variant="primary", size="lg")
                
                with gr.Column():
                    pipeline_out = gr.Textbox(label="處理進度與結果", lines=10)
                    pipeline_gallery = gr.Gallery(
                        label="最終結果預覽", 
                        show_label=True, 
                        elem_id="pipeline_gallery",                        columns=3, 
                        rows=2, 
                        height="auto"
                    )
            
            pipeline_btn.click(
                run_integrated_pipeline,
                inputs=[
                    pipeline_dir,
                    pipeline_main_output_root,
                    enable_validation, enable_face_detection, enable_lpips, enable_cropping,
                    enable_classification, enable_upscaling, enable_tagging,
                    pipeline_face_output, pipeline_min_face, pipeline_lpips_output, pipeline_lpips_batch,
                    pipeline_crop_output, pipeline_upscale_w, pipeline_upscale_h, 
                    pipeline_upscale_model, pipeline_upscale_min                ],
                outputs=[pipeline_out, pipeline_gallery]            )
            
            # 資料夾瀏覽按鈕事件處理器
            pipeline_dir_browse.click(
                fn=browse_input_folder,
                outputs=pipeline_dir
            )
            
            pipeline_output_browse.click(
                fn=browse_output_folder,
                outputs=pipeline_main_output_root
            )
            
            # 路徑即時驗證事件處理器
            pipeline_dir.change(
                fn=lambda path: validate_path_realtime(path, "input"),
                inputs=pipeline_dir,
                outputs=pipeline_dir_status
            )
            
            pipeline_main_output_root.change(
                fn=lambda path: validate_path_realtime(path, "output"),
                inputs=pipeline_main_output_root,
                outputs=pipeline_output_status
            )
            
            # 輸出結構預覽更新事件處理器
            def update_preview(input_dir, output_root, val, face, lpips, crop, classify, upscale, tag):
                enabled_steps = {
                    'validation': val,
                    'face_detection': face,
                    'lpips': lpips,
                    'cropping': crop,
                    'classification': classify,
                    'upscaling': upscale,
                    'tagging': tag
                }
                return preview_output_structure(output_root, input_dir, enabled_steps)
            
            # 當路徑或步驟選擇變化時更新預覽
            for component in [pipeline_dir, pipeline_main_output_root, enable_validation, 
                             enable_face_detection, enable_lpips, enable_cropping, 
                             enable_classification, enable_upscaling, enable_tagging]:
                component.change(
                    fn=update_preview,
                    inputs=[pipeline_dir, pipeline_main_output_root, enable_validation, 
                           enable_face_detection, enable_lpips, enable_cropping, 
                           enable_classification, enable_upscaling, enable_tagging],
                    outputs=output_structure_preview
                )
            
            # 配置管理事件處理器
            save_config_btn.click(
                fn=save_pipeline_config,
                inputs=[pipeline_dir, pipeline_main_output_root, enable_validation, 
                       enable_face_detection, enable_lpips, enable_cropping, 
                       enable_classification, enable_upscaling, enable_tagging,
                       pipeline_face_output, pipeline_min_face, pipeline_lpips_output, 
                       pipeline_lpips_batch, pipeline_crop_output, pipeline_upscale_w, 
                       pipeline_upscale_h, pipeline_upscale_model, pipeline_upscale_min],
                outputs=config_status
            )
            
            load_config_btn.click(
                fn=load_pipeline_config,
                outputs=[pipeline_dir, pipeline_main_output_root, enable_validation, 
                        enable_face_detection, enable_lpips, enable_cropping, 
                        enable_classification, enable_upscaling, enable_tagging,
                        pipeline_face_output, pipeline_min_face, pipeline_lpips_output, 
                        pipeline_lpips_batch, pipeline_crop_output, pipeline_upscale_w, 
                        pipeline_upscale_h, pipeline_upscale_model, pipeline_upscale_min, 
                        config_status]
            )
            
            # 預設配置模板事件處理器
            template_standard.click(
                fn=lambda: load_template_config("標準處理流程"),
                outputs=[pipeline_dir, pipeline_main_output_root, enable_validation, 
                        enable_face_detection, enable_lpips, enable_cropping, 
                        enable_classification, enable_upscaling, enable_tagging,
                        pipeline_face_output, pipeline_min_face, pipeline_lpips_output, 
                        pipeline_lpips_batch, pipeline_crop_output, pipeline_upscale_w, 
                        pipeline_upscale_h, pipeline_upscale_model, pipeline_upscale_min, 
                        config_status]
            )
            
            template_fast.click(
                fn=lambda: load_template_config("快速處理"),
                outputs=[pipeline_dir, pipeline_main_output_root, enable_validation, 
                        enable_face_detection, enable_lpips, enable_cropping, 
                        enable_classification, enable_upscaling, enable_tagging,
                        pipeline_face_output, pipeline_min_face, pipeline_lpips_output, 
                        pipeline_lpips_batch, pipeline_crop_output, pipeline_upscale_w, 
                        pipeline_upscale_h, pipeline_upscale_model, pipeline_upscale_min, 
                        config_status]
            )
            
            template_hq.click(
                fn=lambda: load_template_config("高品質處理"),
                outputs=[pipeline_dir, pipeline_main_output_root, enable_validation, 
                        enable_face_detection, enable_lpips, enable_cropping, 
                        enable_classification, enable_upscaling, enable_tagging,
                        pipeline_face_output, pipeline_min_face, pipeline_lpips_output, 
                        pipeline_lpips_batch, pipeline_crop_output, pipeline_upscale_w, 
                        pipeline_upscale_h, pipeline_upscale_model, pipeline_upscale_min,                        config_status]
            )
        
        with gr.Tab("📋 圖片驗證"):
            gr.Markdown("### 驗證圖像檔案完整性，移除損壞或無效的圖像")
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        v_dir = gr.Textbox(
                            label="圖片資料夾路徑", 
                            value=DEFAULT_VALUES['directory'],
                            placeholder="請輸入包含圖像的資料夾路徑",
                            scale=4
                        )
                        v_browse = gr.Button("📁 瀏覽", variant="secondary", size="sm", scale=1)
                    v_btn = gr.Button("🔍 開始驗證", variant="primary")
                with gr.Column():
                    v_out = gr.Textbox(label="處理結果", lines=3)
            
            # 事件處理器
            v_browse.click(fn=browse_input_folder, outputs=v_dir)
            v_btn.click(run_validation, inputs=v_dir, outputs=v_out)
            
        with gr.Tab("👤 人臉檢測"):
            gr.Markdown("### 偵測圖像中的人臉，篩選符合條件的圖像")
            with gr.Row():
                with gr.Column():
                    f_dir = gr.Textbox(
                        label="圖片資料夾路徑", 
                        value=DEFAULT_VALUES['directory'],
                        placeholder="請輸入包含圖像的資料夾路徑"
                    )
                    f_min = gr.Number(
                        value=DEFAULT_VALUES['min_face_count'], 
                        label="最小人臉數量",
                        info="圖像中至少需要包含的人臉數量"
                    )
                    f_outdir = gr.Textbox(
                        value=DEFAULT_VALUES['face_output_directory'], 
                        label="輸出資料夾",
                        placeholder="符合條件的圖像將保存到此資料夾"
                    )
                    f_btn = gr.Button("👁️ 開始檢測", variant="primary")
                with gr.Column():
                    f_out = gr.Textbox(label="處理結果", lines=3)
                    f_gallery = gr.Gallery(
                        label="檢測結果預覽", 
                        show_label=True, 
                        elem_id="face_gallery",
                        columns=2, 
                        rows=2, 
                        height="auto"
                    )
            f_btn.click(run_face_detection, inputs=[f_dir, f_min, f_outdir], outputs=[f_out, f_gallery])
            
        with gr.Tab("🔄 LPIPS 去重"):
            gr.Markdown("### 使用感知相似度去除重複或高度相似的圖像")
            with gr.Row():
                with gr.Column():
                    l_dir = gr.Textbox(
                        label="圖片資料夾路徑", 
                        value=DEFAULT_VALUES['directory'],
                        placeholder="請輸入包含圖像的資料夾路徑"
                    )
                    l_outdir = gr.Textbox(
                        value=DEFAULT_VALUES['lpips_output_directory'], 
                        label="輸出資料夾",
                        placeholder="去重後的圖像將保存到此資料夾"
                    )
                    l_bs = gr.Number(
                        value=DEFAULT_VALUES['lpips_batch_size'], 
                        label="批次大小",
                        info="每次處理的圖像數量，較大值需要更多記憶體"
                    )
                    l_btn = gr.Button("🔄 開始去重", variant="primary")
                with gr.Column():
                    l_out = gr.Textbox(label="處理結果", lines=3)
            l_btn.click(run_lpips, inputs=[l_dir, l_outdir, l_bs], outputs=l_out)
            
        with gr.Tab("✂️ 裁切與分類"):
            gr.Markdown("### 將圖像裁切成不同規格並分類儲存")
            with gr.Row():
                with gr.Column():
                    c_in = gr.Textbox(
                        label="輸入資料夾", 
                        value=DEFAULT_VALUES['directory'],
                        placeholder="請輸入包含圖像的資料夾路徑"
                    )
                    c_out = gr.Textbox(
                        label="輸出資料夾", 
                        value=DEFAULT_VALUES['output_directory'],
                        placeholder="裁切後的圖像將保存到此資料夾"
                    )
                    with gr.Row():
                        c_btn = gr.Button("✂️ 開始裁切", variant="primary")
                        cl_btn = gr.Button("📂 執行分類", variant="secondary")
                with gr.Column():
                    c_outbox = gr.Textbox(label="裁切結果", lines=2)
                    cl_out = gr.Textbox(label="分類結果", lines=2)
                    c_gallery = gr.Gallery(
                        label="裁切結果預覽", 
                        show_label=True, 
                        elem_id="crop_gallery",
                        columns=2, 
                        rows=2, 
                        height="auto"
                    )
            c_btn.click(run_crop, inputs=[c_in, c_out], outputs=[c_outbox, c_gallery])
            cl_btn.click(run_classify, inputs=c_out, outputs=cl_out)
            
        with gr.Tab("🏷️ 標籤產生"):
            gr.Markdown("### 自動為圖像生成描述性標籤")
            with gr.Row():
                with gr.Column():
                    t_dir = gr.Textbox(
                        label="圖片資料夾路徑", 
                        value=DEFAULT_VALUES['output_directory'],
                        placeholder="請輸入包含圖像的資料夾路徑"
                    )
                    t_btn = gr.Button("🏷️ 開始標記", variant="primary")
                with gr.Column():
                    t_out = gr.Textbox(label="處理結果", lines=3)
            t_btn.click(run_tag, inputs=t_dir, outputs=t_out)
            
        with gr.Tab("🔍 圖片放大"):
            gr.Markdown("### 使用超解析度技術放大低解析度圖像")
            with gr.Row():
                with gr.Column():
                    u_dir = gr.Textbox(
                        label="圖片資料夾路徑", 
                        value=DEFAULT_VALUES['output_directory'],
                        placeholder="請輸入包含圖像的資料夾路徑"
                    )
                    with gr.Row():
                        u_w = gr.Number(
                            value=DEFAULT_VALUES['upscale_target_width'], 
                            label="目標寬度 (px)",
                            info="放大後的圖像寬度"
                        )
                        u_h = gr.Number(
                            value=DEFAULT_VALUES['upscale_target_height'], 
                            label="目標高度 (px)",
                            info="放大後的圖像高度"
                        )
                    u_model = gr.Dropdown(
                        choices=[
                            "HGSR-MHR-anime-aug_X4_320",
                            "Real-ESRGAN_4x",
                            "Waifu2x"
                        ],
                        value=DEFAULT_VALUES['upscale_model'], 
                        label="超解析度模型",
                        info="選擇用於圖像放大的AI模型"
                    )
                    u_min = gr.Number(
                        value=DEFAULT_VALUES['upscale_min_size'], 
                        label="最小處理尺寸 (px)", 
                        precision=0,
                        info="只處理小於此尺寸的圖像"
                    )
                    with gr.Row():
                        u_overwrite = gr.Checkbox(
                            value=True, 
                            label="覆寫原檔案",
                            info="是否覆寫原始圖像檔案"
                        )
                        u_recursive = gr.Checkbox(
                            value=True, 
                            label="遞迴處理子目錄",
                            info="是否處理子資料夾中的圖像"
                        )
                    u_btn = gr.Button("🔍 開始放大", variant="primary")
                with gr.Column():
                    u_out = gr.Textbox(label="處理結果", lines=3)
                    u_gallery = gr.Gallery(
                        label="放大結果預覽", 
                        show_label=True, 
                        elem_id="upscale_gallery",
                        columns=2, 
                        rows=2, 
                        height="auto"
                    )
            u_btn.click(run_upscale, inputs=[u_dir, u_w, u_h, u_model, u_min, u_overwrite, u_recursive], outputs=[u_out, u_gallery])

            # 配置管理事件處理
            save_config_btn.click(
                save_pipeline_config,
                inputs=[
                    pipeline_dir, pipeline_main_output_root,
                    enable_validation, enable_face_detection, enable_lpips, enable_cropping,
                    enable_classification, enable_upscaling, enable_tagging,
                    pipeline_face_output, pipeline_min_face, pipeline_lpips_output, pipeline_lpips_batch,
                    pipeline_crop_output, pipeline_upscale_w, pipeline_upscale_h, 
                    pipeline_upscale_model, pipeline_upscale_min
                ],
                outputs=[pipeline_out]
            )
            
            load_config_btn.click(
                load_pipeline_config,
                outputs=[
                    pipeline_dir, pipeline_main_output_root,
                    enable_validation, enable_face_detection, enable_lpips, enable_cropping,
                    enable_classification, enable_upscaling, enable_tagging,
                    pipeline_face_output, pipeline_min_face, pipeline_lpips_output, pipeline_lpips_batch,
                    pipeline_crop_output, pipeline_upscale_w, pipeline_upscale_h, 
                    pipeline_upscale_model, pipeline_upscale_min,
                    pipeline_out
                ]
            )    
    # 設置隊列以支持異步操作
    demo.queue(max_size=1022)
    return demo


if __name__ == "__main__":
    try:
        demo = build_interface()
        
        # 修復前端資源載入問題的啟動配置
        demo.launch(
            server_name="127.0.0.1",
            server_port=7860,
            show_error=False,  # 隱藏錯誤以避免 404 顯示
            quiet=True,  # 安靜模式減少日誌輸出
            inbrowser=True,
            favicon_path=None,  # 避免favicon載入問題
            prevent_thread_lock=False,
            show_api=False,  # 禁用 API 文檔以減少資源載入
            enable_monitoring=False,  # 禁用監控以減少資源請求
            root_path="/",  # 設定根路徑
            ssl_verify=False,  # 禁用 SSL 驗證
            app_kwargs={
                "docs_url": None,  # 禁用API文檔
                "redoc_url": None,  # 禁用 Redoc
                "openapi_url": None,  # 禁用 OpenAPI
            }
        )
    except Exception as e:
        logger.error(f"Gradio 應用啟動失敗: {str(e)}")
        print(f"❌ 應用啟動失敗: {str(e)}")
        print("🔧 嘗試基本模式啟動...")
        try:
            demo = build_interface()
            demo.launch(
                server_name="127.0.0.1",
                server_port=7860,
                quiet=True,
                show_error=False
            )
        except Exception as fallback_error:
            print(f"❌ 基本模式啟動也失敗: {str(fallback_error)}")
            raise

async def set_global_input_folder():
    """設定全域輸入資料夾，返回狀態訊息"""
    try:
        folder_path = await browse_input_folder()
        if folder_path and folder_path != "":
            return f"✅ 已設定專案主要資料夾: {os.path.basename(folder_path)}"
        else:
            return "❌ 未選擇資料夾"
    except Exception as e:
        return f"❌ 設定失敗: {str(e)}"


def tk_window_asksaveasfilename(title="儲存檔案", defaultext=".json", filetypes=None):
    """使用 tkinter 選擇保存檔案"""
    window = tk.Tk()
    window.wm_attributes('-topmost', 1)
    window.withdraw()
    if filetypes is None:
        filetypes = [("JSON files", "*.json"), ("All files", "*.*")]
    filename = filedialog.asksaveasfilename(
        title=title,
        defaultextension=defaultext,
        filetypes=filetypes
    )
    window.destroy()
    return filename


def tk_window_askopenfilename(title="開啟檔案", filetypes=None):
    """使用 tkinter 選擇開啟檔案"""
    window = tk.Tk()
    window.wm_attributes('-topmost', 1)
    window.withdraw()
    if filetypes is None:
        filetypes = [("JSON files", "*.json"), ("All files", "*.*")]
    filename = filedialog.askopenfilename(
        title=title,
        filetypes=filetypes
    )
    window.destroy()
    return filename


async def tk_asksaveasfilename_async(title="儲存檔案", defaultext=".json", filetypes=None):
    """異步選擇保存檔案"""
    filename = await asyncio.to_thread(tk_window_asksaveasfilename, title, defaultext, filetypes)
    return filename


async def tk_askopenfilename_async(title="開啟檔案", filetypes=None):
    """異步選擇開啟檔案"""
    filename = await asyncio.to_thread(tk_window_askopenfilename, title, filetypes)
    return filename




