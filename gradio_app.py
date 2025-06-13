import os
from dotenv import load_dotenv
import gradio as gr
import time
from tqdm import tqdm

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
        
        # 獲取預覽圖像
        preview_images = []
        if os.path.exists(output_dir):
            image_files = [f for f in os.listdir(output_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))][:4]  # 最多顯示4張
            preview_images = [os.path.join(output_dir, f) for f in image_files]
        
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
        file_paths = [os.path.join(directory, f) for f in os.listdir(directory)
                      if os.path.isfile(os.path.join(directory, f))]
        
        logger.debug(f"發現 {len(file_paths)} 個檔案")
        progress(0.2, desc=f"發現 {len(file_paths)} 張圖片，開始處理...")
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
    enable_validation, enable_face_detection, enable_lpips, enable_cropping, 
    enable_classification, enable_upscaling, enable_tagging,
    face_output_dir, min_face_count, lpips_output_dir, lpips_batch_size,
    crop_output_dir, upscale_width, upscale_height, upscale_model, upscale_min_size,
    progress=gr.Progress()
):
    """執行整合式處理流程"""
    results = []
    step = 0
    total_steps = sum([enable_validation, enable_face_detection, enable_lpips, 
                      enable_cropping, enable_classification, enable_upscaling, enable_tagging])
    
    current_dir = directory
    
    try:
        # 1. 圖片驗證
        if enable_validation:
            progress(step/total_steps, desc="正在驗證圖片...")
            processed, removed = validate_and_remove_invalid_images(current_dir)
            results.append(f"✅ 圖片驗證: 處理 {processed} 張，刪除 {removed} 張無效圖片")
            step += 1
        
        # 2. 人臉檢測
        if enable_face_detection:
            progress(step/total_steps, desc="正在進行人臉檢測...")
            processed, moved = detect_faces_in_directory(current_dir, min_face_count, face_output_dir)
            results.append(f"✅ 人臉檢測: 檢測 {processed} 張，移動 {moved} 張符合條件的圖片")
            current_dir = face_output_dir  # 更新當前工作目錄
            step += 1
        
        # 3. LPIPS 去重
        if enable_lpips:
            progress(step/total_steps, desc="正在進行 LPIPS 去重...")
            file_paths = [os.path.join(current_dir, f) for f in os.listdir(current_dir)
                         if os.path.isfile(os.path.join(current_dir, f))]
            result_dir = process_lpips_clustering(file_paths, lpips_output_dir, lpips_batch_size)
            results.append(f"✅ LPIPS 去重: 處理完成，結果保存在 {result_dir}")
            current_dir = result_dir  # 更新當前工作目錄
            step += 1
        
        # 4. 圖像裁切
        if enable_cropping:
            progress(step/total_steps, desc="正在進行圖像裁切...")
            process_single_folder(current_dir, crop_output_dir)
            results.append(f"✅ 圖像裁切: 完成，結果保存在 {crop_output_dir}")
            current_dir = crop_output_dir  # 更新當前工作目錄
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
        
        # 獲取最終結果預覽
        preview_images = []
        if os.path.exists(current_dir):
            for root, dirs, files in os.walk(current_dir):
                image_files = [f for f in files 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))][:6]
                preview_images.extend([os.path.join(root, f) for f in image_files])
                if len(preview_images) >= 6:
                    break
        
        final_result = "🎉 整合處理流程完成!\n\n" + "\n".join(results)
        return final_result, preview_images[:6]
        
    except Exception as e:
        error_msg = f"❌ 處理過程中發生錯誤: {str(e)}"
        results.append(error_msg)
        logger.error(error_msg)  # 記錄錯誤日誌
        return "\n".join(results), []


def build_interface():
    with gr.Blocks(title="Waifuc 圖像處理系統") as demo:
        gr.Markdown("# 🎨 Waifuc 圖像處理介面")
        gr.Markdown("專為動漫風格圖像預處理設計的自動化工具")
        
        with gr.Tab("🚀 整合處理流程"):
            gr.Markdown("### 一鍵執行完整的圖像預處理流程")
            
            with gr.Row():
                with gr.Column():
                    # 基本設定
                    gr.Markdown("#### 📁 基本設定")
                    pipeline_dir = gr.Textbox(
                        label="主要圖片資料夾", 
                        value=DEFAULT_VALUES['directory'],
                        placeholder="請輸入包含原始圖像的資料夾路徑"
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
                            label="圖像標記"
                        )
                    
                    # 詳細參數設定
                    with gr.Accordion("🔧 詳細參數設定", open=False):
                        gr.Markdown("##### 人臉檢測參數")
                        pipeline_face_output = gr.Textbox(
                            value=DEFAULT_VALUES['face_output_directory'], 
                            label="人臉檢測輸出目錄"
                        )
                        pipeline_min_face = gr.Number(
                            value=DEFAULT_VALUES['min_face_count'], 
                            label="最小人臉數量"
                        )
                        
                        gr.Markdown("##### LPIPS 去重參數")
                        pipeline_lpips_output = gr.Textbox(
                            value=DEFAULT_VALUES['lpips_output_directory'], 
                            label="LPIPS 輸出目錄"
                        )
                        pipeline_lpips_batch = gr.Number(
                            value=DEFAULT_VALUES['lpips_batch_size'], 
                            label="LPIPS 批次大小"
                        )
                        
                        gr.Markdown("##### 裁切參數")
                        pipeline_crop_output = gr.Textbox(
                            value=DEFAULT_VALUES['output_directory'], 
                            label="裁切輸出目錄"
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
                    
                    pipeline_btn = gr.Button("🚀 開始整合處理", variant="primary", size="lg")
                
                with gr.Column():
                    pipeline_out = gr.Textbox(label="處理進度與結果", lines=10)
                    pipeline_gallery = gr.Gallery(
                        label="最終結果預覽", 
                        show_label=True, 
                        elem_id="pipeline_gallery",
                        columns=3, 
                        rows=2, 
                        height="auto"
                    )
            
            pipeline_btn.click(
                run_integrated_pipeline,
                inputs=[
                    pipeline_dir,
                    enable_validation, enable_face_detection, enable_lpips, enable_cropping,
                    enable_classification, enable_upscaling, enable_tagging,
                    pipeline_face_output, pipeline_min_face, pipeline_lpips_output, pipeline_lpips_batch,
                    pipeline_crop_output, pipeline_upscale_w, pipeline_upscale_h, 
                    pipeline_upscale_model, pipeline_upscale_min
                ],
                outputs=[pipeline_out, pipeline_gallery]
            )
        
        with gr.Tab("📋 圖片驗證"):
            gr.Markdown("### 驗證圖像檔案完整性，移除損壞或無效的圖像")
            with gr.Row():
                with gr.Column():
                    v_dir = gr.Textbox(
                        label="圖片資料夾路徑", 
                        value=DEFAULT_VALUES['directory'],
                        placeholder="請輸入包含圖像的資料夾路徑"
                    )
                    v_btn = gr.Button("🔍 開始驗證", variant="primary")
                with gr.Column():
                    v_out = gr.Textbox(label="處理結果", lines=3)
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

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch()
