import os
from dotenv import load_dotenv
import gradio as gr
import time
from tqdm import tqdm

from validate_image import validate_and_remove_invalid_images
from face_detection import detect_faces_in_directory
from lpips_clustering import process_lpips_clustering
from crop import process_single_folder, classify_files_in_directory
from tag import tag_image
from upscale import upscale_images_in_directory

# 載入環境變數
load_dotenv()

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
    progress(0, desc="開始驗證圖像...")
    processed, removed = validate_and_remove_invalid_images(directory)
    progress(1, desc="驗證完成")
    return f"已處理 {processed} 張圖片，刪除 {removed} 張無效圖片"


def run_face_detection(directory, min_face, output_dir, progress=gr.Progress()):
    """人臉檢測並顯示進度"""
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
    
    return f"已檢測 {processed} 張圖片，移動 {moved} 張圖片", preview_images


def run_lpips(directory, output_dir, batch_size, progress=gr.Progress()):
    """LPIPS去重並顯示進度"""
    if not os.path.isdir(directory):
        return "指定目錄不存在"
    
    progress(0, desc="準備處理 LPIPS 去重...")
    file_paths = [os.path.join(directory, f) for f in os.listdir(directory)
                  if os.path.isfile(os.path.join(directory, f))]
    
    progress(0.2, desc=f"發現 {len(file_paths)} 張圖片，開始處理...")
    result_dir = process_lpips_clustering(file_paths, output_dir, batch_size)
    progress(1, desc="LPIPS 去重完成")
    return f"處理完成，結果位於 {result_dir}"


def run_crop(input_path, output_path, progress=gr.Progress()):
    """圖像裁切並顯示進度"""
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
    
    return "裁切完成", preview_images[:4]


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


def build_interface():
    with gr.Blocks(title="Waifuc 圖像處理系統") as demo:
        gr.Markdown("# 🎨 Waifuc 圖像處理介面")
        gr.Markdown("專為動漫風格圖像預處理設計的自動化工具")
        
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
