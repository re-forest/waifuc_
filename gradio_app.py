import os
from dotenv import load_dotenv
import gradio as gr
import main

load_dotenv()

def run_pipeline(directory, output_directory):
    if directory:
        os.environ["directory"] = directory
    if output_directory:
        os.environ["output_directory"] = output_directory
    code = main.main()
    return "處理完成" if code == 0 else f"處理失敗，錯誤碼 {code}"

# 從環境變數讀取預設值
DEFAULT_DIR = os.getenv("directory", "")
DEFAULT_OUT = os.getenv("output_directory", "")

with gr.Blocks() as demo:
    gr.Markdown("# Waifuc 圖像處理工具箱")
    dir_box = gr.Textbox(label="圖片來源資料夾", value=DEFAULT_DIR,
                         placeholder="/path/to/images")
    out_box = gr.Textbox(label="輸出資料夾", value=DEFAULT_OUT,
                         placeholder="/path/to/output")
    run_btn = gr.Button("開始處理")
    status = gr.Textbox(label="狀態輸出", interactive=False)

    run_btn.click(run_pipeline, inputs=[dir_box, out_box], outputs=status)

if __name__ == "__main__":
    demo.launch()
=======
import gradio as gr
from crop import process_single_folder, classify_files_in_directory


def run_crop(input_path, output_path, include_subfolders=False):
    os.makedirs(output_path, exist_ok=True)
    # 處理主資料夾
    process_single_folder(input_path, output_path)

    if include_subfolders:
        for subdir, _, _ in os.walk(input_path):
            rel_path = os.path.relpath(subdir, input_path)
            if rel_path == '.':
                continue
            sub_input = os.path.join(input_path, rel_path)
            sub_output = os.path.join(output_path, rel_path)
            os.makedirs(sub_output, exist_ok=True)
            process_single_folder(sub_input, sub_output)

    # 完成裁切後立即分類
    classify_files_in_directory(output_path)

    return f"裁切與分類完成，輸出目錄：{output_path}"


def ui():
    with gr.Blocks() as demo:
        gr.Markdown("## Waifuc 圖像裁切與分類工具")
        input_path = gr.Textbox(label="輸入資料夾")
        output_path = gr.Textbox(label="輸出資料夾")
        include_subfolders = gr.Checkbox(label="包含子資料夾", value=False)
        run_btn = gr.Button("執行裁切並分類")
        status = gr.Textbox(label="狀態")

        run_btn.click(run_crop, inputs=[input_path, output_path, include_subfolders], outputs=status)
=======

from dotenv import load_dotenv

from validate_image import validate_and_remove_invalid_images
from face_detection import detect_faces_in_directory
from lpips_clustering import process_lpips_clustering

# 確保環境變數被載入
load_dotenv()

def pipeline(directory):
    """執行完整的圖像處理流程並即時回報進度"""
    with gr.Progress(track_tqdm=True):
        yield "開始驗證圖片..."
        validate_and_remove_invalid_images(directory)
        yield "驗證完成\n開始人臉檢測..."

        detect_faces_in_directory(directory)
        yield "人臉檢測完成\n開始LPIPS聚類..."

        file_paths = [os.path.join(directory, f) for f in os.listdir(directory)
                      if os.path.isfile(os.path.join(directory, f))]
        process_lpips_clustering(file_paths)
        yield "LPIPS聚類完成\n所有處理已完成!"

with gr.Blocks() as demo:
    gr.Markdown("# Waifuc 圖像處理工具箱")
    directory = gr.Textbox(label="圖片資料夾")
    run = gr.Button("開始處理")
    output = gr.Textbox(label="狀態", interactive=False)

    run.click(pipeline, inputs=directory, outputs=output)

demo.launch()
=======
from validate_image import validate_and_remove_invalid_images
from face_detection import detect_faces_in_directory
from lpips_clustering import process_lpips_clustering
from crop import process_single_folder, classify_files_in_directory
from tag import tag_image
from upscale import upscale_images_in_directory


def run_validation(directory):
    processed, removed = validate_and_remove_invalid_images(directory)
    return f"已處理 {processed} 張圖片，刪除 {removed} 張無效圖片"


def run_face_detection(directory, min_face, output_dir):
    processed, moved = detect_faces_in_directory(directory, min_face, output_dir)
    return f"已檢測 {processed} 張圖片，移動 {moved} 張圖片"


def run_lpips(directory, output_dir, batch_size):
    if not os.path.isdir(directory):
        return "指定目錄不存在"
    file_paths = [os.path.join(directory, f) for f in os.listdir(directory)
                  if os.path.isfile(os.path.join(directory, f))]
    result_dir = process_lpips_clustering(file_paths, output_dir, batch_size)
    return f"處理完成，結果位於 {result_dir}"


def run_crop(input_path, output_path):
    process_single_folder(input_path, output_path)
    return "裁切完成"


def run_classify(directory):
    classify_files_in_directory(directory)
    return "分類完成"


def run_tag(directory):
    count = tag_image(directory)
    return f"標記完成，共處理 {count} 張圖片"


def run_upscale(directory, width, height, model, min_size):
    total, upscaled = upscale_images_in_directory(
        directory,
        target_width=width,
        target_height=height,
        model=model,
        min_size=min_size,
        overwrite=True,
        recursive=True,
    )
    return f"處理 {total} 張圖片，放大 {upscaled} 張"


def build_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Waifuc 圖像處理介面")
        with gr.Tab("圖片驗證"):
            v_dir = gr.Textbox(label="圖片資料夾")
            v_btn = gr.Button("開始驗證")
            v_out = gr.Textbox()
            v_btn.click(run_validation, inputs=v_dir, outputs=v_out)
        with gr.Tab("人臉檢測"):
            f_dir = gr.Textbox(label="圖片資料夾")
            f_min = gr.Number(value=1, label="最小人臉數量")
            f_outdir = gr.Textbox(value="face_out", label="輸出資料夾")
            f_btn = gr.Button("開始檢測")
            f_out = gr.Textbox()
            f_btn.click(run_face_detection, inputs=[f_dir, f_min, f_outdir], outputs=f_out)
        with gr.Tab("LPIPS 去重"):
            l_dir = gr.Textbox(label="圖片資料夾")
            l_outdir = gr.Textbox(value="lpips_output", label="輸出資料夾")
            l_bs = gr.Number(value=100, label="批次大小")
            l_btn = gr.Button("開始處理")
            l_out = gr.Textbox()
            l_btn.click(run_lpips, inputs=[l_dir, l_outdir, l_bs], outputs=l_out)
        with gr.Tab("裁切與分類"):
            c_in = gr.Textbox(label="輸入資料夾")
            c_out = gr.Textbox(label="輸出資料夾")
            c_btn = gr.Button("開始裁切")
            c_outbox = gr.Textbox()
            c_btn.click(run_crop, inputs=[c_in, c_out], outputs=c_outbox)
            cl_btn = gr.Button("執行分類")
            cl_out = gr.Textbox()
            cl_btn.click(run_classify, inputs=c_out, outputs=cl_out)
        with gr.Tab("標籤產生"):
            t_dir = gr.Textbox(label="圖片資料夾")
            t_btn = gr.Button("開始標記")
            t_out = gr.Textbox()
            t_btn.click(run_tag, inputs=t_dir, outputs=t_out)
        with gr.Tab("圖片放大"):
            u_dir = gr.Textbox(label="圖片資料夾")
            u_w = gr.Number(value=1024, label="寬度")
            u_h = gr.Number(value=1024, label="高度")
            u_model = gr.Textbox(value="HGSR-MHR-anime-aug_X4_320", label="模型")
            u_min = gr.Number(value=800, label="最小尺寸", precision=0)
            u_btn = gr.Button("開始放大")
            u_out = gr.Textbox()
            u_btn.click(run_upscale, inputs=[u_dir, u_w, u_h, u_model, u_min], outputs=u_out)

    return demo


if __name__ == "__main__":
    ui().launch()
=======
    demo = build_interface()
    demo.launch()

