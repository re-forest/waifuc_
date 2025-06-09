import os
import gradio as gr
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
