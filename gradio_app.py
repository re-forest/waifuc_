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
