import os
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
    return demo


if __name__ == "__main__":
    ui().launch()
