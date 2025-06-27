#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gradio as gr
import os
from pathlib import Path
from dotenv import load_dotenv

# 導入重構後的模組函式
from main import run_full_pipeline
from validate_image import validate_and_remove_invalid_images
from transparency import scan_directory, process_transparency
from face_detection import detect_faces_in_directory
from lpips_clustering import process_lpips_clustering_with_summary
from crop import process_cropping_and_classification
from upscale import upscale_images_with_summary
from tag import tag_images_with_summary

def run_complete_pipeline(
    directory,
    selected_steps,
    enable_upscaling,
    min_face_count,
    face_output_dir,
    lpips_output_dir,
    lpips_batch_size,
    crop_output_dir,
    upscale_width,
    upscale_height,
    upscale_model
):
    """執行完整的圖片處理流程"""
    
    if not directory or not Path(directory).exists():
        return "錯誤: 請提供有效的目錄路徑", "處理失敗"
    
    # 建立設定字典
    config = {
        "directory": directory,
        "enable_validation": "驗證圖片" in selected_steps,
        "enable_transparency_processing": "透明度處理" in selected_steps,
        "enable_face_detection": "人臉偵測" in selected_steps,
        "min_face_count": min_face_count,
        "face_output_directory": face_output_dir or "face_out",
        "enable_lpips_clustering": "圖片去重複" in selected_steps,
        "lpips_output_directory": lpips_output_dir or "lpips_output",
        "lpips_batch_size": lpips_batch_size,
        "enable_cropping": "裁切與分類" in selected_steps,
        "enable_classification": "裁切與分類" in selected_steps,
        "enable_upscaling": enable_upscaling,
        "upscale_target_width": upscale_width,
        "upscale_target_height": upscale_height,
        "upscale_model": upscale_model,
        "upscale_min_size": 800,
        "enable_tagging": "圖片標記" in selected_steps,
        "output_directory": crop_output_dir
    }
    
    # 執行完整流程
    result = run_full_pipeline(config)
    
    if result['success']:
        summary = f"""
## 處理完成! ✅

**處理的目錄**: {directory}
**執行的步驟**: {', '.join(selected_steps)}
{'**放大處理**: 已啟用' if enable_upscaling else '**放大處理**: 未啟用'}

**詳細日誌請參考下方輸出**
        """
        return result['logs'], summary
    else:
        error_summary = f"""
## 處理失敗 ❌

**錯誤訊息**: {result['error']}

**詳細日誌請參考下方輸出**
        """
        return result['logs'], error_summary

def validate_images_interface(directory_path):
    """圖片驗證介面"""
    if not directory_path or not Path(directory_path).exists():
        return "錯誤: 請提供有效的目錄路徑"
    
    result = validate_and_remove_invalid_images(directory_path)
    
    if result['error']:
        return f"錯誤: {result['error']}"
    else:
        return f"""驗證完成:
- 總檔案數: {result['total_files']}
- 移除檔案數: {result['removed_files']}
- 有效檔案數: {result['valid_files']}"""

def transparency_scan_interface(directory_path):
    """透明度掃描介面"""
    if not directory_path or not Path(directory_path).exists():
        return "錯誤: 請提供有效的目錄路徑", None
    
    results = scan_directory(directory_path)
    
    # 轉換為顯示格式
    display_data = []
    for result in results:
        display_data.append([
            result['file_path'],
            "是" if result['has_transparency'] else "否"
        ])
    
    summary = f"""掃描完成:
- 總檔案數: {len(results)}
- 透明圖片數: {sum(1 for r in results if r['has_transparency'])}
- 非透明圖片數: {sum(1 for r in results if not r['has_transparency'])}"""
    
    return summary, display_data

def transparency_convert_interface(directory_path):
    """透明度轉換介面"""
    if not directory_path or not Path(directory_path).exists():
        return "錯誤: 請提供有效的目錄路徑"
    
    result = process_transparency(directory_path, convert=True)
    
    if result['error']:
        return f"錯誤: {result['error']}"
    else:
        return f"""轉換完成:
- 總檔案數: {result['total_files']}
- 透明圖片數: {result['transparent_files']}
- 已轉換數: {result['converted_files']}"""

def face_detection_interface(input_dir, min_faces, output_dir):
    """人臉偵測介面"""
    if not input_dir or not Path(input_dir).exists():
        return "錯誤: 請提供有效的輸入目錄路徑"
    
    if not output_dir:
        output_dir = "face_out"
    
    result = detect_faces_in_directory(input_dir, min_faces, output_dir)
    
    if result['error']:
        return f"錯誤: {result['error']}"
    else:
        return f"""人臉偵測完成:
- 處理檔案數: {result['total_files']}
- 移動檔案數: {result['moved_files']}
- 剩餘檔案數: {result['remaining_files']}
- 輸出目錄: {result['output_folder']}"""

def lpips_clustering_interface(input_dir, output_dir, batch_size, similarity_threshold):
    """圖片去重複介面"""
    if not input_dir or not Path(input_dir).exists():
        return "錯誤: 請提供有效的輸入目錄路徑"
    
    if not output_dir:
        output_dir = "lpips_output"
    
    # 獲取檔案路徑列表
    file_paths = [str(p) for p in Path(input_dir).iterdir() if p.is_file()]
    
    if not file_paths:
        return "錯誤: 輸入目錄中沒有找到檔案"
    
    result = process_lpips_clustering_with_summary(file_paths, output_dir, batch_size)
    
    if result['error']:
        return f"錯誤: {result['error']}"
    else:
        return f"""去重複處理完成:
- 輸入檔案數: {result['total_files']}
- 處理檔案數: {result['processed_files']}
- 建立群組數: {result['groups_created']}
- 輸出目錄: {result['output_directory']}"""

def crop_classification_interface(input_dir, output_dir):
    """裁切與分類介面"""
    if not input_dir or not Path(input_dir).exists():
        return "錯誤: 請提供有效的輸入目錄路徑"
    
    if not output_dir:
        output_dir = str(Path(input_dir).parent / "cropped")
    
    result = process_cropping_and_classification(input_dir, output_dir)
    
    if result['error']:
        return f"錯誤: {result['error']}"
    else:
        # 安全地獲取分類統計
        categories = result.get('categories', {})
        head_count = categories.get('head', 0)
        halfbody_count = categories.get('halfbody', 0)
        person_count = categories.get('person', 0)
        other_count = categories.get('other', 0)
        
        return f"""裁切與分類完成:
- 處理檔案數: {result['processed_files']}
- 頭像數量: {head_count}
- 半身數量: {halfbody_count}
- 全身數量: {person_count}
- 其他數量: {other_count}
- 輸出目錄: {result['output_directory']}"""

def tag_images_interface(input_dir):
    """圖片標記介面"""
    if not input_dir or not Path(input_dir).exists():
        return "錯誤: 請提供有效的輸入目錄路徑"
    
    result = tag_images_with_summary(input_dir)
    
    if result['error']:
        return f"錯誤: {result['error']}"
    else:
        return f"""標記完成:
- 處理圖片數: {result['processed_images']}
- 失敗圖片數: {result['failed_images']}
- 生成標籤數: {result['tags_generated']}
- 使用自定義角色: {result['custom_character'] if result['custom_character'] else '無'}
- 使用自定義藝術家: {result['custom_artist'] if result['custom_artist'] else '無'}
- Wildcard 功能: {'啟用' if result['wildcard_enabled'] else '停用'}"""

# 建立 Gradio 介面
def create_gradio_interface():
    """建立 Gradio 使用者介面"""
    
    # 載入環境變數作為預設值
    load_dotenv()

    with gr.Blocks(title="Waifuc 圖片處理工具箱") as app:

        gr.Markdown("# 🎨 Waifuc 圖片處理工具箱")
        gr.Markdown("整合式圖片處理工具，支援完整流程執行和單獨功能操作")
        
        with gr.Tabs():
            # 完整流程標籤
            with gr.TabItem("🔄 完整流程"):
                gr.Markdown("## 執行完整的圖片處理流程")
                
                with gr.Row():
                    with gr.Column():
                        directory_input = gr.Textbox(
                            label="📁 處理目錄路徑",
                            placeholder="輸入要處理的圖片目錄路徑",
                            value=os.getenv("directory", "")
                        )
                        
                        steps_checkbox = gr.CheckboxGroup(
                            label="🛠️ 選擇處理步驟",
                            choices=["驗證圖片", "透明度處理", "人臉偵測", "圖片去重複", "裁切與分類", "圖片標記"],
                            value=["驗證圖片", "透明度處理", "人臉偵測", "圖片去重複", "裁切與分類", "圖片標記"]
                        )
                        
                        upscale_checkbox = gr.Checkbox(
                            label="📈 執行圖片放大 (可選)",
                            value=False
                        )
                
                with gr.Accordion("🔧 進階設定", open=False):
                    with gr.Row():
                        min_face_count = gr.Number(
                            label="最小人臉數量",
                            value=1,
                            minimum=1
                        )
                        face_output = gr.Textbox(
                            label="人臉偵測輸出目錄",
                            value="face_out"
                        )
                    
                    with gr.Row():
                        lpips_output = gr.Textbox(
                            label="LPIPS 輸出目錄",
                            value="lpips_output"
                        )
                        lpips_batch = gr.Number(
                            label="LPIPS 批次大小",
                            value=100,
                            minimum=1
                        )
                    
                    with gr.Row():
                        crop_output = gr.Textbox(
                            label="裁切輸出目錄",
                            placeholder="留空使用預設值"
                        )
                    
                    with gr.Row():
                        upscale_width = gr.Number(
                            label="放大目標寬度",
                            value=1024,
                            minimum=64
                        )
                        upscale_height = gr.Number(
                            label="放大目標高度",
                            value=1024,
                            minimum=64
                        )
                        upscale_model = gr.Textbox(
                            label="放大模型名稱",
                            value="HGSR-MHR-anime-aug_X4_320"
                        )
                
                run_button = gr.Button("▶️ 開始完整處理流程", variant="primary", size="lg")
                
                with gr.Row():
                    pipeline_logs = gr.Textbox(
                        label="📋 處理日誌",
                        lines=15,
                        max_lines=20
                    )
                    pipeline_summary = gr.Markdown("等待開始處理...")
                
                run_button.click(
                    fn=run_complete_pipeline,
                    inputs=[
                        directory_input, steps_checkbox, upscale_checkbox,
                        min_face_count, face_output, lpips_output, lpips_batch,
                        crop_output, upscale_width, upscale_height, upscale_model
                    ],
                    outputs=[pipeline_logs, pipeline_summary]
                )
            
            # 單元功能標籤
            with gr.TabItem("🔧 單元功能"):
                
                # 圖片驗證
                with gr.Accordion("✅ 圖片驗證", open=False):
                    with gr.Row():
                        validate_dir = gr.Textbox(label="目錄路徑")
                        validate_btn = gr.Button("執行驗證")
                    validate_output = gr.Textbox(label="驗證結果", lines=3)
                    validate_btn.click(validate_images_interface, validate_dir, validate_output)
                
                # 透明度處理
                with gr.Accordion("🔍 透明度處理", open=False):
                    with gr.Row():
                        transparency_dir = gr.Textbox(label="目錄路徑")
                    with gr.Row():
                        scan_btn = gr.Button("掃描透明圖片")
                        convert_btn = gr.Button("轉換透明圖片")
                    
                    scan_result = gr.Textbox(label="掃描結果", lines=3)
                    scan_data = gr.Dataframe(
                        headers=["檔案路徑", "是否透明"],
                        label="掃描詳情"
                    )
                    convert_result = gr.Textbox(label="轉換結果", lines=3)
                    
                    scan_btn.click(transparency_scan_interface, transparency_dir, [scan_result, scan_data])
                    convert_btn.click(transparency_convert_interface, transparency_dir, convert_result)
                
                # 人臉偵測
                with gr.Accordion("👤 人臉偵測", open=False):
                    with gr.Row():
                        face_input_dir = gr.Textbox(label="輸入目錄路徑")
                        face_min_count = gr.Number(label="最小人臉數量", value=1, minimum=1)
                        face_out_dir = gr.Textbox(label="輸出目錄路徑", value="face_out")
                    face_btn = gr.Button("執行人臉偵測")
                    face_result = gr.Textbox(label="偵測結果", lines=3)
                    face_btn.click(face_detection_interface, [face_input_dir, face_min_count, face_out_dir], face_result)
                
                # 圖片去重複
                with gr.Accordion("🔄 圖片去重複", open=False):
                    with gr.Row():
                        lpips_input_dir = gr.Textbox(label="輸入目錄路徑")
                        lpips_out_dir = gr.Textbox(label="輸出目錄路徑", value="lpips_output")
                    with gr.Row():
                        lpips_batch_size = gr.Number(label="批次大小", value=100, minimum=1)
                        similarity_threshold = gr.Slider(label="相似度閾值", minimum=0.0, maximum=1.0, value=0.8, step=0.01)
                    lpips_btn = gr.Button("執行去重複")
                    lpips_result = gr.Textbox(label="處理結果", lines=3)
                    lpips_btn.click(lpips_clustering_interface, [lpips_input_dir, lpips_out_dir, lpips_batch_size, similarity_threshold], lpips_result)
                
                # 圖片裁切與分類
                with gr.Accordion("✂️ 圖片裁切與分類", open=False):
                    with gr.Row():
                        crop_input_dir = gr.Textbox(label="輸入目錄路徑")
                        crop_out_dir = gr.Textbox(label="輸出目錄路徑", placeholder="留空使用預設值")
                    crop_btn = gr.Button("執行裁切與分類")
                    crop_result = gr.Textbox(label="處理結果", lines=5)
                    crop_btn.click(crop_classification_interface, [crop_input_dir, crop_out_dir], crop_result)
                
                # 圖片標記
                with gr.Accordion("🏷️ 圖片標記", open=False):
                    tag_input_dir = gr.Textbox(label="輸入目錄路徑")
                    tag_btn = gr.Button("執行圖片標記")
                    tag_result = gr.Textbox(label="標記結果", lines=5)
                    tag_btn.click(tag_images_interface, tag_input_dir, tag_result)
        
        gr.Markdown("""
        ---
        💡 **使用提示**:
        - 完整流程會依序執行選定的處理步驟
        - 單元功能可以獨立執行特定的處理任務
        - 建議先備份重要圖片再進行處理
        - 某些功能可能需要較長的處理時間，請耐心等待
        """)
    
    return app

if __name__ == "__main__":
    # 啟動應用程式
    app = create_gradio_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7888,
        share=False,
        debug=False
    )