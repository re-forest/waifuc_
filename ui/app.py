# ui/app.py
import gradio as gr
import os
import sys
import logging
import traceback
from typing import Union, Any
from PIL import Image # Import PIL Image for type hinting and conversion if needed

# --- Add project root to sys.path for direct execution ---
if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
# --- End sys.path modification ---

# Fallback settings class - define globally
class FallbackSettings:
        GRADIO_TITLE = "Fallback WaifuC - Error Mode"
        CUSTOM_CSS = None
        LOG_DIR = os.path.join(os.getcwd(), "logs_fallback")
        LOG_LEVEL = "DEBUG"
        GRADIO_THEME = "default"
        GRADIO_SERVER_NAME = "0.0.0.0"
        GRADIO_SERVER_PORT = 7861
        GRADIO_SHARE = False
        TEMP_PROCESSING_DIR = os.path.join(os.getcwd(), "temp_processing_fallback")
        GRADIO_TEMP_DIR = os.path.join(os.getcwd(), "temp_previews_fallback")
        URL_DOWNLOAD_TIMEOUT = 10

        AVAILABLE_STEPS = {
            "mock_fallback_action1": "Mock Action 1 (Fallback UI)",
            "mock_fallback_action2": "Mock Action 2 (Fallback UI)"
        }
        # Define ENABLE flags for these mock steps
        ENABLE_MOCK_FALLBACK_ACTION1 = False
        ENABLE_MOCK_FALLBACK_ACTION2 = False

        # Define standard ENABLE_ flags that the UI logic might attempt to access/modify
        # on current_request_config, even if MockOrchestrator's step_definitions
        # primarily uses its own mock flags. This makes current_request_config more robust.
        ENABLE_VALIDATION = False
        ENABLE_FACE_DETECTION = False
        ENABLE_LPIPS_CLUSTERING = False
        ENABLE_CROP = False
        ENABLE_TAGGING = False
        ENABLE_UPSCALE = False
        
        # Add any other critical settings Orchestrator might access via getattr(self.config, ...)
        OUTPUT_DIR = os.path.join(os.getcwd(), "output_fallback")

try:
    from config import settings
    from core.orchestrator import Orchestrator
    from core.ui_adapter import UIAdapter, LegacyOrchestrator  # 新增檔案導向架構
    from services import validator_service, upscale_service, crop_service, face_detection_service, tag_service, lpips_clustering_service
    from utils.logger_config import setup_logging as project_setup_logging_actual
    from utils.file_utils import prepare_preview_image
except ImportError as e:
    print(f"Error importing project modules in ui/app.py: {e}")
    print("Ensure you are running from the project root or have the correct PYTHONPATH.")
    
    # Use fallback settings
    settings = FallbackSettings()

    # Ensure fallback directories exist
    os.makedirs(settings.LOG_DIR, exist_ok=True)
    os.makedirs(settings.TEMP_PROCESSING_DIR, exist_ok=True)
    os.makedirs(settings.GRADIO_TEMP_DIR, exist_ok=True)
    if hasattr(settings, 'OUTPUT_DIR'):
         os.makedirs(settings.OUTPUT_DIR, exist_ok=True)

    _prepare_preview_image_fallback = None # To store prepare_preview_image if it was imported
    # If we are in this except block, the initial `from utils.file_utils import prepare_preview_image` might have failed.
    # However, the `globals()` check inside MockOrchestrator is more direct for its own scope.
    # For clarity, let's try to capture it if it was part of a partial success before the final ImportError.
    try:
        # Check if the function exists in globals()
        if 'prepare_preview_image' in globals():
            _prepare_preview_image_fallback = globals()['prepare_preview_image']
        else:
            print("Fallback: prepare_preview_image utility not available.")
    except (NameError, KeyError):
        print("Fallback: prepare_preview_image utility not available.")

    class MockOrchestrator:
        def __init__(self, config: Union[Any, 'FallbackSettings'], *args, **kwargs): # config can be module or FallbackSettings instance
            self.config = config
            self.logger = kwargs.get('logger', logging.getLogger('MockOrchestrator'))
            self.logger.warning("Using MockOrchestrator due to import error.")

            self.step_definitions = {}
            if hasattr(self.config, 'AVAILABLE_STEPS') and isinstance(self.config.AVAILABLE_STEPS, dict):
                for step_key in self.config.AVAILABLE_STEPS.keys():
                    # Flag name must exist as an attribute in self.config (FallbackSettings instance)
                    flag_name = f"ENABLE_{step_key.upper()}" # e.g., ENABLE_MOCK_FALLBACK_ACTION1
                    if not hasattr(self.config, flag_name):
                        self.logger.error(f"CRITICAL FALLBACK ERROR: FallbackSettings instance is missing expected flag {flag_name} for step {step_key}. This indicates a setup flaw in FallbackSettings. Defaulting to False.")
                        setattr(self.config, flag_name, False) # Define on the instance if missing
                    
                    self.step_definitions[step_key] = {
                        "flag": flag_name,
                        "service": None # Mock service
                    }
            else: # Minimal fallback if AVAILABLE_STEPS is somehow missing from config
                 self.logger.error("CRITICAL FALLBACK ERROR: config object has no AVAILABLE_STEPS. Using emergency mock step.")
                 self.step_definitions = {"emergency_mock": {"flag": "ENABLE_EMERGENCY_MOCK", "service": None}}
                 if not hasattr(self.config, "ENABLE_EMERGENCY_MOCK"):
                     setattr(self.config, "ENABLE_EMERGENCY_MOCK", True)

        def process_single_image(self, image_path_or_url, output_filename_prefix):
            self.logger.info(f"MockOrchestrator.process_single_image called with {image_path_or_url}, prefix: {output_filename_prefix}")
            preview_path = None
            
            enabled_steps_log = []
            # Check which mock steps are "enabled" based on the orchestrator's current config
            # (which would have been updated by the UI logic)
            for step_key, step_def_val in self.step_definitions.items():
                flag_to_check = step_def_val["flag"]
                if hasattr(self.config, flag_to_check) and getattr(self.config, flag_to_check):
                    step_display_name = self.config.AVAILABLE_STEPS.get(step_key, step_key) # Get display name
                    enabled_steps_log.append(f"'{step_display_name}' (flag: {flag_to_check})")

            message = f"Mock processing successful. "
            if enabled_steps_log:
                message += f"Enabled mock steps in current config: {', '.join(enabled_steps_log)}."
            else:
                message += "No mock steps were found enabled in current config."
            self.logger.info(message)

            if image_path_or_url and isinstance(image_path_or_url, str) and os.path.exists(image_path_or_url):
                try:
                    # Use _prepare_preview_image_fallback if available, or check globals() again
                    # The globals() check is more robust within the method's execution context.
                    if 'prepare_preview_image' in globals() and callable(globals()['prepare_preview_image']):
                        pil_image = Image.open(image_path_or_url) # Need PIL.Image for prepare_preview_image
                        # Ensure GRADIO_TEMP_DIR exists on the config object
                        temp_dir_for_preview = getattr(self.config, 'GRADIO_TEMP_DIR', 'temp_previews_fallback_inline_mock')
                        if not os.path.exists(temp_dir_for_preview):
                            os.makedirs(temp_dir_for_preview, exist_ok=True)
                        preview_path = globals()['prepare_preview_image'](pil_image, "mock_preview", temp_dir_for_preview, self.logger)
                        self.logger.info(f"MockOrchestrator: Preview image prepared at {preview_path}")
                    else:
                        self.logger.warning("MockOrchestrator: prepare_preview_image utility not found in globals. Using original path as preview.")
                        preview_path = image_path_or_url 
                except ImportError: # Catch if PIL itself is missing in extreme fallback
                    self.logger.error("MockOrchestrator: PIL.Image could not be imported for preview generation.")
                    preview_path = image_path_or_url 
                except Exception as e_preview:
                    self.logger.error(f"MockOrchestrator: Error preparing preview for {image_path_or_url}: {e_preview}", exc_info=True)
                    preview_path = image_path_or_url 
            elif image_path_or_url:
                 self.logger.warning(f"MockOrchestrator: Input path '{image_path_or_url}' does not exist or is not a string.")


            return {
                "success": True,
                "message": message,
                "processed_image_path": image_path_or_url, 
                "preview_image_path": preview_path,
                "tags": {"mock_general_tags": ["fallback_tag1", "fallback_tag2"]},
                "faces_found": False,
                "intermediate_results": {"mock_info": "Data from MockOrchestrator"}
            }

        def process_directory(self, input_dir_path, output_base_dir):
            self.logger.info(f"MockOrchestrator.process_directory called with {input_dir_path}")
            return [{"success": True, "message": "Mock directory processing complete."}]

        def process_batch(self, input_directory, output_directory, recursive=True, preserve_structure=True, selected_steps=None):
            self.logger.info(f"MockOrchestrator.process_batch called with {input_directory}")
            self.logger.info(f"Output directory: {output_directory}, Recursive: {recursive}, Preserve structure: {preserve_structure}")
            self.logger.info(f"Selected steps: {selected_steps}")
            
            # Mock batch processing result
            return {
                "success": True,
                "message": "Mock batch processing complete.",
                "total_files": 10,
                "successful_files": 8,
                "failed_files": 2,
                "success_rate": 80.0,
                "errors": ["Mock error 1", "Mock error 2"]
            }

    Orchestrator = MockOrchestrator # Fallback Orchestrator

    # Mock service modules if Orchestrator fallback needs them for instantiation (it does)
    class MockServiceModule: pass
    validator_service = MockServiceModule()
    upscale_service = MockServiceModule()
    crop_service = MockServiceModule()
    face_detection_service = MockServiceModule()
    tag_service = MockServiceModule()
    lpips_clustering_service = MockServiceModule()

    # Mock logger setup
    def project_setup_logging_actual(module_name, log_dir, log_level_str, max_bytes=0, backup_count=0):
        fb_logger = logging.getLogger(module_name)
        if not fb_logger.hasHandlers(): # Avoid adding multiple handlers on re-runs if module is reloaded
            handler = logging.StreamHandler(sys.stdout) # Log to stdout for fallback visibility
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            fb_logger.addHandler(handler)
            fb_logger.setLevel(log_level_str.upper())
        fb_logger.warning(f"Using fallback logger setup for {module_name} due to import error. Logging to console.")
        return fb_logger

ui_app_logger = None # Global logger for this module

def get_ui_app_logger():
    global ui_app_logger
    if ui_app_logger is None:
        log_dir = getattr(settings, 'LOG_DIR', 'logs')
        log_level = getattr(settings, 'LOG_LEVEL', 'INFO')
        try:
            # This will use the actual or mocked project_setup_logging_actual
            ui_app_logger = project_setup_logging_actual(
                module_name="ui_app_standalone",
                log_dir=log_dir,
                log_level_str=log_level
                # Pass max_bytes and backup_count if your setup_logging requires them
                # and they are defined in FallbackSettings or actual settings
            )
        except Exception as e:
            # Ultimate fallback to basic Python logging
            _logger = logging.getLogger("ui_app_ultimate_fallback")
            if not _logger.hasHandlers():
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                _logger.addHandler(handler)
                _logger.setLevel(log_level.upper())
            _logger.error(f"Critical error setting up logger via project_setup_logging_actual, using basic fallback: {e}")
            ui_app_logger = _logger # Assign the fallback logger
    return ui_app_logger

def handle_submit_action(processing_mode, input_image_path, selected_step_keys, 
                         input_directory, output_directory, recursive_checkbox, preserve_structure_checkbox,
                         progress=gr.Progress(track_tqdm=True)):
    logger_to_use = get_ui_app_logger()

    logger_to_use.info(f"UI: Submit button clicked. Mode: {processing_mode}")
    logger_to_use.info(f"UI: Selected step keys: {selected_step_keys}")

    app_config = settings
    preview_image_path_for_gradio = None
    batch_progress_text = ""
    tags_string = ""
    detailed_log_messages = []

    try:
        # 使用新的檔案導向架構
        use_file_based_architecture = True  # 切換開關
        
        if use_file_based_architecture:
            # 使用新的檔案導向UIAdapter
            ui_adapter = UIAdapter(
                config=app_config,
                logger=logger_to_use
            )
            logger_to_use.info("UI: Using new file-based architecture")
        else:
            # 使用舊的記憶體導向Orchestrator
            orchestrator_instance = Orchestrator(
                config=app_config,
                logger=logger_to_use
            )
            logger_to_use.info("UI: Using legacy memory-based architecture")

        # Configure enabled steps based on UI selection
        current_request_config = app_config
        
        if use_file_based_architecture:
            # 新架構：直接使用選擇的步驟列表
            # 不需要修改配置，直接傳遞給UIAdapter
            logger_to_use.info(f"UI: Selected steps for file-based processing: {selected_step_keys}")
        else:
            # 舊架構：修改配置標誌
            all_step_definitions = orchestrator_instance.step_definitions
            for step_key_internal, step_def in all_step_definitions.items():
                enable_flag_name = step_def["flag"]
                if step_key_internal in selected_step_keys:
                    setattr(current_request_config, enable_flag_name, True)
                    logger_to_use.info(f"UI: Enabling step {step_key_internal} ({enable_flag_name}=True)")
                else:
                    setattr(current_request_config, enable_flag_name, False)
                    logger_to_use.info(f"UI: Disabling step {step_key_internal} ({enable_flag_name}=False)")
            
            orchestrator_instance.config = current_request_config

        # 根據處理模式執行不同的邏輯
        if processing_mode == "single":
            # 單張圖片處理模式
            if not input_image_path:
                logger_to_use.warning("UI: No image uploaded for single mode.")
                return None, "", "錯誤：請先上傳圖片。", "請上傳圖片。"

            logger_to_use.info(f"UI: Single image processing: {input_image_path}")
            
            if use_file_based_architecture:
                # 使用新的檔案導向架構
                processing_result = ui_adapter.process_uploaded_image(
                    uploaded_file_path=input_image_path,
                    selected_steps=selected_step_keys,
                    preview_mode=True  # UI模式預設為預覽
                )
            else:
                # 使用舊的記憶體導向架構
                output_filename_prefix = "processed_image"
                if input_image_path and isinstance(input_image_path, str):
                    output_filename_prefix = os.path.splitext(os.path.basename(input_image_path))[0]

                processing_result = orchestrator_instance.process_single_image(
                    image_path_or_url=input_image_path,
                    output_filename_prefix=output_filename_prefix
                )
            
            logger_to_use.info("UI: Single image processing complete.")
            detailed_log_messages.append(f"Processing result: {processing_result.get('message', 'No message.')}")

            if processing_result["success"]:
                preview_image_path_for_gradio = processing_result.get("preview_image_path")
                if processing_result.get("tags"):
                    tags_dict = processing_result["tags"]
                    if isinstance(tags_dict, dict) and "tags" in tags_dict:
                        tags_string = tags_dict["tags"]
                    elif isinstance(tags_dict, str):
                        tags_string = tags_dict
                    else:
                        tags_string = str(tags_dict)
                    logger_to_use.info(f"UI: Tags received: {tags_string}")
            else:
                logger_to_use.error(f"UI: Single image processing failed: {processing_result.get('message')}")
                detailed_log_messages.append(f"Error: {processing_result.get('message')}")

            final_summary_message = processing_result.get("message", "Processing finished.")
            if tags_string:
                final_summary_message += f"\\n\\n偵測到的標籤：\\n{tags_string}"
            
            batch_progress_text = ""  # 單張模式不需要進度文字
            
        else:
            # 批量處理模式
            if not input_directory:
                logger_to_use.warning("UI: No input directory specified for batch mode.")
                return None, "錯誤：請指定輸入資料夾路徑。", "錯誤：請指定輸入資料夾路徑。", "請指定輸入資料夾路徑。"

            logger_to_use.info(f"UI: Batch processing: {input_directory}")
            
            # 使用預設輸出目錄如果未指定
            batch_output_dir = output_directory if output_directory else getattr(current_request_config, 'OUTPUT_DIR', 'output_images')
            
            if use_file_based_architecture:
                # 使用新的檔案導向架構
                batch_result = ui_adapter.process_batch_directory(
                    input_directory=input_directory,
                    output_directory=batch_output_dir,
                    selected_steps=selected_step_keys,
                    recursive=recursive_checkbox,
                    preserve_structure=preserve_structure_checkbox
                )
            else:
                # 使用舊的記憶體導向架構
                batch_result = orchestrator_instance.process_batch(
                    input_directory=input_directory,
                    output_directory=batch_output_dir,
                    recursive=recursive_checkbox,
                    preserve_structure=preserve_structure_checkbox,
                    selected_steps=selected_step_keys
                )
            
            logger_to_use.info("UI: Batch processing complete.")
            
            # 格式化批量處理進度文字
            if batch_result["success"]:
                batch_progress_text = (
                    f"批量處理完成\\n"
                    f"總文件數: {batch_result['total_files']}\\n"
                    f"成功處理: {batch_result['successful_files']}\\n"
                    f"失敗數量: {batch_result['failed_files']}\\n"
                    f"成功率: {batch_result.get('success_rate', 0):.1f}%"
                )
                if batch_result.get('errors'):
                    batch_progress_text += f"\\n\\n錯誤摘要:\\n" + "\\n".join(batch_result['errors'][:5])
                    if len(batch_result['errors']) > 5:
                        batch_progress_text += f"\\n... 還有 {len(batch_result['errors']) - 5} 個錯誤"
            else:
                batch_progress_text = f"批量處理失敗: {batch_result.get('message', '未知錯誤')}"
            
            final_summary_message = batch_result.get("message", "Batch processing finished.")
            detailed_log_messages.append(f"Batch result: {final_summary_message}")
            
            preview_image_path_for_gradio = None  # 批量模式不顯示預覽圖片

        detailed_log_output = "\\n".join(detailed_log_messages)
        return preview_image_path_for_gradio, batch_progress_text, final_summary_message, detailed_log_output
    except Exception as e:
        logger_to_use.error(f"UI: Error calling orchestrator or during processing: {e}", exc_info=True)
        return None, f"處理過程中發生錯誤：{e}", f"處理過程中發生錯誤：{e}", f"詳細錯誤資訊請查看日誌。\\n{traceback.format_exc()}"

def create_ui(app_logger_instance=None):
    global ui_app_logger # Still using global for module-wide access if needed elsewhere
    if app_logger_instance:
        ui_app_logger = app_logger_instance
    else:
        ui_app_logger = get_ui_app_logger() # Correctly assign the returned logger

    # Now ui_app_logger is guaranteed to be a valid logger instance
    ui_app_logger.info("UI: Creating Gradio UI.")

    custom_css_path = getattr(settings, 'CUSTOM_CSS', None)
    if custom_css_path and not os.path.exists(custom_css_path):
        ui_app_logger.warning(f"Custom CSS file not found at {custom_css_path}. Gradio will use default styling.")
        custom_css_path = None
    
    gradio_theme = getattr(settings, 'GRADIO_THEME', 'default')
    gradio_title = getattr(settings, 'GRADIO_TITLE', 'Image Processor')
    
    available_steps_dict = getattr(settings, 'AVAILABLE_STEPS', {})
    step_choices = [(label, key) for key, label in available_steps_dict.items()]
    if not step_choices:
        ui_app_logger.warning("No processing steps found in settings.AVAILABLE_STEPS. Check config/settings.py.")
        step_choices = [("No steps configured", "no_steps")]


    with gr.Blocks(theme=gradio_theme, css=custom_css_path, title=gradio_title) as demo:
        gr.Markdown(f"<h1 style='text-align: center;'>{gradio_title}</h1>")

        # 處理模式選擇
        with gr.Row():
            processing_mode = gr.Radio(
                choices=[("單張圖片處理", "single"), ("批量處理資料夾", "batch")],
                label="處理模式",
                value="single"
            )

        with gr.Row():
            with gr.Column(scale=1, min_width=400):
                # 單張圖片輸入
                with gr.Group(visible=True) as single_input_group:
                    input_image = gr.Image(type="filepath", label="上傳圖片", sources=["upload", "clipboard"])
                
                # 批量處理輸入
                with gr.Group(visible=False) as batch_input_group:
                    input_directory = gr.Textbox(label="輸入資料夾路徑", placeholder="/path/to/input/directory")
                    output_directory = gr.Textbox(label="輸出資料夾路徑", placeholder="/path/to/output/directory (留空使用預設)")
                    
                    with gr.Row():
                        recursive_checkbox = gr.Checkbox(label="遞歸處理子資料夾", value=True)
                        preserve_structure_checkbox = gr.Checkbox(label="保持原有目錄結構", value=True)
                
                gr.Markdown("### 選擇處理步驟")
                step_checkboxes = gr.CheckboxGroup(
                    choices=step_choices,
                    label="處理流程",
                    value=[] # Default to no steps selected
                )
                
                submit_button = gr.Button("開始處理", variant="primary", elem_id="submit_button_custom")

            with gr.Column(scale=1, min_width=400):
                # 單張圖片輸出
                with gr.Group(visible=True) as single_output_group:
                    output_image = gr.Image(type="filepath", label="處理結果預覽", interactive=False)
                
                # 批量處理輸出
                with gr.Group(visible=False) as batch_output_group:
                    batch_progress = gr.Textbox(label="批量處理進度", lines=3, interactive=False)
                
                # 共用輸出
                status_message = gr.Textbox(label="處理摘要與標籤", lines=5, interactive=False, max_lines=15)
                detailed_log = gr.Textbox(label="詳細日誌", lines=8, interactive=False, max_lines=20)

        # 處理模式切換的可見性控制
        def toggle_processing_mode(mode):
            if mode == "single":
                return (
                    gr.update(visible=True),   # single_input_group
                    gr.update(visible=False),  # batch_input_group
                    gr.update(visible=True),   # single_output_group
                    gr.update(visible=False)   # batch_output_group
                )
            else:  # batch
                return (
                    gr.update(visible=False),  # single_input_group
                    gr.update(visible=True),   # batch_input_group
                    gr.update(visible=False),  # single_output_group
                    gr.update(visible=True)    # batch_output_group
                )

        processing_mode.change(
            fn=toggle_processing_mode,
            inputs=[processing_mode],
            outputs=[single_input_group, batch_input_group, single_output_group, batch_output_group]
        )

        # 設定輸入參數
        all_inputs = [
            processing_mode, input_image, step_checkboxes,
            input_directory, output_directory, recursive_checkbox, preserve_structure_checkbox
        ]

        submit_button.click(
            fn=handle_submit_action,
            inputs=all_inputs,
            outputs=[output_image, batch_progress, status_message, detailed_log]
        )
    ui_app_logger.info("UI: Gradio UI creation complete.")
    return demo

if __name__ == "__main__":
    print("Starting Gradio UI directly from ui/app.py for testing...")
    logger_for_standalone = get_ui_app_logger() # Ensures logger is initialized
    logger_for_standalone.info("UI (standalone): Launching Gradio app...")

    try:
        current_settings = settings 
    except NameError: 
        logger_for_standalone.error("UI (standalone): FallbackSettings not defined when settings import failed.")
        class BasicFallback: GRADIO_SERVER_NAME, GRADIO_SERVER_PORT, GRADIO_SHARE = "0.0.0.0", 7861, False
        current_settings = BasicFallback()

    app_instance = create_ui(app_logger_instance=logger_for_standalone)
    
    server_name = getattr(current_settings, 'GRADIO_SERVER_NAME', "0.0.0.0")
    server_port = getattr(current_settings, 'GRADIO_SERVER_PORT', 7861)
    share_option = getattr(current_settings, 'GRADIO_SHARE', False)

    logger_for_standalone.info(f"UI (standalone): Launching on {server_name}:{server_port}, Share: {share_option}")
    
    app_instance.launch(
        server_name=server_name,
        server_port=server_port, 
        share=share_option
    )
    logger_for_standalone.info("UI (standalone): Gradio app finished.")
