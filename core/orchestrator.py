# core/orchestrator.py
import os
import logging
from typing import Optional
from PIL import Image
from services import validator_service, face_detection_service, crop_service, tag_service, upscale_service, lpips_clustering_service
from utils.file_utils import handle_input_path, save_processed_image, scan_directory_for_images, get_relative_output_path, safe_move_file
from utils.error_handler import safe_execute, ConfigError
from config import settings as default_settings

class Orchestrator:
    def __init__(self, config=None, logger: Optional[logging.Logger] = None):
        self.config = config if config else default_settings
        
        # 如果沒有提供 logger，創建一個默認的
        if logger is None:
            self.logger = logging.getLogger(__name__)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)
        else:
            self.logger = logger
        
        # Import service modules
        self.validator_service = validator_service
        self.face_detection_service = face_detection_service
        self.crop_service = crop_service
        self.tag_service = tag_service
        self.upscale_service = upscale_service
        self.lpips_clustering_service = lpips_clustering_service

        # Define step mapping to service calls and configuration flags
        self.step_definitions = {
            "validate": {
                "service": self.validator_service.validate_image,
                "flag": "ENABLE_VALIDATION",
                "pil_input": True,
                "pil_output": False
            },
            "upscale": {
                "service": self.upscale_service.upscale_image_service_entry_for_orchestrator,
                "flag": "ENABLE_UPSCALE",
                "pil_input": True,
                "pil_output": True
            },
            "crop": {
                "service": self.crop_service.crop_image_service_entry,
                "flag": "ENABLE_CROP",
                "pil_input": True,
                "pil_output": True
            },
            "face_detect": {
                "service": self.face_detection_service.detect_faces_service_entry,
                "flag": "ENABLE_FACE_DETECTION",
                "pil_input": True,
                "pil_output": True
            },
            "tag": {
                "service": self.tag_service.tag_image_service_entry,
                "flag": "ENABLE_TAGGING",
                "pil_input": True,
                "pil_output": False
            },
            "cluster": {
                "service": self.lpips_clustering_service.cluster_images_service_entry,
                "flag": "ENABLE_LPIPS_CLUSTERING",
                "pil_input": False,
                "pil_output": False
            }
        }
        self.step_execution_order = ["validate", "face_detect", "upscale", "crop", "tag", "cluster"]

    def _execute_step(self, step_key, current_pil_image, original_path, output_filename_prefix, intermediate_results):
        step_config = self.step_definitions.get(step_key)
        step_name = step_key.capitalize()
        
        if not step_config or not getattr(self.config, step_config["flag"], False):
            self.logger.info(f"[Orchestrator] {step_name} step skipped (disabled, undefined, or flag not in config).")
            return current_pil_image, f"{step_name} step skipped."

        service_function = step_config["service"]
        requires_pil_input = step_config["pil_input"]
        produces_pil_output = step_config["pil_output"]

        if requires_pil_input and not isinstance(current_pil_image, Image.Image):
            error_msg = f"{step_name} requires PIL image input but received {type(current_pil_image)}"
            self.logger.error(f"[Orchestrator] {error_msg}")
            return current_pil_image, error_msg

        try:
            if step_key == "validate":
                is_valid, result_or_message, path_list = service_function(current_pil_image, self.logger, self.config)
                if is_valid:
                    return result_or_message, f"{step_name} passed"
                else:
                    return current_pil_image, f"{step_name} failed: {result_or_message}"

            elif step_key in ["upscale", "crop"]:
                result_image, output_path, message = service_function(current_pil_image, self.logger, self.config)
                intermediate_results[f"{step_key}_output_path"] = output_path
                return result_image, message

            elif step_key == "face_detect":
                result_image, output_path, faces_data, message = service_function(current_pil_image, self.logger, self.config)
                intermediate_results["faces_data"] = faces_data
                intermediate_results[f"{step_key}_output_path"] = output_path
                return result_image, message

            elif step_key == "tag":
                tags_dict, message = service_function(current_pil_image, self.logger, self.config)
                intermediate_results["tags"] = tags_dict
                return current_pil_image, message

            elif step_key == "cluster":
                if not original_path:
                    return current_pil_image, "Clustering requires original path"
                results, message = service_function([original_path], self.logger, self.config)
                intermediate_results["cluster_results"] = results
                return current_pil_image, message

            else:
                return current_pil_image, f"Unknown step: {step_key}"

        except Exception as e:
            error_msg = f"{step_name} failed with exception: {str(e)}"
            self.logger.error(f"[Orchestrator] {error_msg}", exc_info=True)
            return current_pil_image, error_msg

    def process_image(self, image_path, selected_steps=None):
        """
        Process an image through the selected pipeline steps.
        """
        if selected_steps is None:
            selected_steps = self.step_execution_order

        self.logger.info(f"[Orchestrator] Starting processing for: {image_path}")
        self.logger.info(f"[Orchestrator] Selected steps: {selected_steps}")

        # Load initial image
        try:
            if os.path.isdir(image_path):
                # Handle directory input - find first valid image
                is_valid, message, valid_paths = self.validator_service.validate_image_service(
                    image_path, self.logger, self.config, is_directory=True
                )
                if not is_valid or not valid_paths:
                    return None, message, "", ""
                image_path = valid_paths[0]

            current_pil_image = handle_input_path(image_path, self.logger)
            if current_pil_image is None:
                return None, f"Failed to load image: {image_path}", "", ""

        except Exception as e:
            error_msg = f"Failed to load image: {str(e)}"
            self.logger.error(f"[Orchestrator] {error_msg}", exc_info=True)
            return None, error_msg, "", ""

        # Process through pipeline
        intermediate_results = {}
        pipeline_messages = []
        
        for step_key in self.step_execution_order:
            if step_key in selected_steps:
                current_pil_image, step_message = self._execute_step(
                    step_key, current_pil_image, image_path, 
                    f"processed_{os.path.basename(image_path)}", 
                    intermediate_results
                )
                pipeline_messages.append(f"{step_key.capitalize()}: {step_message}")        # Save final result
        final_path = None
        if isinstance(current_pil_image, Image.Image):
            final_path = save_processed_image(
                current_pil_image,
                f"processed_{os.path.splitext(os.path.basename(image_path))[0]}",
                getattr(self.config, 'OUTPUT_DIR', 'output_images'),
                self.logger
            )

        pipeline_summary = "\n".join(pipeline_messages)
        tags_string = ""
        if "tags" in intermediate_results:
            tags_data = intermediate_results["tags"]
            if isinstance(tags_data, dict):
                tags_string = tags_data.get("tags", "")
            else:
                tags_string = str(tags_data)

        return final_path, pipeline_summary, pipeline_summary, tags_string

    def process_batch(self, input_directory, output_directory=None, recursive=True, preserve_structure=True, selected_steps=None):
        """
        批量處理資料夾中的所有圖片。
        
        Args:
            input_directory (str): 輸入目錄路徑
            output_directory (str): 輸出目錄路徑，預設為 config.OUTPUT_DIR
            recursive (bool): 是否遞歸處理子目錄
            preserve_structure (bool): 是否保持原有目錄結構
            selected_steps (list): 要執行的處理步驟，預設為所有啟用的步驟
            
        Returns:
            dict: 批量處理結果統計
        """
        if output_directory is None:
            output_directory = getattr(self.config, 'OUTPUT_DIR', 'output_images')
        
        if selected_steps is None:
            selected_steps = []
            for step_key, step_config in self.step_definitions.items():
                flag_name = step_config["flag"]
                if getattr(self.config, flag_name, False):
                    selected_steps.append(step_key)

        self.logger.info(f"[Orchestrator] Starting batch processing")
        self.logger.info(f"[Orchestrator] Input directory: {input_directory}")
        self.logger.info(f"[Orchestrator] Output directory: {output_directory}")
        self.logger.info(f"[Orchestrator] Recursive: {recursive}")
        self.logger.info(f"[Orchestrator] Preserve structure: {preserve_structure}")
        self.logger.info(f"[Orchestrator] Selected steps: {selected_steps}")

        # 掃描所有圖片文件
        image_files = scan_directory_for_images(input_directory, recursive=recursive)
        
        if not image_files:
            self.logger.warning(f"[Orchestrator] No image files found in {input_directory}")
            return {
                "success": False,
                "message": f"No image files found in {input_directory}",
                "total_files": 0,
                "processed_files": 0,
                "successful_files": 0,
                "failed_files": 0,
                "errors": []
            }

        self.logger.info(f"[Orchestrator] Found {len(image_files)} image files")
        
        # 處理統計
        total_files = len(image_files)
        processed_files = 0
        successful_files = 0
        failed_files = 0
        errors = []
        
        # 確保輸出目錄存在
        os.makedirs(output_directory, exist_ok=True)
        
        # 批量處理每個圖片
        for i, image_file in enumerate(image_files, 1):
            try:
                self.logger.info(f"[Orchestrator] Processing file {i}/{total_files}: {image_file}")
                
                # 計算輸出路徑
                if preserve_structure:
                    output_dir = get_relative_output_path(image_file, input_directory, output_directory)
                else:
                    output_dir = output_directory
                  # 處理單個圖片
                final_path, pipeline_summary, detailed_log, tags_string = self.process_image(
                    image_file, selected_steps
                )
                
                if final_path:
                    # 如果需要保持結構，移動文件到正確位置
                    if preserve_structure and os.path.dirname(final_path) != output_dir:
                        filename = os.path.basename(final_path)
                        new_final_path = os.path.join(output_dir, filename)
                        os.makedirs(output_dir, exist_ok=True)
                        if os.path.exists(final_path):
                            moved_path = safe_move_file(final_path, new_final_path, self.logger, overwrite=True)
                            if moved_path:
                                final_path = moved_path
                            else:
                                # 移動失敗，但處理成功，記錄警告
                                self.logger.warning(f"[Orchestrator] Failed to move file to final location, keeping at: {final_path}")
                    
                    successful_files += 1
                    self.logger.info(f"[Orchestrator] Successfully processed: {image_file} -> {final_path}")
                else:
                    failed_files += 1
                    error_msg = f"Failed to process {image_file}: {pipeline_summary}"
                    errors.append(error_msg)
                    self.logger.error(f"[Orchestrator] {error_msg}")
                
                processed_files += 1
                
            except Exception as e:
                failed_files += 1
                error_msg = f"Exception processing {image_file}: {str(e)}"
                errors.append(error_msg)
                self.logger.error(f"[Orchestrator] {error_msg}", exc_info=True)
                processed_files += 1

        # 返回處理結果
        success_rate = (successful_files / total_files * 100) if total_files > 0 else 0
        
        result = {
            "success": successful_files > 0,
            "message": f"Batch processing completed. {successful_files}/{total_files} files processed successfully ({success_rate:.1f}%)",
            "total_files": total_files,
            "processed_files": processed_files,
            "successful_files": successful_files,
            "failed_files": failed_files,
            "success_rate": success_rate,
            "errors": errors
        }
        
        self.logger.info(f"[Orchestrator] Batch processing completed: {result['message']}")
        return result

    def process_single_image(self, image_path_or_url, output_filename_prefix=None):
        """
        Legacy compatibility method for UI integration.
        Returns a dictionary with processing results.
        """
        try:
            # Select steps based on configuration flags
            selected_steps = []
            for step_key, step_config in self.step_definitions.items():
                flag_name = step_config["flag"]
                if getattr(self.config, flag_name, False):
                    selected_steps.append(step_key)
            
            self.logger.info(f"[Orchestrator] Enabled steps based on config: {selected_steps}")
            
            final_path, pipeline_summary, detailed_log, tags_string = self.process_image(
                image_path_or_url, selected_steps
            )
            
            # Prepare preview image path for UI
            preview_image_path = None
            if final_path and os.path.exists(final_path):
                # Use the final path as preview
                preview_image_path = final_path
            
            return {
                "success": True,
                "final_image_path": final_path,
                "preview_image_path": preview_image_path,
                "message": "Processing completed successfully",
                "pipeline_summary": pipeline_summary,
                "detailed_log": detailed_log,
                "tags": tags_string
            }
            
        except Exception as e:
            self.logger.error(f"[Orchestrator] process_single_image failed: {e}", exc_info=True)
            return {
                "success": False,
                "final_image_path": None,
                "preview_image_path": None,
                "message": f"Processing failed: {str(e)}",
                "pipeline_summary": f"Error: {str(e)}",
                "detailed_log": f"Exception: {str(e)}",
                "tags": ""
            }


# Legacy imports and utilities for backward compatibility
from utils.file_utils import handle_input_path, save_processed_image

# Note: Orchestrator uses the logger instance passed to it by the caller (e.g., UI layer or main script)
# and app_settings from config.settings, also passed by the caller.