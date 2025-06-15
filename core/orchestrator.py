# core/orchestrator.py

from PIL import Image
import os

# Import service functions
from services.validator_service import validate_image_service
from services.face_detection_service import detect_faces_service
from services.lpips_clustering_service import cluster_images_service
from services.crop_service import crop_image_service
from services.tag_service import tag_image_service
from services.upscale_service import upscale_image_service

# Import file utilities
from utils.file_utils import handle_input_path, save_processed_image, prepare_preview_image # Added prepare_preview_image
from utils.error_handler import ImageProcessingError, ConfigError # Ensure error types are available
from config import settings as default_app_settings # For default config

class Orchestrator:
    def __init__(self, config, validator_service, upscale_service, crop_service, 
                 face_detection_service, tag_service, lpips_clustering_service, logger):
        self.config = config
        self.validator_service = validator_service
        self.upscale_service = upscale_service
        self.crop_service = crop_service
        self.face_detection_service = face_detection_service
        self.tag_service = tag_service
        self.lpips_clustering_service = lpips_clustering_service
        self.logger = logger

        # Define step mapping to service calls and configuration flags
        self.step_definitions = {
            "validate": {
                "service": self.validator_service.validate_image, # Assuming service has validate_image method
                "flag": "ENABLE_VALIDATION",
                "pil_input": True, # Indicates if the service expects a PIL image as primary input
                "pil_output": False # Indicates if the service primarily returns a new PIL image
            },
            "upscale": {
                "service": self.upscale_service.upscale_image_service_entry,
                "flag": "ENABLE_UPSCALE",
                "pil_input": True,
                "pil_output": True
            },
            "crop": {
                "service": self.crop_service.crop_image_service_entry,
                "flag": "ENABLE_CROP",
                "pil_input": True,
                "pil_output": True # crop_image_service_entry returns (pil_image, output_path, message)
            },
            "face_detect": {
                "service": self.face_detection_service.detect_faces_service_entry,
                "flag": "ENABLE_FACE_DETECTION",
                "pil_input": True,
                "pil_output": True # detect_faces_service_entry returns (pil_image, output_path, faces_data, message)
            },
            "tag": {
                "service": self.tag_service.tag_image_service_entry,
                "flag": "ENABLE_TAGGING",
                "pil_input": True,
                "pil_output": False # tag_image_service_entry returns (tags_dict, message)
            },
            "cluster": { # Example, assuming cluster service might work on paths
                "service": self.lpips_clustering_service.cluster_images_service_entry, # Placeholder
                "flag": "ENABLE_LPIPS_CLUSTERING",
                "pil_input": False, # Operates on paths
                "pil_output": False
            }
        }
        self.step_execution_order = ["validate", "face_detect", "upscale", "crop", "tag", "cluster"]

    def _execute_step(self, step_key, current_pil_image, original_path, output_filename_prefix, intermediate_results):
        step_config = self.step_definitions.get(step_key)
        step_name = step_key.capitalize()
        # Use getattr to safely access the flag from self.config, defaulting to False if not found
        if not step_config or not getattr(self.config, step_config["flag"], False):
            self.logger.info(f"[Orchestrator] {step_name} step skipped (disabled, undefined, or flag not in config).")
            return current_pil_image, f"{step_name} step skipped."

        self.logger.info(f"[Orchestrator] === Executing step: {step_name} ===")
        service_call = step_config["service"]
        service_message = ""
        service_pil_output = None

        try:
            if step_key == "validate":
                # Validator service might have a different signature: (is_valid, message_or_pil_image, original_path)
                is_valid, msg_or_pil, _ = service_call(original_path, self.logger, config=self.config)
                service_message = msg_or_pil if isinstance(msg_or_pil, str) else "Validated"
                if not is_valid:
                    raise ImageProcessingError(service_message, "Validation")
                # If validation returns a PIL image (e.g. after loading), use it.
                if isinstance(msg_or_pil, Image.Image):
                    current_pil_image = msg_or_pil
            elif step_key == "tag":
                tags_dict, service_message = service_call(current_pil_image, self.logger, config=self.config)
                intermediate_results["tags"] = tags_dict
            elif step_key == "face_detect":
                # Returns: (pil_image, output_path, faces_data, message)
                output_pil, _, faces_data, service_message = service_call(current_pil_image, self.logger, config=self.config, output_path=None) # No save here
                service_pil_output = output_pil
                intermediate_results["faces_data"] = faces_data
            elif step_key == "upscale" or step_key == "crop":
                # Upscale: (pil_image, output_path, message)
                # Crop: (pil_image, output_path, message)
                output_pil, _, service_message = service_call(current_pil_image, self.logger, config=self.config, output_path=None) # No save here
                service_pil_output = output_pil
            # Add other step-specific logic here if their call signatures or return values differ significantly
            # For example, clustering might operate on a list of paths and not return a PIL image.
            elif step_key == "cluster":
                 # Assuming cluster service takes list of paths and returns some clustering info + message
                 # This is a placeholder, adjust based on actual lpips_clustering_service signature
                if original_path:
                    cluster_info, service_message = service_call([original_path], self.logger, config=self.config)
                    intermediate_results["cluster_info"] = cluster_info
                else:
                    service_message = "Clustering skipped: no original path available."
            else: # Generic call for services that take PIL, logger, config and return (PIL, msg) or similar
                if step_config["pil_input"] and not isinstance(current_pil_image, Image.Image):
                    raise ImageProcessingError(f"{step_name} requires PIL image, but none available.", step_name)
                
                # This generic part needs refinement based on actual service signatures
                # For now, assuming a common (pil_output, message) or just message if no pil_output
                if step_config["pil_output"]:
                    # Example: service_pil_output, service_message = service_call(current_pil_image, self.logger, self.config)
                    # This needs to be more specific per service if signatures vary greatly beyond the handled cases.
                    self.logger.warning(f"Generic call path for {step_name} - ensure signature matches.")
                    # Fallback to assuming it might be like upscale/crop for now if it produces PIL
                    service_pil_output, _, service_message = service_call(current_pil_image, self.logger, config=self.config, output_path=None)

                else:
                    # Example: service_message = service_call(current_pil_image, self.logger, self.config)
                    self.logger.warning(f"Generic call path for {step_name} (no PIL output) - ensure signature matches.")
                    # Fallback for services that don't return PIL but take it as input
                    _, service_message = service_call(current_pil_image, self.logger, config=self.config) 


            if service_pil_output and isinstance(service_pil_output, Image.Image):
                self.logger.info(f"Step {step_name} updated PIL image.")
                current_pil_image = service_pil_output
            
            self.logger.info(f"[Orchestrator] Step {step_name} completed. Message: {service_message}")
            return current_pil_image, service_message

        except ImageProcessingError as ipe:
            self.logger.error(f"[Orchestrator] ImageProcessingError in {step_name}: {ipe.message} (Context: {ipe.context})", exc_info=True)
            raise # Re-raise to be caught by process_single_image
        except ConfigError as ce:
            self.logger.error(f"[Orchestrator] ConfigError in {step_name}: {ce.message}", exc_info=True)
            raise
        except Exception as e:
            self.logger.error(f"[Orchestrator] Unexpected error in {step_name}: {str(e)}", exc_info=True)
            raise ImageProcessingError(f"Unexpected error in {step_name}: {str(e)}", step_name) from e

    def process_single_image(self, image_path_or_url, output_filename_prefix):
        self.logger.info(f"[Orchestrator] Starting processing for: {image_path_or_url}, Output prefix: {output_filename_prefix}")
        result = {
            "success": False,
            "message": "",
            "processed_image_path": None,
            "preview_image_path": None,
            "tags": None,
            "faces_found": False, # Default to false
            "intermediate_results": {}
        }
        current_pil_image = None
        original_input_path = image_path_or_url # Keep track of the very first input path/URL

        try:
            # 1. Load Image using file_utils.handle_input_path
            # This handles URLs and local paths, downloads if necessary.
            temp_processing_dir = getattr(self.config, 'TEMP_PROCESSING_DIR', None)
            current_pil_image = handle_input_path(image_path_or_url, self.logger, temp_dir=temp_processing_dir)
            if not current_pil_image:
                raise ImageProcessingError(f"Failed to load image from {image_path_or_url}", "Image Loading")
            
            # Ensure image is in a common mode (RGB/RGBA) after loading
            if current_pil_image.mode not in ('RGB', 'RGBA'):
                self.logger.info(f"[Orchestrator] Converting loaded image from {current_pil_image.mode} to RGB.")
                current_pil_image = current_pil_image.convert('RGB')

            # The path used for subsequent operations if it was a URL (downloaded path)
            # or the original path if local.
            # handle_input_path might return the original path or a temp path for URLs.
            # For simplicity, we'll use the original_input_path for naming, etc.,
            # but current_pil_image is what gets passed around.
            # If a more concrete path of the loaded image is needed (e.g. for services that need path),
            # handle_input_path would need to return it, or we assume it's `image_path_or_url` if local.
            # For now, we assume services primarily take PIL images if pil_input=True.
            # The `original_path` for _execute_step will be `image_path_or_url`.

            step_messages = []

            for step_key in self.step_execution_order:
                current_pil_image, msg = self._execute_step(
                    step_key, current_pil_image, image_path_or_url, 
                    output_filename_prefix, result["intermediate_results"]
                )
                step_messages.append(msg)
                if "faces_data" in result["intermediate_results"] and result["intermediate_results"]["faces_data"]:
                    result["faces_found"] = True
            
            result["message"] = "Pipeline completed. Messages: " + " | ".join(filter(None, step_messages))

            # 2. Save Processed Image using file_utils.save_processed_image
            if current_pil_image:
                output_dir = getattr(self.config, 'OUTPUT_DIR', 'output_images') # Get from config or default
                # Use output_filename_prefix for the saved file name
                saved_path = save_processed_image(current_pil_image, output_filename_prefix, output_dir, self.logger)
                if not saved_path:
                    raise ImageProcessingError("Failed to save processed image.", "Save Processed Image")
                result["processed_image_path"] = saved_path
                self.logger.info(f"[Orchestrator] Processed image saved to: {saved_path}")

                # 3. Prepare Preview Image using file_utils.prepare_preview_image
                preview_dir = getattr(self.config, 'GRADIO_TEMP_DIR', 'preview_images') # Get from config
                # Use original filename or prefix for preview name consistency
                # base_name_for_preview = os.path.basename(image_path_or_url) if not image_path_or_url.startswith('http') else output_filename_prefix + ".png" # This line is kept for context but base_name_for_preview is not used in the corrected call below
                preview_path = prepare_preview_image(pil_image_or_path=current_pil_image, logger=self.logger, temp_dir=preview_dir)
                if not preview_path:
                    self.logger.warning("[Orchestrator] Failed to generate preview image.")
                result["preview_image_path"] = preview_path
            else:
                result["message"] += " | No final image to save or preview."

            result["success"] = True
            if "tags" in result["intermediate_results"]:
                 result["tags"] = result["intermediate_results"]["tags"]
            self.logger.info(f"[Orchestrator] Successfully processed {image_path_or_url}. Output: {result.get('processed_image_path')}")

        except ImageProcessingError as ipe:
            result["message"] = f"Error processing {image_path_or_url}: {ipe.message} (Stage: {ipe.context})"
            self.logger.error(f"[Orchestrator] {result['message']}", exc_info=False) # exc_info already logged in _execute_step
        except ConfigError as ce:
            result["message"] = f"Configuration error during processing of {image_path_or_url}: {ce.message}"
            self.logger.error(f"[Orchestrator] {result['message']}", exc_info=False)
        except Exception as e:
            result["message"] = f"Unexpected error processing {image_path_or_url}: {str(e)}"
            self.logger.error(f"[Orchestrator] {result['message']}", exc_info=True)
        
        return result

    def process_directory(self, input_dir_path, output_base_dir):
        self.logger.info(f"[Orchestrator] Starting batch processing for directory: {input_dir_path}")
        results = []
        if not os.path.isdir(input_dir_path):
            msg = f"Input path {input_dir_path} is not a directory."
            self.logger.error(f"[Orchestrator] {msg}")
            results.append({"success": False, "message": msg, "input_path": input_dir_path})
            return results
        if not os.path.exists(input_dir_path):
            msg = f"Input directory {input_dir_path} does not exist."
            self.logger.error(f"[Orchestrator] {msg}")
            results.append({"success": False, "message": msg, "input_path": input_dir_path})
            return results

        for root, _, files in os.walk(input_dir_path):
            for file_name in files:
                # Construct full path of the source image
                image_file_path = os.path.join(root, file_name)
                
                # Determine output filename prefix (filename without extension)
                output_filename_prefix = os.path.splitext(file_name)[0]
                
                # Create a unique output subdirectory for this image based on its original subfolder structure
                # relative_path = os.path.relpath(root, input_dir_path)
                # current_output_dir = os.path.join(output_base_dir, relative_path)
                # os.makedirs(current_output_dir, exist_ok=True) # save_processed_image will handle dir creation

                self.logger.info(f"[Orchestrator] Processing file in directory: {image_file_path}")
                # Pass output_base_dir to process_single_image, or let save_processed_image use configured OUTPUT_DIR
                # For batch, it's common to specify an output_base_dir and then construct subpaths.
                # Here, process_single_image uses self.config.OUTPUT_DIR. If that needs to change per image in batch,
                # this logic needs adjustment, or config needs to be dynamic.
                # For now, all outputs from batch will go into the global OUTPUT_DIR but with original filename as prefix.
                
                # The `output_filename_prefix` is used by `save_processed_image` to name the file.
                # The `output_dir` for `save_processed_image` is taken from `self.config.OUTPUT_DIR`.
                # If you want batch processed files to go into `output_base_dir` subfolders,
                # you'd need to modify `self.config.OUTPUT_DIR` before each call or modify `save_processed_image`.
                # Simpler: ensure `self.config.OUTPUT_DIR` is set to `output_base_dir` before calling `process_directory`.
                # This is usually handled by the caller setting the global config appropriately.

                result = self.process_single_image(image_file_path, output_filename_prefix)
                result["input_path"] = image_file_path # Add input path for context in results
                results.append(result)
        
        self.logger.info(f"[Orchestrator] Finished batch processing for directory: {input_dir_path}. Processed {len(results)} files.")
        return results

# Remove or refactor the old process_image_pipeline function if Orchestrator class replaces its functionality.
# For now, keeping it separate. If the UI layer or other scripts use Orchestrator directly,
# then process_image_pipeline might become obsolete or an internal helper.
# ... (rest of the existing process_image_pipeline function, if it's still needed)
# If process_image_pipeline is to be removed, delete from here to the end of the file.

# The existing process_image_pipeline function is below. 
# It seems the Orchestrator class is intended to be the new primary way of handling processing.
# Consider deprecating or integrating process_image_pipeline into the Orchestrator class methods.

# core/orchestrator.py

# from PIL import Image # Already imported
# import os # Already imported

# # Import service functions
# from services.validator_service import validate_image_service # Already imported
# from services.face_detection_service import detect_faces_service # Already imported
# from services.lpips_clustering_service import cluster_images_service # Already imported
# from services.crop_service import crop_image_service # Already imported
# from services.tag_service import tag_image_service # Already imported
# from services.upscale_service import upscale_image_service # Already imported

# # Import file utilities
# from utils.file_utils import handle_input_path, save_processed_image # Already imported

# Note: Orchestrator uses the logger instance passed to it by the caller (e.g., UI layer or main script)
# and app_settings from config.settings, also passed by the caller.

def process_image_pipeline(image_upload_path, selected_step_keys, logger, app_settings):
    logger.info(f"[Orchestrator] Starting image processing pipeline for: {image_upload_path}")
    logger.info(f"[Orchestrator] Selected steps: {selected_step_keys}")

    processed_image_pil = None
    intermediate_data = {}
    
    pipeline_messages = []
    detailed_log_entries = []
    final_saved_image_path = None # Will store the path of the finally saved image

    # --- Initial Image Loading using handle_input_path ---
    initial_image_path_for_processing = image_upload_path # Default for single file/URL
    
    # If the input is a directory, we first find a specific image file to process.
    # handle_input_path itself does not iterate directories.
    if os.path.isdir(image_upload_path):
        logger.info(f"[Orchestrator] Input is a directory: {image_upload_path}. Attempting to find first valid image.")
        _is_valid_dir, dir_msg, valid_image_paths = validate_image_service(
            image_upload_path, 
            logger,
            config=app_settings, 
            is_directory=True
        )
        pipeline_messages.append(f"Directory Scan: {dir_msg}")
        detailed_log_entries.append(f"[ValidatorService-DirScan] {dir_msg}")

        if not _is_valid_dir or not valid_image_paths:
            logger.error(f"[Orchestrator] Directory scan failed or no valid images found: {dir_msg}")
            return None, "\\n".join(pipeline_messages), "\\n".join(detailed_log_entries), None
        
        initial_image_path_for_processing = valid_image_paths[0] # Process first valid image
        logger.info(f"[Orchestrator] Proceeding with first valid image from directory: {initial_image_path_for_processing}")
    
    # Now, use handle_input_path to load the determined image path (file or URL)
    logger.info(f"[Orchestrator] Attempting to load image using handle_input_path for: {initial_image_path_for_processing}")
    processed_image_pil = handle_input_path(
        initial_image_path_for_processing, 
        logger,
        temp_dir=getattr(app_settings, 'TEMP_PROCESSING_DIR', None) # Use configured temp dir
    )

    if processed_image_pil is None:
        msg = f"Failed to load image using handle_input_path for: {initial_image_path_for_processing}."
        logger.error(f"[Orchestrator] {msg}")
        pipeline_messages.append(msg)
        detailed_log_entries.append(f"[Orchestrator] Image Load Error (handle_input_path): {msg}")
        return None, "\\n".join(pipeline_messages), "\\n".join(detailed_log_entries), None

    logger.info(f"[Orchestrator] Image loaded successfully via handle_input_path: {initial_image_path_for_processing} (Mode: {processed_image_pil.mode})")
    intermediate_data['original_path'] = initial_image_path_for_processing # Store the path that was actually loaded
    intermediate_data['current_pil_image'] = processed_image_pil
    
    # Ensure it's in a common mode like RGB or RGBA, as some services might expect this.
    if processed_image_pil.mode not in ('RGB', 'RGBA'):
         logger.info(f"[Orchestrator] Converting image from {processed_image_pil.mode} to RGB.")
         processed_image_pil = processed_image_pil.convert('RGB')
         intermediate_data['current_pil_image'] = processed_image_pil


    # --- Define step execution order and process selected steps ---
    step_execution_order = ["validate", "face_detect", "cluster", "crop", "tag", "upscale"]
    final_tags_string = ""

    for step_key in step_execution_order:
        if step_key in selected_step_keys:
            step_name = app_settings.AVAILABLE_STEPS.get(step_key, step_key)
            logger.info(f"[Orchestrator] === Executing step: {step_name} ({step_key}) ===")
            
            current_pil_for_step = intermediate_data.get('current_pil_image')
            
            detailed_log_entries.append(f"-- Attempting step: {step_name} on image (Original: {intermediate_data.get('original_path', 'N/A')}) --")
            
            # Crucial check: Ensure current_pil_for_step is a valid PIL Image before proceeding to most services
            if not isinstance(current_pil_for_step, Image.Image):
                if step_key == "cluster": 
                    logger.info(f"[Orchestrator] Step {step_name} does not strictly require a PIL image if it operates on paths. Proceeding cautiously.")
                else:
                    err_msg = f"Step {step_name} requires a valid PIL image, but none is available (current_pil_for_step is {type(current_pil_for_step)}). Skipping step."
                    logger.error(f"[Orchestrator] {err_msg}")
                    pipeline_messages.append(err_msg)
                    detailed_log_entries.append(f"[{step_key.capitalize()}Service] Skipped: No valid PIL image.")
                    continue 
            
            try:
                service_result_image = None 
                service_message = ""

                if step_key == "validate":
                    if not isinstance(current_pil_for_step, Image.Image):
                        service_message = "Validation skipped: No valid PIL image available for validation step."
                        logger.error(f"[Orchestrator] {service_message}")
                    else:
                        is_valid_pil, msg, _ = validate_image_service(
                            current_pil_for_step, 
                            logger,
                            config=app_settings,
                            is_directory=False
                        )
                        service_message = msg
                        if not is_valid_pil:
                            logger.error(f"[Orchestrator] Image content validation failed for {intermediate_data.get('original_path')}: {msg}")
                            pipeline_messages.append(f"Validation Failed: {msg}")
                            detailed_log_entries.append(f"[ValidatorService-PIL] {msg} - Pipeline Halted.")
                            return None, "\\n".join(pipeline_messages), "\\n".join(detailed_log_entries), None 

                elif step_key == "face_detect":
                    if not isinstance(current_pil_for_step, Image.Image):
                        service_message = "Face detection skipped: No valid PIL image available."
                        logger.error(f"[Orchestrator] {service_message}")
                    else:
                        img_out, msg, detection_data = detect_faces_service(current_pil_for_step, logger, config=app_settings)
                        service_result_image = img_out 
                        service_message = msg
                        intermediate_data['face_detection_results'] = detection_data
                        logger.info(f"[Orchestrator] Face detection data stored: {detection_data}")

                elif step_key == "cluster":
                    # Cluster service might not need a PIL image if it works on paths
                    img_path_for_cluster = intermediate_data.get('original_path')
                    if not img_path_for_cluster or not os.path.exists(img_path_for_cluster):
                        service_message = "Clustering skipped: Valid image path for current image not available."
                        logger.error(f"[Orchestrator] {service_message}")
                    else:
                        _, service_message = cluster_images_service([img_path_for_cluster], logger, config=app_settings)

                elif step_key == "crop":
                    if not isinstance(current_pil_for_step, Image.Image):
                        service_message = "Crop skipped: No valid PIL image available."
                        logger.error(f"[Orchestrator] {service_message}")
                    else:
                        crop_input_data = intermediate_data.get('face_detection_results')
                        cropped_images_data_list, msg = crop_image_service(current_pil_for_step, crop_data=crop_input_data, logger=logger, config=app_settings)
                        service_message = msg
                        if cropped_images_data_list:
                            first_crop_data = cropped_images_data_list[0]
                            if isinstance(first_crop_data, dict) and 'image' in first_crop_data and isinstance(first_crop_data['image'], Image.Image):
                                service_result_image = first_crop_data['image']
                                logger.info(f"[Orchestrator] Using cropped image of type '{first_crop_data.get('type', 'unknown')}' for further processing.")
                            else:
                                logger.warning(f"[Orchestrator] Crop service returned data for the first crop, but it was not in the expected format. Using previous image.")
                            
                            if len(cropped_images_data_list) > 1:
                                logger.info(f"[Orchestrator] Crop service returned {len(cropped_images_data_list)} images/crops. Using the first one.")
                                service_message += f" (Used first of {len(cropped_images_data_list)} crops)"
                        else:
                            logger.warning("[Orchestrator] Crop service did not return any cropped images. Using previous image.")
                
                elif step_key == "tag":
                    if not isinstance(current_pil_for_step, Image.Image):
                        service_message = "Tagging skipped: No valid PIL image available."
                        logger.error(f"[Orchestrator] {service_message}")
                    else:
                        tags_string, msg = tag_image_service(current_pil_for_step, logger, config=app_settings)
                        service_message = msg
                        final_tags_string = tags_string

                elif step_key == "upscale":
                    if not isinstance(current_pil_for_step, Image.Image):
                        service_message = "Upscale skipped: No valid PIL image available."
                        logger.error(f"[Orchestrator] {service_message}")
                    else:
                        img_out, msg = upscale_image_service(current_pil_for_step, logger, config=app_settings)
                        service_result_image = img_out
                        service_message = msg

                # Common handling after a step execution
                pipeline_messages.append(f"{step_name}: {service_message}")
                detailed_log_entries.append(f"[{step_key.capitalize()}Service] {service_message}")

                if service_result_image and isinstance(service_result_image, Image.Image):
                    logger.info(f"[Orchestrator] Step {step_name} resulted in an updated PIL image.")
                    intermediate_data['current_pil_image'] = service_result_image
                else:
                    logger.info(f"[Orchestrator] Step {step_name} did not return a new PIL image, or returned an unexpected type. Continuing with previous image state.")
                    # Ensure 'current_pil_image' still holds the valid image from before this step
                    if not isinstance(intermediate_data.get('current_pil_image'), Image.Image):
                        # This case should ideally not happen if initial load was successful
                        logger.error("[Orchestrator] Critical error: current_pil_image in intermediate_data is not a PIL image after a step that didn't return one.")
                        # Halt or handle error appropriately
                        # For now, we'll let it proceed, but this indicates a logic flaw if it occurs.


            except Exception as e:
                err_msg = f"Error during step {step_name}: {e}"
                logger.error(f"[Orchestrator] {err_msg}", exc_info=True)
                pipeline_messages.append(err_msg)
                detailed_log_entries.append(f"[{step_key.capitalize()}Service] Error: {err_msg}")
                # Optionally, decide if pipeline should halt on any error or continue
                # For now, we continue to the next step if possible, but the image might be in an inconsistent state.
                # Consider adding a flag to halt pipeline on error.

    # --- End of pipeline: Save the final processed image ---
    final_pil_image_to_save = intermediate_data.get('current_pil_image')
    
    if final_pil_image_to_save and isinstance(final_pil_image_to_save, Image.Image):
        original_filename_base = os.path.basename(intermediate_data.get('original_path', 'processed_image.png'))
        output_directory = getattr(app_settings, 'OUTPUT_DIR', os.path.join(app_settings.BASE_DIR, 'output_images'))
        
        logger.info(f"[Orchestrator] Attempting to save final processed image. Original base: {original_filename_base}, Output dir: {output_directory}")
        
        final_saved_image_path = save_processed_image(
            pil_image=final_pil_image_to_save,
            original_filename_or_base=original_filename_base,
            output_dir=output_directory,
            logger=logger
            # new_suffix can be customized if needed, defaults to "_processed"
        )
        
        if final_saved_image_path:
            msg = f"Final image successfully saved to: {final_saved_image_path}"
            logger.info(f"[Orchestrator] {msg}")
            pipeline_messages.append(msg)
            detailed_log_entries.append(f"[Orchestrator] Save Success: {msg}")
        else:
            msg = f"Failed to save the final processed image."
            logger.error(f"[Orchestrator] {msg}")
            pipeline_messages.append(msg)
            detailed_log_entries.append(f"[Orchestrator] Save Failed: {msg}")
            # Even if saving fails, we might still have tags and logs to return.
            # The UI will get None for the image path.
    else:
        logger.warning("[Orchestrator] No final PIL image available to save, or it's not a PIL Image object.")
        pipeline_messages.append("No final image was produced or available to save.")
        detailed_log_entries.append("[Orchestrator] No final image saved.")

    pipeline_summary_str = "\\n".join(pipeline_messages)
    detailed_log_str = "\\n".join(detailed_log_entries)
    
    logger.info(f"[Orchestrator] Pipeline finished. Returning saved path: {final_saved_image_path}, Summary: {len(pipeline_summary_str)} chars, Logs: {len(detailed_log_str)} chars, Tags: '{final_tags_string}'")
    
    return final_saved_image_path, pipeline_summary_str, detailed_log_str, final_tags_string
