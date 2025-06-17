# services/crop_service.py
import os
import tempfile
import shutil
from PIL import Image
from waifuc.action import ThreeStageSplitAction
from waifuc.export import SaveExporter
from waifuc.source import LocalSource
from utils.error_handler import safe_execute

# Expect logger and config to be passed.

def _safe_waifuc_process_internal(pil_image: Image.Image, logger, config=None):
    """
    Internal wrapper for waifuc processing using a temporary directory.
    Takes a PIL image, saves it temporarily, processes it with waifuc,
    and returns a list of PIL images from the waifuc output.
    """
    temp_input_dir = None
    temp_output_dir = None
    cropped_images_pil = []
    status_message = ""

    try:
        # Create temporary directories for waifuc input and output
        temp_input_dir = tempfile.mkdtemp()
        temp_output_dir = tempfile.mkdtemp()
        logger.debug(f"[CropService-Internal] Created temp dirs: Input '{temp_input_dir}', Output '{temp_output_dir}'")

        # Save the input PIL image to the temporary input directory
        # Waifuc typically operates on files, so we need to save the PIL image first.
        # Give it a generic name as its original filename isn't directly available here.
        temp_image_filename = "temp_input_image.png" # Assuming PNG, adjust if needed or get format from PIL
        temp_image_path = os.path.join(temp_input_dir, temp_image_filename)
        pil_image.save(temp_image_path)
        logger.debug(f"[CropService-Internal] Saved input PIL image to '{temp_image_path}'")

        # Configure and run waifuc processing
        source = LocalSource(temp_input_dir) # Source is the directory containing the temp image
        # The ThreeStageSplitAction will find characters/subjects and crop them.
        # It saves outputs with suffixes like _person1_head, _person1_halfbody, etc.
        source.attach(ThreeStageSplitAction()).export(SaveExporter(temp_output_dir))
        logger.info(f"[CropService-Internal] Waifuc ThreeStageSplitAction completed on '{temp_input_dir}' to '{temp_output_dir}'")

        # Collect cropped images from the temporary output directory
        output_files = os.listdir(temp_output_dir)
        if not output_files:
            status_message = "Waifuc processing ran but produced no output files."
            logger.warning(f"[CropService-Internal] {status_message}")
            # Return empty list, message will indicate no output
        else:
            for filename in output_files:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')):
                    file_path = os.path.join(temp_output_dir, filename)
                    try:
                        img = Image.open(file_path)
                        # We might want to associate the type of crop (head, halfbody, person)
                        # with the image. This can be inferred from the filename suffix.
                        crop_type = "unknown"
                        if "_head" in filename:
                            crop_type = "head"
                        elif "_halfbody" in filename:
                            crop_type = "halfbody"
                        elif "_person" in filename: # General person crop if not more specific
                            crop_type = "person"
                        
                        cropped_images_pil.append({"image": img.copy(), "type": crop_type, "original_filename": filename})
                        logger.debug(f"[CropService-Internal] Loaded cropped image: {filename} (type: {crop_type})")
                        img.close() # Close file handle after copying
                    except Exception as e_load:
                        logger.error(f"[CropService-Internal] Failed to load cropped image '{file_path}': {e_load}")
            status_message = f"Successfully processed and loaded {len(cropped_images_pil)} cropped image(s)."

        return cropped_images_pil, status_message

    except Exception as e:
        logger.error(f"[CropService-Internal] Error during waifuc processing: {e}", exc_info=True)
        # Re-raise to be caught by safe_execute, which will log it with prefix
        raise
    finally:
        # Clean up temporary directories
        try:
            if temp_input_dir and os.path.exists(temp_input_dir):
                shutil.rmtree(temp_input_dir)
                logger.debug(f"[CropService-Internal] Cleaned up temp input dir: {temp_input_dir}")
            if temp_output_dir and os.path.exists(temp_output_dir):
                shutil.rmtree(temp_output_dir)
                logger.debug(f"[CropService-Internal] Cleaned up temp output dir: {temp_output_dir}")
        except Exception as e_cleanup:
            logger.error(f"[CropService-Internal] Error cleaning up temp directories: {e_cleanup}")

def crop_image_service(image_pil: Image.Image, crop_data, logger, config=None):
    """
    Service to crop an image using waifuc's ThreeStageSplitAction.
    `crop_data` is currently not used by ThreeStageSplitAction but is kept for API consistency.
    """
    logger.info(f"[CropService] Starting image cropping for provided PIL image.")

    if not isinstance(image_pil, Image.Image):
        logger.error("[CropService] Input is not a PIL Image object.")
        return None, "Error: Input is not a valid PIL Image."
    
    # crop_data might be from face_detection (e.g., bounding boxes).
    # However, waifuc's ThreeStageSplitAction does its own detection and splitting.
    # If specific bounding boxes were to be used, a different waifuc action or a custom PIL crop would be needed.
    # For now, we ignore crop_data as ThreeStageSplitAction is autonomous.
    if crop_data:
        logger.info(f"[CropService] Received crop_data, but ThreeStageSplitAction performs its own detection. crop_data will be ignored for this action.")
    
    # Use safe_execute to run the internal waifuc processing function
    # The result from _safe_waifuc_process_internal is (list_of_pil_images, status_message)
    # safe_execute will return this tuple, or default_return if an exception occurs within _safe_waifuc_process_internal
    result_tuple = safe_execute(
        _safe_waifuc_process_internal,
        image_pil, logger, config,
        logger=logger,
        default_return=(None, "Cropping failed due to an internal error."),
        error_msg_prefix="[CropService] Error during waifuc image processing"
    )
    
    # Unpack the tuple from safe_execute or _safe_waifuc_process_internal
    # If _safe_waifuc_process_internal failed and raised, safe_execute returns default_return
    # If _safe_waifuc_process_internal succeeded, it returns (list_of_images, message_from_internal)

    cropped_images_data, message = result_tuple

    if cropped_images_data is None:
        # This means safe_execute caught an exception from _safe_waifuc_process_internal
        # and returned its default_return. The error was already logged by safe_execute.
        logger.error(f"[CropService] Cropping returned no image data. Message: {message}")
        return None, message # Message here is from safe_execute's default_return

    if not cropped_images_data: # Empty list, but not None
        logger.info(f"[CropService] Cropping completed, but no crops were generated (or loaded). Message: {message}")
        # This could be a valid scenario if the image had no detectable persons/parts by ThreeStageSplitAction.
        # Message is from _safe_waifuc_process_internal
        return [], message 

    logger.info(f"[CropService] Cropping successful. Produced {len(cropped_images_data)} image(s). Message: {message}")
    # The main return for success is a list of dicts: {"image": PIL.Image, "type": str, "original_filename": str}
    return cropped_images_data, message

def save_crops_to_categories(cropped_images_data, base_output_dir, original_filename, logger, config=None):
    """
    將裁切後的圖片按類型分類存儲到不同資料夾
    """
    try:
        saved_paths = {}
        
        for crop_data in cropped_images_data:
            crop_image = crop_data["image"]
            crop_type = crop_data["type"]
            
            # 創建分類資料夾
            category_dir = os.path.join(base_output_dir, f"crop_{crop_type}")
            os.makedirs(category_dir, exist_ok=True)
            
            # 生成文件名
            base_name = os.path.splitext(original_filename)[0]
            ext = os.path.splitext(original_filename)[1] or '.png'
            save_filename = f"{base_name}_{crop_type}{ext}"
            save_path = os.path.join(category_dir, save_filename)
            
            # 處理重名文件
            if os.path.exists(save_path):
                counter = 1
                while os.path.exists(save_path):
                    save_filename = f"{base_name}_{crop_type}_{counter}{ext}"
                    save_path = os.path.join(category_dir, save_filename)
                    counter += 1
            
            # 保存圖片
            crop_image.save(save_path)
            
            if crop_type not in saved_paths:
                saved_paths[crop_type] = []
            saved_paths[crop_type].append(save_path)
            
            logger.info(f"[CropService] Saved {crop_type} crop to: {save_path}")
        
        return saved_paths
        
    except Exception as e:
        logger.error(f"[CropService] Error saving crops to categories: {e}")
        return {}

def crop_batch_with_categorization(input_directory, output_directory, logger, config=None):
    """
    批量裁切圖片並按類型分類
    """
    logger.info(f"[CropService] Starting batch crop with categorization")
    logger.info(f"[CropService] Input: {input_directory}, Output: {output_directory}")
    
    if not os.path.isdir(input_directory):
        return False, "Input directory not found", {}
    
    os.makedirs(output_directory, exist_ok=True)
    
    # 掃描所有圖片文件
    image_files = []
    for root, dirs, files in os.walk(input_directory):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')):
                image_files.append(os.path.join(root, filename))
    
    if not image_files:
        return False, "No image files found", {}
    
    results = {
        "processed_files": 0,
        "successful_crops": 0,
        "failed_crops": 0,
        "crop_categories": {
            "head": [],
            "halfbody": [],
            "person": [],
            "unknown": []
        }
    }
    
    for image_path in image_files:
        try:
            logger.info(f"[CropService] Processing: {os.path.basename(image_path)}")
            
            # 載入圖片
            image_pil = Image.open(image_path)
            
            # 執行裁切
            cropped_images_data, message = crop_image_service(image_pil, None, logger, config)
            
            if cropped_images_data and isinstance(cropped_images_data, list):
                # 保存裁切結果到分類資料夾
                saved_paths = save_crops_to_categories(
                    cropped_images_data, 
                    output_directory, 
                    os.path.basename(image_path), 
                    logger, 
                    config
                )
                
                # 統計結果
                for crop_type, paths in saved_paths.items():
                    if crop_type in results["crop_categories"]:
                        results["crop_categories"][crop_type].extend(paths)
                    else:
                        results["crop_categories"]["unknown"].extend(paths)
                
                results["successful_crops"] += 1
                logger.info(f"[CropService] Successfully cropped {os.path.basename(image_path)}: {len(cropped_images_data)} crops")
            else:
                results["failed_crops"] += 1
                logger.warning(f"[CropService] No crops generated for {os.path.basename(image_path)}")
            
            results["processed_files"] += 1
            image_pil.close()
            
        except Exception as e:
            results["failed_crops"] += 1
            logger.error(f"[CropService] Error processing {image_path}: {e}")
    
    # 生成摘要
    summary = f"Batch crop completed. Processed: {results['processed_files']}, Success: {results['successful_crops']}, Failed: {results['failed_crops']}"
    category_summary = {k: len(v) for k, v in results["crop_categories"].items()}
    logger.info(f"[CropService] {summary}")
    logger.info(f"[CropService] Crop categories: {category_summary}")
    
    return True, summary, results

def crop_image_service_entry(image_pil: Image.Image, logger, config=None):
    """
    Entry point for orchestrator - crops image using waifuc.
    Returns: (pil_image, output_path, message)
    """
    try:
        # Use existing crop logic, but adapt for orchestrator
        result_data, message = crop_image_service(image_pil, None, logger, config)
        
        if result_data is None:
            return image_pil, None, message
        elif isinstance(result_data, list) and len(result_data) > 0:
            # Return the first cropped image (most common use case)
            # Could be enhanced to return multiple crops or best crop
            best_crop = result_data[0]["image"]
            return best_crop, None, f"Cropped image ({result_data[0]['type']}). {message}"
        else:
            return image_pil, None, "No crops generated"
            
    except Exception as e:
        logger.error(f"[CropService] Error in crop_image_service_entry: {e}")
        return image_pil, None, f"Crop error: {str(e)}"
