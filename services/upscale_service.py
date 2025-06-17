# services/upscale_service.py
from PIL import Image
from imgutils.upscale import upscale_with_cdc # Main upscaling function
import os

from config import settings as default_settings
from utils.error_handler import safe_execute, ImageProcessingError, ModelError, ConfigError

# Logger will be passed from orchestrator or individual script

def _pil_resize_image(image: Image.Image, target_width: int, target_height: int, preserve_aspect_ratio: bool, logger) -> Image.Image:
    """
    Resizes a PIL image to target dimensions, optionally preserving aspect ratio.
    If preserving aspect ratio, scales the image so that it fits within or covers the target dimensions,
    then crops or pads if necessary (currently, it scales to cover and expects subsequent crop if needed).
    This simplified version scales to ensure both dimensions are >= target if preserve_aspect_ratio is True.
    """
    current_width, current_height = image.size
    logger.debug(f"[UpscaleService] Resizing. Current: {current_width}x{current_height}, Target: {target_width}x{target_height}, Preserve Ratio: {preserve_aspect_ratio}")

    if not target_width and not target_height:
        logger.warning("[UpscaleService] Resize called with no target dimensions. Returning original.")
        return image

    final_width, final_height = target_width, target_height

    if preserve_aspect_ratio:
        if target_width and target_height:
            # Scale to fit/cover: make sure the upscaled image is at least target_width x target_height
            width_scale = target_width / current_width if current_width > 0 else 0
            height_scale = target_height / current_height if current_height > 0 else 0
            scale = max(width_scale, height_scale) # Ensure both dimensions meet target
            if scale <= 0: # Avoid issues with zero or negative scales
                logger.warning(f"[UpscaleService] Invalid scale factor ({scale}) calculated. Using original size for this dimension.")
                final_width = current_width if not target_width else target_width
                final_height = current_height if not target_height else target_height
            else:
                final_width = int(current_width * scale)
                final_height = int(current_height * scale)
        elif target_width: # Only width specified
            if current_width == 0: return image # Avoid division by zero
            scale = target_width / current_width
            final_height = int(current_height * scale)
        elif target_height: # Only height specified
            if current_height == 0: return image # Avoid division by zero
            scale = target_height / current_height
            final_width = int(current_width * scale)
    else:
        # Use target dimensions directly if not preserving aspect ratio
        # Ensure they are positive, otherwise use current dimension
        final_width = target_width if target_width and target_width > 0 else current_width
        final_height = target_height if target_height and target_height > 0 else current_height

    if final_width <= 0 or final_height <= 0:
        logger.error(f"[UpscaleService] Calculated invalid final dimensions: {final_width}x{final_height}. Returning original image.")
        return image

    logger.debug(f"[UpscaleService] Calculated final resize dimensions: {final_width}x{final_height}")

    resample_filter = None
    try:
        if hasattr(Image, 'Resampling'):  # Pillow >= 9.1.0
            if final_width < current_width or final_height < current_height: # Downscaling
                resample_filter = Image.Resampling.LANCZOS
            else: # Upscaling or same size
                resample_filter = Image.Resampling.BICUBIC
            logger.debug(f"[UpscaleService] Using Image.Resampling: {resample_filter}")
        else:  # Pillow < 9.1.0
            if final_width < current_width or final_height < current_height: # Downscaling
                resample_filter = getattr(Image, 'LANCZOS', 1) # Default to 1 if LANCZOS not found
                logger.debug(f"[UpscaleService] Using Image.LANCZOS (fallback: 1): {resample_filter}")
            else: # Upscaling or same size
                resample_filter = getattr(Image, 'BICUBIC', 3) # Default to 3 if BICUBIC not found
                logger.debug(f"[UpscaleService] Using Image.BICUBIC (fallback: 3): {resample_filter}")
    except Exception as e:
        logger.error(f"[UpscaleService] Error selecting resampling filter: {e}. Falling back to integers.", exc_info=True)
        resample_filter = 1 if (final_width < current_width or final_height < current_height) else 3

    if resample_filter is None: # Safeguard
        logger.error("[UpscaleService] Resampling filter is None after selection. Defaulting.")
        resample_filter = 1 if (final_width < current_width or final_height < current_height) else 3

    logger.info(f"[UpscaleService] Final resample filter selected: {resample_filter}")

    try:
        resized_image = image.resize((final_width, final_height), resample=resample_filter)
    except ValueError as e:
        logger.error(f"[UpscaleService] ValueError during resize (filter: {resample_filter}): {e}. Attempting with explicit integer fallback.", exc_info=True)
        fallback_filter = 1 if (final_width < current_width or final_height < current_height) else 3
        logger.info(f"[UpscaleService] Retrying resize with integer filter: {fallback_filter}")
        try:
            resized_image = image.resize((final_width, final_height), resample=fallback_filter)
        except Exception as final_e:
            logger.error(f"[UpscaleService] Resize failed even with integer fallback filter: {final_e}", exc_info=True)
            raise ImageProcessingError(f"Failed to resize image: {final_e}")
    except Exception as e:
        logger.error(f"[UpscaleService] Unexpected error during image resize (filter: {resample_filter}): {e}", exc_info=True)
        raise ImageProcessingError(f"Unexpected error during image resize: {e}")
    
    logger.debug(f"[UpscaleService] Image resized to {resized_image.size}")
    return resized_image

def _center_crop_image(image: Image.Image, target_width: int, target_height: int, logger) -> Image.Image:
    """
    Crops the image to the target dimensions from the center.
    """
    current_width, current_height = image.size
    if target_width <= 0 or target_height <= 0:
        logger.error(f"[UpscaleService] Invalid target dimensions for crop: {target_width}x{target_height}. Returning original.")
        return image

    logger.debug(f"[UpscaleService] Center cropping. Current: {current_width}x{current_height}, Target: {target_width}x{target_height}")

    if current_width < target_width or current_height < target_height:
        logger.warning("[UpscaleService] Image is smaller than target crop dimensions. Padding or different strategy might be needed. Returning original for now.")
        # Or, alternatively, pad the image to target_width/target_height before cropping, 
        # or crop to the largest possible size within image bounds.
        # For simplicity, returning original if it's smaller than crop target.
        return image

    left = (current_width - target_width) // 2
    top = (current_height - target_height) // 2
    right = left + target_width
    bottom = top + target_height
    
    cropped_image = image.crop((left, top, right, bottom))
    logger.info(f"[UpscaleService] Image cropped from {current_width}x{current_height} to {target_width}x{target_height}.")
    return cropped_image

def _upscale_image_core_logic(image_pil: Image.Image, logger, config):
    if not isinstance(image_pil, Image.Image):
        raise ImageProcessingError("Invalid input: image_pil must be a PIL Image object.", "N/A")

    # Load settings from config or use defaults
    model_name = getattr(config, "UPSCALE_MODEL_NAME", default_settings.UPSCALE_MODEL_NAME)
    target_width = getattr(config, "UPSCALE_TARGET_WIDTH", default_settings.UPSCALE_TARGET_WIDTH)
    target_height = getattr(config, "UPSCALE_TARGET_HEIGHT", default_settings.UPSCALE_TARGET_HEIGHT)
    preserve_aspect_ratio = getattr(config, "UPSCALE_PRESERVE_ASPECT_RATIO", default_settings.UPSCALE_PRESERVE_ASPECT_RATIO)
    center_crop_after_upscale = getattr(config, "UPSCALE_CENTER_CROP_AFTER_UPSCALE", default_settings.UPSCALE_CENTER_CROP_AFTER_UPSCALE)
    tile_size = getattr(config, "UPSCALE_TILE_SIZE", default_settings.UPSCALE_TILE_SIZE)
    tile_overlap = getattr(config, "UPSCALE_TILE_OVERLAP", default_settings.UPSCALE_TILE_OVERLAP)
    batch_size = getattr(config, "UPSCALE_BATCH_SIZE", default_settings.UPSCALE_BATCH_SIZE)
    min_size_threshold = getattr(config, "UPSCALE_MIN_SIZE_THRESHOLD", default_settings.UPSCALE_MIN_SIZE_THRESHOLD)

    original_width, original_height = image_pil.size
    logger.info(f"[UpscaleService] Original image size: {original_width}x{original_height}. Model: {model_name}")

    # Check min_size_threshold
    if min_size_threshold is not None:
        if original_width >= min_size_threshold and original_height >= min_size_threshold:
            msg = f"Skipped: Image size ({original_width}x{original_height}) meets/exceeds threshold ({min_size_threshold}px)."
            logger.info(f"[UpscaleService] {msg}")
            return image_pil, msg # Return original image

    # Convert to RGB if not already (some models might require it)
    if image_pil.mode not in ['RGB', 'L']: # L for grayscale, some models might handle it
        logger.debug(f"[UpscaleService] Converting image from {image_pil.mode} to RGB.")
        image_pil = image_pil.convert('RGB')

    # --- Stage 1: AI Upscaling with imgutils.upscale_with_cdc ---
    try:
        logger.info(f"[UpscaleService] Starting AI upscaling with model: {model_name}, tile: {tile_size}, overlap: {tile_overlap}")
        ai_upscaled_image = upscale_with_cdc(
            image_pil,
            model=model_name,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            batch_size=batch_size,
            silent=True # Assuming logger provides enough feedback
        )
        logger.info(f"[UpscaleService] AI upscaling complete. New size: {ai_upscaled_image.size}")
    except Exception as e:
        error_msg = f"AI Upscaling (model: {model_name}) failed: {str(e)}"
        logger.error(f"[UpscaleService] {error_msg}", exc_info=True)
        if "model" in str(e).lower() or "cuda" in str(e).lower() or "memory" in str(e).lower() or "onnx" in str(e).lower():
            raise ModelError(error_msg, model_name) from e
        else:
            raise ImageProcessingError(error_msg, "AI Upscaling Stage") from e

    current_processed_image = ai_upscaled_image

    # --- Stage 2: Resizing to target dimensions (if specified) ---
    # This step is crucial if the AI upscaler doesn't hit the exact target, or if a specific size is needed.
    # The upscale_to_target_size in the original upscale.py combined AI upscale and resize.
    # Here, we separate them for clarity. AI upscale first, then resize if needed.

    # If target dimensions are set, we might need to resize the AI upscaled image.
    # This is especially true if preserve_aspect_ratio is True, as AI upscale might give, e.g., 4x, 
    # and then we need to scale it to fit the target_width/target_height box.
    if target_width or target_height:
        logger.info(f"[UpscaleService] Resizing AI upscaled image to fit target dimensions: W={target_width}, H={target_height}, PreserveRatio={preserve_aspect_ratio}")
        current_processed_image = _pil_resize_image(current_processed_image, target_width, target_height, preserve_aspect_ratio, logger)
    
    # --- Stage 3: Center Cropping (if enabled and target dimensions are set) ---
    if center_crop_after_upscale and target_width and target_height:
        logger.info(f"[UpscaleService] Performing center crop to {target_width}x{target_height}.")
        final_image = _center_crop_image(current_processed_image, target_width, target_height, logger)
    else:
        final_image = current_processed_image

    # Construct a meaningful message
    status_message = f"Image processed. Final size: {final_image.size}."
    ai_upscaled_flag = False # Flag to track if AI upscaling was attempted/done

    # Check if AI upscaling was performed
    try:
        # This check assumes ai_upscaled_image is defined if cdc was called.
        # A more direct check would be if mock_cdc was called in a test, or a flag from cdc call.
        if 'ai_upscaled_image' in locals() and ai_upscaled_image is not None:
            ai_upscaled_flag = True
            if current_processed_image is ai_upscaled_image and final_image is ai_upscaled_image:
                 status_message = f"AI upscaling performed. Final size: {final_image.size}."
            else:
                 status_message = f"AI upscaling and further processing performed. Final size: {final_image.size}."
        # If not ai_upscaled_flag, it means either skipped due to threshold or cdc was not called.
        # The initial status_message is generic enough for this.
    except Exception:
        # This try-except is mainly for the locals() check, though it's broad.
        pass

    logger.info(f"[UpscaleService] Core logic finished. {status_message}")
    return final_image, status_message

def upscale_image_service(image_pil: Image.Image, logger, config=None):
    """
    Service function to upscale an image.
    Accepts a PIL Image object.
    Uses settings from the provided config or defaults.
    """
    logger.info(f"[UpscaleService] Received request to upscale image.")
    
    if config is None:
        config = default_settings # Fallback to default settings
        logger.info("[UpscaleService] No specific config provided, using default settings.")

    result_or_error = safe_execute(
        _upscale_image_core_logic,
        image_pil,
        logger,
        config,
        logger=logger,
        default_return=None,
        error_msg_prefix="[UpscaleService] Error during image upscaling"
    )

    if result_or_error is not None:
        upscaled_image, msg = result_or_error
        return upscaled_image, msg
    else:
        logger.error(f"[UpscaleService] Upscaling failed")
        return None, "Upscaling failed due to error"

def upscale_image_service_entry(image_path, logger, config=None, output_path=None):
    """
    Main entry point for the upscale service.
    Handles loading an image, upscaling it, and saving the result.
    """
    if config is None:
        config_obj = default_settings # Use global settings
    elif isinstance(config, dict):
        # Create a simple namespace object from dict for attribute access
        # This allows using config.SETTING_NAME like with the settings module
        class DictConfig:
            def __init__(self, dictionary):
                for key, value in dictionary.items():
                    setattr(self, key, value)
        config_obj = DictConfig(config)
    else: # Assuming it's already a config object (e.g. module or class instance)
        config_obj = config

    logger.info(f"[UpscaleService] Starting upscale for image: {image_path}")

    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image path does not exist: {image_path}")
        image_pil = Image.open(image_path)
    except FileNotFoundError as e:
        logger.error(f"[UpscaleService] File not found: {image_path}. Error: {e}", exc_info=True)
        raise ImageProcessingError(f"Input file not found: {image_path}", image_path) from e
    except Exception as e:
        logger.error(f"[UpscaleService] Error loading image {image_path}: {e}", exc_info=True)
        raise ImageProcessingError(f"Failed to load image: {image_path}", image_path) from e

    processed_image, message = _upscale_image_core_logic(image_pil, logger, config_obj)
    logger.info(f"[UpscaleService] Core processing message: {message}")

    if output_path:
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                logger.info(f"[UpscaleService] Created output directory: {output_dir}")
            
            # Determine image format from output_path extension or original image
            save_format = None
            output_ext = os.path.splitext(output_path)[1].lower()
            if output_ext in ['.jpg', '.jpeg']:
                save_format = 'JPEG'
            elif output_ext == '.png':
                save_format = 'PNG'
            # Add more formats if needed
            else: # Fallback to original image format or PNG
                save_format = image_pil.format if image_pil.format else 'PNG'
                logger.warning(f"[UpscaleService] Unknown output extension '{output_ext}'. Saving as {save_format}.")

            # Handle RGB conversion for JPEG
            if save_format == 'JPEG' and processed_image.mode == 'RGBA':
                logger.debug("[UpscaleService] Converting RGBA image to RGB for JPEG saving.")
                processed_image = processed_image.convert('RGB')

            processed_image.save(output_path, format=save_format)
            logger.info(f"[UpscaleService] Processed image saved to: {output_path}")
            return processed_image, output_path, message
        except Exception as e:
            logger.error(f"[UpscaleService] Error saving image to {output_path}: {e}", exc_info=True)
            # Decide if this should raise an error or just return the processed image without path
            raise ImageProcessingError(f"Failed to save processed image to {output_path}", output_path) from e
    else:
        logger.info("[UpscaleService] Output path not provided. Returning processed PIL image.")
        return processed_image, None, message # Return PIL image, no path, and message

def upscale_batch_images(input_directory, output_directory, logger, config=None):
    """
    批量放大圖片到指定尺寸
    """
    logger.info(f"[UpscaleService] Starting batch upscale")
    logger.info(f"[UpscaleService] Input: {input_directory}, Output: {output_directory}")
    
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
        "successful_upscales": 0,
        "failed_upscales": 0,
        "skipped_files": 0,
        "total_size_before": 0,
        "total_size_after": 0,
        "upscaled_files": []
    }
    
    for image_path in image_files:
        try:
            logger.info(f"[UpscaleService] Processing: {os.path.basename(image_path)}")
            
            # 記錄原始文件大小
            original_size = os.path.getsize(image_path)
            results["total_size_before"] += original_size
            
            # 生成輸出路徑
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            output_filename = f"{name}_upscaled{ext}"
            output_path = os.path.join(output_directory, output_filename)
            
            # 處理重名文件
            if os.path.exists(output_path):
                counter = 1
                while os.path.exists(output_path):
                    output_filename = f"{name}_upscaled_{counter}{ext}"
                    output_path = os.path.join(output_directory, output_filename)
                    counter += 1
            
            # 執行放大
            result_image, final_output_path, message = upscale_image_service_entry(
                image_path, logger, config, output_path
            )
            
            if result_image and final_output_path and os.path.exists(final_output_path):
                # 記錄處理後文件大小
                upscaled_size = os.path.getsize(final_output_path)
                results["total_size_after"] += upscaled_size
                results["successful_upscales"] += 1
                results["upscaled_files"].append(final_output_path)
                
                logger.info(f"[UpscaleService] Successfully upscaled {filename}: {original_size/1024:.1f}KB -> {upscaled_size/1024:.1f}KB")
            elif "Skipped" in message:
                results["skipped_files"] += 1
                logger.info(f"[UpscaleService] Skipped {filename}: {message}")
            else:
                results["failed_upscales"] += 1
                logger.warning(f"[UpscaleService] Failed to upscale {filename}: {message}")
            
            results["processed_files"] += 1
            
        except Exception as e:
            results["failed_upscales"] += 1
            logger.error(f"[UpscaleService] Error processing {image_path}: {e}")
    
    # 生成摘要
    size_increase = ((results["total_size_after"] / max(results["total_size_before"], 1)) - 1) * 100
    summary = f"Batch upscale completed. Processed: {results['processed_files']}, Success: {results['successful_upscales']}, Failed: {results['failed_upscales']}, Skipped: {results['skipped_files']}, Size increase: {size_increase:.1f}%"
    logger.info(f"[UpscaleService] {summary}")
    
    return True, summary, results

def upscale_to_training_size(image_pil, target_size, logger, config=None):
    """
    將圖片放大到適合訓練的尺寸
    target_size: tuple (width, height) 或 int (正方形)
    """
    try:
        if isinstance(target_size, int):
            target_width = target_height = target_size
        else:
            target_width, target_height = target_size
        
        original_width, original_height = image_pil.size
        logger.info(f"[UpscaleService] Upscaling for training: {original_width}x{original_height} -> {target_width}x{target_height}")
        
        # 創建臨時配置
        class TrainingUpscaleConfig:
            def __init__(self, base_config, target_w, target_h):
                # 複製基礎配置
                if base_config:
                    for attr in dir(base_config):
                        if not attr.startswith('_'):
                            setattr(self, attr, getattr(base_config, attr))
                
                # 覆蓋訓練特定設置
                self.UPSCALE_TARGET_WIDTH = target_w
                self.UPSCALE_TARGET_HEIGHT = target_h
                self.UPSCALE_PRESERVE_ASPECT_RATIO = False  # 精確尺寸
                self.UPSCALE_CENTER_CROP_AFTER_UPSCALE = True  # 裁切到精確尺寸
                self.UPSCALE_MIN_SIZE_THRESHOLD = None  # 總是執行
        
        training_config = TrainingUpscaleConfig(config, target_width, target_height)
        
        # 執行放大
        result_image, message = upscale_image_service(image_pil, logger, training_config)
        
        if result_image:
            final_width, final_height = result_image.size
            logger.info(f"[UpscaleService] Training upscale completed: {final_width}x{final_height}")
            
            # 驗證尺寸是否符合要求
            if final_width == target_width and final_height == target_height:
                return result_image, f"Successfully upscaled to training size: {target_width}x{target_height}"
            else:
                logger.warning(f"[UpscaleService] Size mismatch. Expected: {target_width}x{target_height}, Got: {final_width}x{final_height}")
                return result_image, f"Upscaled but size mismatch. Got: {final_width}x{final_height}"
        else:
            return image_pil, f"Training upscale failed: {message}"
            
    except Exception as e:
        logger.error(f"[UpscaleService] Error in training upscale: {e}")
        return image_pil, f"Training upscale error: {str(e)}"

def upscale_image_service_entry_for_orchestrator(image_pil: Image.Image, logger, config=None):
    """
    Entry point for orchestrator - upscales PIL image.
    Returns: (result_image, output_path, message) - standardized 3-value return
    
    This function bridges the gap between orchestrator's expectation and 
    the existing upscale_image_service function.
    """
    try:
        # Use the existing upscale_image_service which accepts PIL Image
        result_image, message = upscale_image_service(image_pil, logger, config)
        
        # Return in the format orchestrator expects: (image, path, message)
        # output_path is None since we're working with PIL images in memory
        return result_image, None, message
        
    except Exception as e:
        logger.error(f"[UpscaleService] Error in orchestrator entry point: {e}", exc_info=True)
        # Return original image with error message in case of failure
        return image_pil, None, f"Upscale failed: {str(e)}"

# Example usage (for testing this module directly)
if __name__ == '__main__':
    from utils.logger_config import setup_logging
    
    test_logger = setup_logging("test_upscale_service", default_settings.LOG_DIR, default_settings.LOG_LEVEL)
    
    try:
        # Create a dummy PIL image (e.g., a 100x100 red image)
        dummy_image_orig = Image.new('RGB', (100, 150), color = 'red') # Non-square
        test_logger.info(f"Created a dummy PIL image ({dummy_image_orig.size}) for testing.")

        # Test 1: Default config (should upscale and crop if defaults are set to do so)
        test_logger.info("\\n--- Test 1: Default Config ---")
        upscaled_img_default, msg_default = upscale_image_service(dummy_image_orig.copy(), test_logger)
        if upscaled_img_default:
            test_logger.info(f"Default Upscale Message: {msg_default}")
            test_logger.info(f"Default Upscaled Image size: {upscaled_img_default.size}")
            # upscaled_img_default.save(os.path.join(default_settings.OUTPUT_DIR, "test_upscaled_default.png"))
        else:
            test_logger.error(f"Default Upscale Failed: {msg_default}")

        # Test 2: Custom config - upscale only, no crop, different target
        class MockUpscaleConfig1:
            UPSCALE_MODEL_NAME = default_settings.UPSCALE_MODEL_NAME # Use a fast default or specify one known to be available
            UPSCALE_TARGET_WIDTH = 600
            UPSCALE_TARGET_HEIGHT = 0 # Auto height based on aspect ratio
            UPSCALE_PRESERVE_ASPECT_RATIO = True
            UPSCALE_CENTER_CROP_AFTER_UPSCALE = False # Important: no crop
            UPSCALE_MIN_SIZE_THRESHOLD = 50 # Ensure it runs
            # Tile, overlap, batch can be default
            UPSCALE_TILE_SIZE = default_settings.UPSCALE_TILE_SIZE
            UPSCALE_TILE_OVERLAP = default_settings.UPSCALE_TILE_OVERLAP
            UPSCALE_BATCH_SIZE = default_settings.UPSCALE_BATCH_SIZE

        test_logger.info("\\n--- Test 2: Custom Config - Upscale, Preserve Ratio, No Crop ---")
        custom_config1 = MockUpscaleConfig1()
        upscaled_img_custom1, msg_custom1 = upscale_image_service(dummy_image_orig.copy(), test_logger, config=custom_config1)
        if upscaled_img_custom1:
            test_logger.info(f"Custom1 Upscale Message: {msg_custom1}")
            test_logger.info(f"Custom1 Upscaled Image size: {upscaled_img_custom1.size}")
            # upscaled_img_custom1.save(os.path.join(default_settings.OUTPUT_DIR, "test_upscaled_custom1_no_crop.png"))
        else:
            test_logger.error(f"Custom1 Upscale Failed: {msg_custom1}")

        # Test 3: Custom config - upscale AND crop to a fixed size, not preserving intermediate ratio
        class MockUpscaleConfig2:
            UPSCALE_MODEL_NAME = default_settings.UPSCALE_MODEL_NAME
            UPSCALE_TARGET_WIDTH = 500 
            UPSCALE_TARGET_HEIGHT = 500
            UPSCALE_PRESERVE_ASPECT_RATIO = False # Resize to cover 500x500 then crop
            UPSCALE_CENTER_CROP_AFTER_UPSCALE = True # Crop to 500x500
            UPSCALE_MIN_SIZE_THRESHOLD = None # Ensure it runs
            UPSCALE_TILE_SIZE = default_settings.UPSCALE_TILE_SIZE
            UPSCALE_TILE_OVERLAP = default_settings.UPSCALE_TILE_OVERLAP
            UPSCALE_BATCH_SIZE = default_settings.UPSCALE_BATCH_SIZE

        test_logger.info("\\n--- Test 3: Custom Config - Upscale to cover 500x500, then Center Crop to 500x500 ---")
        custom_config2 = MockUpscaleConfig2()
        upscaled_img_custom2, msg_custom2 = upscale_image_service(dummy_image_orig.copy(), test_logger, config=custom_config2)
        if upscaled_img_custom2:
            test_logger.info(f"Custom2 Upscale Message: {msg_custom2}")
            test_logger.info(f"Custom2 Upscaled Image size: {upscaled_img_custom2.size}")
            # upscaled_img_custom2.save(os.path.join(default_settings.OUTPUT_DIR, "test_upscaled_custom2_cropped.png"))
            assert upscaled_img_custom2.size == (500, 500), "Test 3 failed: Cropped size mismatch"
        else:
            test_logger.error(f"Custom2 Upscale Failed: {msg_custom2}")

        # Test 4: Min size threshold skip
        class MockUpscaleConfig3:
            UPSCALE_MODEL_NAME = default_settings.UPSCALE_MODEL_NAME
            UPSCALE_TARGET_WIDTH = 500 
            UPSCALE_TARGET_HEIGHT = 500
            UPSCALE_PRESERVE_ASPECT_RATIO = True
            UPSCALE_CENTER_CROP_AFTER_UPSCALE = True
            UPSCALE_MIN_SIZE_THRESHOLD = 80 # Original is 100x150, so it should skip
            UPSCALE_TILE_SIZE = default_settings.UPSCALE_TILE_SIZE
            UPSCALE_TILE_OVERLAP = default_settings.UPSCALE_TILE_OVERLAP
            UPSCALE_BATCH_SIZE = default_settings.UPSCALE_BATCH_SIZE

        test_logger.info("\\n--- Test 4: Min Size Threshold Skip ---")
        custom_config3 = MockUpscaleConfig3()
        upscaled_img_custom3, msg_custom3 = upscale_image_service(dummy_image_orig.copy(), test_logger, config=custom_config3)
        if upscaled_img_custom3:
            test_logger.info(f"Custom3 Upscale Message: {msg_custom3}")
            test_logger.info(f"Custom3 Upscaled Image size: {upscaled_img_custom3.size}")
            assert upscaled_img_custom3.size == dummy_image_orig.size, "Test 4 failed: Image should not have been upscaled"
            assert "Skipped" in msg_custom3, "Test 4 failed: Message should indicate skip"
        else:
            test_logger.error(f"Custom3 Upscale Failed: {msg_custom3}")

    except ImportError as ie:
        test_logger.error(f"Import error, ensure necessary libraries (PIL, imgutils, onnxruntime) are installed: {ie}")
    except ModelError as me:
         test_logger.error(f"Model error during testing (this is expected if models are not downloaded/configured): {me}")
    except Exception as e:
        test_logger.error(f"An unexpected error occurred during testing: {e}", exc_info=True)

    test_logger.info("Upscale service test finished.")
