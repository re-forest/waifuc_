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
    # safe_execute will return this tuple, or (None, error_message_prefix) if an exception occurs within _safe_waifuc_process_internal
    
    result_tuple = safe_execute(
        _safe_waifuc_process_internal,
        image_pil,
        logger=logger, # logger for _safe_waifuc_process_internal
        config=config, # config for _safe_waifuc_process_internal
        default_return=(None, "Cropping failed due to an internal error."), # Default if _safe_waifuc_process_internal raises unhandled
        error_msg_prefix="[CropService] Error during waifuc image processing",
        logger_for_safe_execute=logger # logger for safe_execute itself
    )
    
    # Unpack the tuple from safe_execute or _safe_waifuc_process_internal
    # If _safe_waifuc_process_internal failed and raised, safe_execute returns (default_return_value_of_safe_execute, error_msg_prefix)
    # If _safe_waifuc_process_internal succeeded, it returns (list_of_images, message_from_internal)
    # If _safe_waifuc_process_internal failed and default_return was used by safe_execute, it returns that default.

    cropped_images_data, message = result_tuple

    if cropped_images_data is None:
        # This means safe_execute caught an exception from _safe_waifuc_process_internal
        # and returned its default_return. The error was already logged by safe_execute.
        logger.error(f"[CropService] Cropping returned no image data. Message: {message}")
        return None, message # Message here is from safe_execute's default_return or error_msg_prefix

    if not cropped_images_data: # Empty list, but not None
        logger.info(f"[CropService] Cropping completed, but no crops were generated (or loaded). Message: {message}")
        # This could be a valid scenario if the image had no detectable persons/parts by ThreeStageSplitAction.
        # Message is from _safe_waifuc_process_internal
        return [], message 

    logger.info(f"[CropService] Cropping successful. Produced {len(cropped_images_data)} image(s). Message: {message}")
    # The main return for success is a list of dicts: {"image": PIL.Image, "type": str, "original_filename": str}
    return cropped_images_data, message
