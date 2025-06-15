# services/tag_service.py
import os
import tempfile
from PIL import Image
from imgutils.tagging import get_wd14_tags, tags_to_text

from config import settings as default_settings # Import default settings
from utils.error_handler import safe_execute, ImageProcessingError, ModelError

# Logger will be passed from orchestrator or individual script

def _process_tags_with_config(rating, features, chars, config, logger):
    """
    Helper function to process raw tags based on configuration.
    """
    custom_character_tag = getattr(config, "TAG_CUSTOM_CHARACTER_TAG", default_settings.TAG_CUSTOM_CHARACTER_TAG)
    custom_artist_name = getattr(config, "TAG_CUSTOM_ARTIST_NAME", default_settings.TAG_CUSTOM_ARTIST_NAME)
    enable_wildcard = getattr(config, "TAG_ENABLE_WILDCARD", default_settings.TAG_ENABLE_WILDCARD)
    wildcard_template = getattr(config, "TAG_WILDCARD_TEMPLATE", default_settings.TAG_WILDCARD_TEMPLATE)
    general_threshold = getattr(config, "TAG_GENERAL_THRESHOLD", default_settings.TAG_GENERAL_THRESHOLD)
    # character_threshold = getattr(config, "TAG_CHARACTER_THRESHOLD", default_settings.TAG_CHARACTER_THRESHOLD) # imgutils handles this internally
    excluded_tags_list = getattr(config, "TAG_EXCLUDED_TAGS", default_settings.TAG_EXCLUDED_TAGS)
    prepend_tags_str = getattr(config, "TAG_PREPEND_TAGS", default_settings.TAG_PREPEND_TAGS)
    append_tags_str = getattr(config, "TAG_APPEND_TAGS", default_settings.TAG_APPEND_TAGS)

    parts = []

    # 1. Prepend tags
    if prepend_tags_str:
        parts.append(prepend_tags_str)

    # 2. Character tags
    processed_chars = []
    if chars: # chars is a dict {'character_name': probability}
        for char, prob in chars.items():
            # Assuming character_threshold is applied by get_wd14_tags or we use general_threshold as a fallback if not
            # For now, let's assume get_wd14_tags already filtered by its internal character threshold.
            # If not, we might need to filter here, e.g., if prob >= character_threshold:
            processed_chars.append(char.replace('_', ' ')) # Replace underscores for readability
    
    if processed_chars:
        parts.append(", ".join(processed_chars))
    elif custom_character_tag:
        parts.append(custom_character_tag)

    # 3. Artist name
    if custom_artist_name:
        parts.append(f"by {custom_artist_name}") # Common practice to prefix with "by"

    # 4. Feature tags (general tags)
    # Filter features by general_threshold and exclude specified tags
    filtered_features = {}
    if features: # features is a dict {'tag_name': probability}
        for tag, prob in features.items():
            tag_lower = tag.lower()
            if prob >= general_threshold and tag_lower not in excluded_tags_list and tag_lower not in [pt.lower() for pt in processed_chars]:
                filtered_features[tag.replace('_', ' ')] = prob
    
    if filtered_features:
        # Sort tags by probability (optional, but can be useful)
        # sorted_tags = sorted(filtered_features.keys(), key=lambda t: filtered_features[t], reverse=True)
        # parts.append(", ".join(sorted_tags))
        parts.append(tags_to_text(filtered_features)) # tags_to_text handles formatting

    # 5. Append tags
    if append_tags_str:
        parts.append(append_tags_str)
        
    # Combine all parts, removing empty strings
    final_tags_list = [part for part in parts if part]
    text_output = ", ".join(final_tags_list).strip()
    
    # Ensure "1girl" or "1boy" (if present) is at the beginning if they exist in character tags
    # This is a common convention for some systems.
    # More robustly, this could be a configurable "priority_tags" list.
    priority_keywords = ["1girl", "1boy", "2girls", "multiple girls", "multiple boys", "solo"]
    found_priority_tags = []

    # Create a temporary list of tags for manipulation
    current_tags_set = [t.strip() for t in text_output.split(',') if t.strip()]
    
    for keyword in priority_keywords:
        if keyword in current_tags_set:
            found_priority_tags.append(keyword)
            current_tags_set.remove(keyword) # Remove to avoid duplication

    # Prepend found priority tags (if any) to the beginning
    if found_priority_tags:
        text_output = ", ".join(found_priority_tags + current_tags_set)
    else:
        text_output = ", ".join(current_tags_set)


    # Wildcard line (if enabled and artist name is present)
    wildcard_output = ""
    if enable_wildcard and custom_artist_name:
        wildcard_output = wildcard_template.format(artist=custom_artist_name)
        # text_output += f"\\n{wildcard_output}" # Appending to main tags or returning separately?
                                                # For now, let's return it separately.

    logger.debug(f"[TagService] Rating: {rating}")
    logger.debug(f"[TagService] Raw Chars: {chars}")
    logger.debug(f"[TagService] Raw Features: {features}")
    logger.debug(f"[TagService] Processed Text Output: {text_output}")
    if wildcard_output:
        logger.debug(f"[TagService] Wildcard Line: {wildcard_output}")
        
    return text_output, wildcard_output


def _tag_image_core_logic(image_pil, logger, config):
    if not isinstance(image_pil, Image.Image):
        raise ImageProcessingError("Invalid input: image_pil must be a PIL Image object.", "N/A")

    model_name = getattr(config, "TAG_MODEL_NAME", default_settings.TAG_MODEL_NAME)
    general_threshold = getattr(config, "TAG_GENERAL_THRESHOLD", default_settings.TAG_GENERAL_THRESHOLD)
    # character_threshold = getattr(config, "TAG_CHARACTER_THRESHOLD", default_settings.TAG_CHARACTER_THRESHOLD) # For get_wd14_tags

    temp_file_path = None
    try:
        # get_wd14_tags expects a file path. Save PIL Image to a temporary file.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            image_pil.save(tmpfile, format="PNG")
            temp_file_path = tmpfile.name
        
        logger.info(f"[TagService] Image saved to temporary file: {temp_file_path} for tagging.")

        # Call the tagging function
        # Note: get_wd14_tags has its own thresholds (general_threshold, character_threshold)
        # We pass general_threshold here. It also has character_threshold with a default.
        # The model_name is crucial.
        try:
            # Ensure ONNXRuntime sessions are managed correctly if this is called frequently.
            # For now, assume imgutils handles session creation/destruction or uses a global session.
            rating, features, chars = get_wd14_tags(
                temp_file_path,
                model_name=model_name,
                general_threshold=general_threshold, # This is the general threshold for feature tags
                # character_threshold=character_threshold, # This is for character tags, imgutils has a default
                # Other parameters like `onnx_models`, `ignore_exif_errors` can be exposed if needed.
            )
        except Exception as e:
            # Catch specific ONNX/model loading errors if possible
            error_msg = f"WD14 Tagger model ({model_name}) processing failed: {str(e)}"
            logger.error(f"[TagService] {error_msg}", exc_info=True)
            if ("model" in str(e).lower() or "download" in str(e).lower() or 
                "onnx" in str(e).lower() or "cuda" in str(e).lower() or "not found" in str(e).lower()): # Added "not found"
                raise ModelError(error_msg, model_name) from e
            else:
                raise ImageProcessingError(error_msg, temp_file_path) from e

        logger.info(f"[TagService] Raw tags obtained. Rating: {rating}, Features: {len(features)}, Chars: {len(chars)}")

        # Process tags with configuration (custom tags, exclusions, etc.)
        processed_tags, wildcard_line = _process_tags_with_config(rating, features, chars, config, logger)
        
        final_output_tags = processed_tags
        if wildcard_line: # If UI/orchestrator wants to handle this separately
            # For now, append to the main tags for simplicity in service return
            final_output_tags += f" | Wildcard: {wildcard_line}"


        num_tags = len(processed_tags.split(',')) if processed_tags else 0
        msg = f"Image tagged successfully with {num_tags} tags using model {model_name}."
        logger.info(f"[TagService] {msg}")
        return final_output_tags, msg

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.debug(f"[TagService] Temporary file {temp_file_path} deleted.")
            except Exception as e:
                logger.warning(f"[TagService] Failed to delete temporary file {temp_file_path}: {e}", exc_info=True)


def tag_image_service(image_pil: Image.Image, logger, config=None):
    """
    Service function to tag an image using WD14 tagger.
    Accepts a PIL Image object.
    Saves the PIL image to a temporary file to be processed by get_wd14_tags.
    """
    logger.info(f"[TagService] Received request to tag image.")
    
    if config is None:
        config = default_settings # Fallback to default settings if no specific config is passed
        logger.info("[TagService] No specific config provided, using default settings.")

    # Use safe_execute to wrap the core logic
    success, result_or_error = safe_execute(
        _tag_image_core_logic,
        image_pil,
        logger=logger,
        config=config, # Pass config to the core logic
        error_msg_prefix="[TagService] Error during image tagging"
    )

    if success:
        tags, msg = result_or_error
        return tags, msg
    else:
        # error_msg_prefix is already included by safe_execute in result_or_error
        logger.error(f"[TagService] Tagging failed: {result_or_error}")
        return "", f"Tagging failed: {result_or_error}"

# Example usage (for testing this module directly)
if __name__ == '__main__':
    from utils.logger_config import setup_logging
    
    # Setup a basic logger for testing
    test_logger = setup_logging("test_tag_service", default_settings.LOG_DIR, default_settings.LOG_LEVEL)
    
    # Create a dummy PIL image (e.g., a 100x100 black image)
    try:
        dummy_image = Image.new('RGB', (100, 100), color = 'red')
        test_logger.info("Created a dummy PIL image for testing.")

        # Test with default config
        test_logger.info("--- Testing with default config ---")
        tags, message = tag_image_service(dummy_image, test_logger)
        test_logger.info(f"Message: {message}")
        test_logger.info(f"Tags: {tags}")

        # Test with custom config (simulating app_settings)
        class MockConfig:
            TAG_MODEL_NAME = "wd-v1-4-convnext-tagger-v2" # A different model
            TAG_GENERAL_THRESHOLD = 0.5
            TAG_CUSTOM_CHARACTER_TAG = "test_char"
            TAG_CUSTOM_ARTIST_NAME = "test_artist"
            TAG_ENABLE_WILDCARD = True
            TAG_WILDCARD_TEMPLATE = "artwork by {artist}"
            TAG_EXCLUDED_TAGS = ["red background", "simple background"] # Example
            TAG_PREPEND_TAGS = "high quality"
            TAG_APPEND_TAGS = "illustration"
            # LOG_DIR = default_settings.LOG_DIR # For logger setup if service managed its own
            # LOG_LEVEL = "DEBUG"

        test_logger.info("\\n--- Testing with custom mock config ---")
        custom_config = MockConfig()
        tags_custom, message_custom = tag_image_service(dummy_image, test_logger, config=custom_config)
        test_logger.info(f"Custom Message: {message_custom}")
        test_logger.info(f"Custom Tags: {tags_custom}")

    except ImportError as ie:
        test_logger.error(f"Import error, ensure necessary libraries (PIL, imgutils, onnxruntime) are installed: {ie}")
    except ModelError as me:
         test_logger.error(f"Model error during testing (this is expected if models are not downloaded): {me}")
    except Exception as e:
        test_logger.error(f"An unexpected error occurred during testing: {e}", exc_info=True)

    test_logger.info("Tag service test finished.")
