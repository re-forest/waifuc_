# services/face_detection_service.py
from PIL import Image
from imgutils.detect import detect_faces # Using the core detection function
from utils.error_handler import safe_execute # For safely executing the detection

# No direct logger setup here, expect it to be passed.

def detect_faces_service(image_pil: Image.Image, logger, config=None):
    logger.info(f"[FaceDetectionService] Detecting faces in provided PIL image.")

    if not isinstance(image_pil, Image.Image):
        logger.error("[FaceDetectionService] Input is not a PIL Image object.")
        return None, "Error: Input is not a valid PIL Image.", None

    # The detect_faces function from imgutils can take a PIL image directly.
    # We need a way to handle potential errors from detect_faces itself.
    # The original `detect_faces_in_single_image` had specific error raising.
    # We'll use safe_execute to wrap the call to `detect_faces`.

    def _detect_faces_internal(pil_image):
        # This internal function matches the signature for safe_execute
        # and allows us to use the existing detect_faces directly.
        # It should return something that can be checked, e.g., the result list or raise an exception.
        try:
            # detect_faces returns a list of dicts, each with 'bbox' and 'score'
            detection_result = detect_faces(pil_image)
            return detection_result 
        except Exception as e:
            # Re-raise to be caught by safe_execute, which will log it.
            # Or, handle specific exceptions from imgutils if known.
            logger.debug(f"_detect_faces_internal failed: {e}") # Log the specific error before re-raising
            raise # Re-raise for safe_execute to handle and log more broadly

    detection_results = safe_execute(
        _detect_faces_internal,
        image_pil, # Pass the PIL image
        logger=logger,
        default_return=None, # Return None if detection fails catastrophically
        error_msg_prefix="[FaceDetectionService] Error during face detection model execution"
    )

    if detection_results is None:
        # safe_execute already logged the error
        return image_pil, "Face detection failed or an error occurred.", None # Return original image, error message, no data
    
    face_count = len(detection_results)
    logger.info(f"[FaceDetectionService] Detected {face_count} faces.")

    # For now, the service will return the raw detection_results (list of dicts)
    # and a success message. The orchestrator or a subsequent step (e.g., drawing boxes)
    # would use this data. If the service were to draw boxes, it would return a modified PIL image.
    
    # Example of what could be done if drawing on the image:
    # if face_count > 0 and config and config.get("DRAW_FACE_BOXES", False):
    #     from PIL import ImageDraw
    #     drawable_image = image_pil.copy()
    #     draw = ImageDraw.Draw(drawable_image)
    #     for face in detection_results:
    #         bbox = face['bbox'] # [x0, y0, x1, y1]
    #         draw.rectangle(bbox, outline="red", width=2)
    #     logger.info("[FaceDetectionService] Drew bounding boxes on image.")
    #     return drawable_image, f"Detected {face_count} faces and drew boxes.", detection_results # Return modified image

    return image_pil, f"Detected {face_count} faces.", detection_results # Return original image, success message, detection data
