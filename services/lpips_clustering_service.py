# services/lpips_clustering_service.py
import os
from imgutils.metrics import lpips_clustering
from utils.error_handler import safe_execute

# Expect logger and config to be passed.

def _batch_generator(lst, batch_size):
    """Generator to batch process a list of items."""
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

def _safe_lpips_clustering_internal(file_paths_batch, logger):
    """
    Internal wrapper for lpips_clustering to be used with safe_execute.
    Handles specific exceptions if necessary, or re-raises for safe_execute.
    """
    try:
        # lpips_clustering from imgutils expects a list of file paths
        # and returns a list of cluster IDs (integers) or -1 for noise.
        result = lpips_clustering(file_paths_batch)
        logger.debug(f"_safe_lpips_clustering_internal: Processed batch of {len(file_paths_batch)} files.")
        return result
    except Exception as e:
        # Log the specific error before re-raising, so safe_execute can provide broader context.
        logger.error(f"_safe_lpips_clustering_internal: Error during LPIPS clustering: {e}", exc_info=True)
        # Re-raise the exception to be caught and handled by safe_execute
        raise # This allows safe_execute to return default_return and log the error message prefix

def cluster_images_service(image_paths: list, logger, config=None):
    logger.info(f"[LPIPSClusteringService] Starting LPIPS clustering for {len(image_paths)} images.")

    if not image_paths:
        logger.warning("[LPIPSClusteringService] No image paths provided for clustering.")
        return [], "No images to cluster."

    # Default batch size, can be overridden by config if provided
    batch_size = 100 
    if config and hasattr(config, 'LPIPS_BATCH_SIZE') and isinstance(config.LPIPS_BATCH_SIZE, int):
        batch_size = config.LPIPS_BATCH_SIZE
        logger.info(f"[LPIPSClusteringService] Using batch size from config: {batch_size}")
    
    all_clustering_results = [] # To store (file_path, cluster_id) tuples
    processed_files_count = 0

    for batch_num, image_paths_batch in enumerate(_batch_generator(image_paths, batch_size)):
        logger.info(f"[LPIPSClusteringService] Processing batch {batch_num + 1} with {len(image_paths_batch)} images.")
        
        batch_cluster_ids = safe_execute(
            _safe_lpips_clustering_internal, # The function to execute
            image_paths_batch, # First argument to _safe_lpips_clustering_internal
            logger, # Second argument to _safe_lpips_clustering_internal
            logger=logger, # Logger for safe_execute itself
            default_return=None, # Return None if this batch fails catastrophically
            error_msg_prefix=f"[LPIPSClusteringService] Error processing LPIPS batch {batch_num + 1}"
        )

        if batch_cluster_ids is None:
            logger.error(f"[LPIPSClusteringService] Failed to process batch {batch_num + 1}. Skipping.")
            # We could add placeholders or error markers for these files if needed
            # For now, we just skip adding them to results.
            processed_files_count += len(image_paths_batch) # Still count them as processed (attempted)
            continue

        if len(batch_cluster_ids) != len(image_paths_batch):
            logger.error(
                f"[LPIPSClusteringService] Mismatch in batch size and result size for batch {batch_num + 1}. "
                f"Expected {len(image_paths_batch)} results, got {len(batch_cluster_ids)}. Skipping batch results."
            )
            processed_files_count += len(image_paths_batch)
            continue

        for file_path, cluster_id in zip(image_paths_batch, batch_cluster_ids):
            all_clustering_results.append((file_path, cluster_id))
        
        processed_files_count += len(image_paths_batch)
        logger.info(f"[LPIPSClusteringService] Finished batch {batch_num + 1}. Total processed so far: {processed_files_count}")

    logger.info(f"[LPIPSClusteringService] LPIPS clustering completed for {processed_files_count} images.")
    logger.info(f"[LPIPSClusteringService] Generated {len(all_clustering_results)} cluster assignments.")

    # The service now returns a list of tuples: (file_path, cluster_id)
    # The orchestrator or a subsequent step would be responsible for any file operations (like moving to folders)
    # based on these results.
    return all_clustering_results, f"LPIPS clustering complete. Processed {processed_files_count} images. Found {len(all_clustering_results)} assignments."
