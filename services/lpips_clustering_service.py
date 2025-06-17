# services/lpips_clustering_service.py
import os
from imgutils.metrics import lpips_clustering
from utils.error_handler import safe_execute

# Expect logger and config to be passed.

def _batch_generator(lst, batch_size):
    """Generator to batch process a list of items."""
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

def _safe_lpips_clustering_internal(file_paths_batch, logger, config=None):
    """
    Internal wrapper for lpips_clustering to be used with safe_execute.
    Handles specific exceptions if necessary, or re-raises for safe_execute.
    """
    try:
        # lpips_clustering from imgutils expects a list of file paths
        # and returns a list of cluster IDs (integers) or -1 for noise.
        # Try with adjusted threshold for better clustering
        threshold = 0.5  # Lower threshold for more aggressive clustering
        
        if config:
            threshold = getattr(config, 'LPIPS_CLUSTERING_THRESHOLD', 0.5)
        
        logger.debug(f"Using LPIPS clustering threshold: {threshold}")
        
        # Try different threshold values if default doesn't work well
        result = lpips_clustering(file_paths_batch, threshold=threshold)
        logger.debug(f"_safe_lpips_clustering_internal: Processed batch of {len(file_paths_batch)} files with threshold {threshold}.")
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
            config, # Third argument to _safe_lpips_clustering_internal
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

    # Analyze clustering results
    cluster_counts = {}
    for file_path, cluster_id in all_clustering_results:
        cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
    
    if cluster_counts:
        logger.info(f"[LPIPSClusteringService] Cluster distribution: {cluster_counts}")
        
        # Count noise points (cluster_id = -1)
        noise_count = cluster_counts.get(-1, 0)
        actual_clusters = len([cid for cid in cluster_counts.keys() if cid != -1])
        
        if noise_count == len(all_clustering_results):
            logger.warning("[LPIPSClusteringService] All images classified as noise. Consider lowering threshold.")
        elif actual_clusters == 0:
            logger.warning("[LPIPSClusteringService] No clusters formed. Images too dissimilar or threshold too low.")
        else:
            logger.info(f"[LPIPSClusteringService] Formed {actual_clusters} clusters with {noise_count} noise points.")

    # The service now returns a list of tuples: (file_path, cluster_id)
    # The orchestrator or a subsequent step would be responsible for any file operations (like moving to folders)
    # based on these results.
    return all_clustering_results, f"LPIPS clustering complete. Processed {processed_files_count} images. Found {len(all_clustering_results)} assignments."

def eliminate_duplicates_from_clusters(clustering_results, logger, config=None):
    """
    從聚類結果中淘汰重複圖片，保留每個聚類中品質最好的圖片
    """
    logger.info(f"[LPIPSClusteringService] Starting duplicate elimination from {len(clustering_results)} clustering results")
    
    # 按聚類ID分組
    clusters = {}
    noise_images = []
    
    for file_path, cluster_id in clustering_results:
        if cluster_id == -1:  # 噪音點
            noise_images.append(file_path)
        else:
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(file_path)
    
    eliminated_images = []
    kept_images = []
    
    # 處理每個聚類
    for cluster_id, image_paths in clusters.items():
        if len(image_paths) <= 1:
            # 單張圖片的聚類，直接保留
            kept_images.extend(image_paths)
            continue
        
        logger.info(f"[LPIPSClusteringService] Processing cluster {cluster_id} with {len(image_paths)} images")
        
        # 評估圖片品質並選擇最佳圖片
        best_image = select_best_image_from_cluster(image_paths, logger, config)
        
        if best_image:
            kept_images.append(best_image)
            for img_path in image_paths:
                if img_path != best_image:
                    eliminated_images.append(img_path)
        else:
            # 如果無法確定最佳圖片，保留第一張
            kept_images.append(image_paths[0])
            eliminated_images.extend(image_paths[1:])
    
    # 噪音圖片（獨特圖片）全部保留
    kept_images.extend(noise_images)
    
    # 實際移動被淘汰的圖片
    if getattr(config, 'LPIPS_AUTO_ELIMINATE_DUPLICATES', True):
        move_eliminated_images(eliminated_images, logger, config)
    
    logger.info(f"[LPIPSClusteringService] Duplicate elimination complete. Kept: {len(kept_images)}, Eliminated: {len(eliminated_images)}")
    
    return {
        'kept_images': kept_images,
        'eliminated_images': eliminated_images,
        'noise_images': noise_images,
        'clusters_processed': len(clusters)
    }

def select_best_image_from_cluster(image_paths, logger, config=None):
    """
    從聚類中選擇品質最好的圖片
    評估標準：文件大小、解析度、圖片清晰度等
    """
    try:
        best_image = None
        best_score = -1
        
        for image_path in image_paths:
            try:
                # 計算圖片品質分數
                score = calculate_image_quality_score(image_path, logger)
                
                if score > best_score:
                    best_score = score
                    best_image = image_path
                    
            except Exception as e:
                logger.warning(f"[LPIPSClusteringService] Failed to evaluate {image_path}: {e}")
                continue
        
        if best_image:
            logger.debug(f"[LPIPSClusteringService] Selected best image: {os.path.basename(best_image)} (score: {best_score:.2f})")
        
        return best_image
        
    except Exception as e:
        logger.error(f"[LPIPSClusteringService] Error selecting best image: {e}")
        return image_paths[0] if image_paths else None

def calculate_image_quality_score(image_path, logger):
    """
    計算圖片品質分數
    """
    try:
        from PIL import Image
        import os
        
        # 文件大小分數 (30%)
        file_size = os.path.getsize(image_path)
        size_score = min(file_size / (1024 * 1024), 10) / 10  # 最多10MB給滿分
        
        # 解析度分數 (50%)
        with Image.open(image_path) as img:
            width, height = img.size
            resolution = width * height
            resolution_score = min(resolution / (2048 * 2048), 1)  # 4MP給滿分
        
        # 文件格式分數 (20%)
        ext = os.path.splitext(image_path)[1].lower()
        format_scores = {
            '.png': 1.0,
            '.jpg': 0.8,
            '.jpeg': 0.8,
            '.webp': 0.9,
            '.bmp': 0.6,
            '.gif': 0.3
        }
        format_score = format_scores.get(ext, 0.5)
        
        # 總分計算
        total_score = (size_score * 0.3 + resolution_score * 0.5 + format_score * 0.2) * 100
        
        logger.debug(f"[LPIPSClusteringService] Quality score for {os.path.basename(image_path)}: {total_score:.2f}")
        return total_score
        
    except Exception as e:
        logger.warning(f"[LPIPSClusteringService] Failed to calculate quality score for {image_path}: {e}")
        return 0

def move_eliminated_images(eliminated_images, logger, config=None):
    """
    移動被淘汰的圖片到指定資料夾
    """
    if not eliminated_images:
        return
    
    try:
        # 確定淘汰圖片的存放位置
        if config and hasattr(config, 'LPIPS_ELIMINATED_DIR'):
            eliminated_dir = config.LPIPS_ELIMINATED_DIR
        else:
            # 在第一張圖片的目錄中創建eliminated_duplicates資料夾
            first_image_dir = os.path.dirname(eliminated_images[0])
            eliminated_dir = os.path.join(first_image_dir, 'eliminated_duplicates')
        
        os.makedirs(eliminated_dir, exist_ok=True)
        
        moved_count = 0
        for image_path in eliminated_images:
            try:
                filename = os.path.basename(image_path)
                target_path = os.path.join(eliminated_dir, filename)
                
                # 處理重名文件
                if os.path.exists(target_path):
                    name, ext = os.path.splitext(filename)
                    counter = 1
                    while os.path.exists(target_path):
                        target_path = os.path.join(eliminated_dir, f"{name}_{counter}{ext}")
                        counter += 1
                
                import shutil
                shutil.move(image_path, target_path)
                moved_count += 1
                
            except Exception as e:
                logger.error(f"[LPIPSClusteringService] Failed to move {image_path}: {e}")
        
        logger.info(f"[LPIPSClusteringService] Moved {moved_count}/{len(eliminated_images)} eliminated images to {eliminated_dir}")
        
    except Exception as e:
        logger.error(f"[LPIPSClusteringService] Error moving eliminated images: {e}")

def cluster_images_service_entry(image_paths: list, logger, config=None):
    """
    Entry point for orchestrator - clusters images using LPIPS.
    Returns: clustering results and message
    Note: This service operates on paths, not PIL images
    """
    try:
        results, message = cluster_images_service(image_paths, logger, config)
        
        # 如果啟用了自動淘汰重複圖片
        if getattr(config, 'LPIPS_AUTO_ELIMINATE_DUPLICATES', True) and results:
            logger.info("[LPIPSClusteringService] Auto-elimination enabled, processing duplicates...")
            elimination_results = eliminate_duplicates_from_clusters(results, logger, config)
            
            # 更新返回消息
            elimination_summary = f"Kept {len(elimination_results['kept_images'])} images, eliminated {len(elimination_results['eliminated_images'])} duplicates"
            message = f"{message}. {elimination_summary}"
            
            return elimination_results, message
        
        return results, message
    except Exception as e:
        logger.error(f"[LPIPSClusteringService] Error in cluster_images_service_entry: {e}")
        return [], f"Clustering error: {str(e)}"
