# services/face_detection_service.py
from PIL import Image
from imgutils.detect import detect_faces # Using the core detection function
from utils.error_handler import safe_execute # For safely executing the detection

# No direct logger setup here, expect it to be passed.

def detect_faces_service(image_pil: Image.Image, logger, config=None):
    """
    人臉偵測服務 - 正確使用 detect_faces 並返回詳細的人臉資訊
    """
    logger.info(f"[FaceDetectionService] 開始偵測人臉...")

    if not isinstance(image_pil, Image.Image):
        logger.error("[FaceDetectionService] 輸入不是 PIL Image 對象")
        return None, "錯誤: 輸入不是有效的 PIL Image", None

    def _detect_faces_internal(pil_image):
        """
        內部人臉偵測函數 - 正確使用 imgutils.detect.detect_faces
        """
        try:
            # 獲取配置參數
            conf_threshold = getattr(config, 'FACE_DETECTION_CONFIDENCE_THRESHOLD', 0.3)
            
            # 使用 detect_faces 進行偵測
            # detect_faces 返回 list of dict，每個 dict 包含:
            # - 'bbox': [x1, y1, x2, y2] 邊界框座標
            # - 'score': float 置信度分數
            detection_result = detect_faces(pil_image, conf_threshold=conf_threshold)
            
            logger.debug(f"[FaceDetectionService] detect_faces 返回 {len(detection_result)} 個結果")
            
            # 處理偵測結果，建構詳細資訊
            processed_faces = []
            for i, face_data in enumerate(detection_result):
                face_info = {
                    'id': i,
                    'bbox': face_data.get('bbox', [0, 0, 0, 0]),  # 邊界框 [x1, y1, x2, y2]
                    'confidence': face_data.get('score', 0.0),    # 置信度
                    'area': 0,                                     # 人臉區域面積
                    'center': [0, 0]                              # 人臉中心點
                }
                
                # 計算人臉區域面積和中心點
                if 'bbox' in face_data and len(face_data['bbox']) >= 4:
                    x1, y1, x2, y2 = face_data['bbox'][:4]
                    face_info['area'] = abs((x2 - x1) * (y2 - y1))
                    face_info['center'] = [(x1 + x2) / 2, (y1 + y2) / 2]
                
                processed_faces.append(face_info)
            
            return processed_faces
            
        except Exception as e:
            logger.error(f"[FaceDetectionService] detect_faces 調用失敗: {e}")
            raise

    # 安全執行人臉偵測
    detection_results = safe_execute(
        _detect_faces_internal,
        image_pil,
        logger=logger,
        default_return=[],  # 預設返回空列表而非 None
        error_msg_prefix="[FaceDetectionService] 人臉偵測執行錯誤"
    )

    if detection_results is None:
        detection_results = []
    
    face_count = len(detection_results)
    
    # 生成詳細的偵測報告
    if face_count == 0:
        message = "未偵測到人臉"
        logger.info(f"[FaceDetectionService] {message}")
    else:
        # 計算平均置信度
        avg_confidence = sum(face['confidence'] for face in detection_results) / face_count
        total_area = sum(face['area'] for face in detection_results)
        
        message = f"偵測到 {face_count} 個人臉 (平均置信度: {avg_confidence:.2f}, 總面積: {total_area:.0f})"
        logger.info(f"[FaceDetectionService] {message}")
        
        # 記錄每個人臉的詳細資訊
        for face in detection_results:
            logger.debug(f"[FaceDetectionService] 人臉 {face['id']}: "
                        f"位置 {face['bbox']}, 置信度 {face['confidence']:.3f}, "
                        f"面積 {face['area']:.0f}")

    return image_pil, message, detection_results

def classify_by_face_count(image_path, face_count, faces_data=None, logger=None, config=None):
    """
    根據人臉數量分類圖片到不同資料夾
    
    Args:
        image_path: 圖片文件路徑
        face_count: 偵測到的人臉數量
        faces_data: 詳細的人臉資訊 (可選)
        logger: 日誌記錄器
        config: 配置對象
    
    Returns:
        tuple: (新路徑, 分類名稱, 分類統計)
    """
    try:
        base_dir = os.path.dirname(image_path)
        filename = os.path.basename(image_path)
        
        # 定義分類規則和資料夾名稱
        classification_rules = {
            0: {
                'folder': 'no_faces',
                'name': '無人臉',
                'description': '沒有偵測到人臉的圖片'
            },
            1: {
                'folder': 'single_face', 
                'name': '單人臉',
                'description': '只有一個人臉的圖片'
            },
            2: {
                'folder': 'two_faces',
                'name': '雙人臉', 
                'description': '有兩個人臉的圖片'
            }
        }
        
        # 3個或以上人臉歸類為多人臉
        if face_count >= 3:
            category_info = {
                'folder': 'multiple_faces',
                'name': '多人臉',
                'description': f'有{face_count}個人臉的圖片'
            }
        else:
            category_info = classification_rules.get(face_count, classification_rules[0])
        
        # 創建分類目標資料夾
        category_dir = os.path.join(base_dir, category_info['folder'])
        os.makedirs(category_dir, exist_ok=True)
        
        # 計算目標路徑，處理重名文件
        target_path = os.path.join(category_dir, filename)
        if os.path.exists(target_path):
            name, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(target_path):
                target_path = os.path.join(category_dir, f"{name}_{counter}{ext}")
                counter += 1
        
        # 移動文件到分類資料夾
        import shutil
        shutil.move(image_path, target_path)
        
        # 生成分類統計資訊
        classification_stats = {
            'original_path': image_path,
            'new_path': target_path,
            'category': category_info['folder'],
            'category_name': category_info['name'],
            'face_count': face_count,
            'description': category_info['description']
        }
        
        # 如果有詳細人臉資訊，加入統計
        if faces_data and len(faces_data) > 0:
            classification_stats['faces_info'] = {
                'avg_confidence': sum(f['confidence'] for f in faces_data) / len(faces_data),
                'total_area': sum(f['area'] for f in faces_data),
                'face_positions': [f['center'] for f in faces_data]
            }
        
        if logger:
            logger.info(f"[FaceDetectionService] 已將 {filename} 分類到 {category_info['name']} "
                       f"({category_info['folder']}) - {face_count} 個人臉")
        
        return target_path, category_info['name'], classification_stats
        
    except Exception as e:
        if logger:
            logger.error(f"[FaceDetectionService] 分類失敗 {image_path}: {e}")
        return image_path, "分類失敗", {'error': str(e)}

def filter_images_for_training(input_directory, logger, config=None):
    """
    根據人臉數量過濾圖片用於訓練
    """
    logger.info(f"[FaceDetectionService] 開始為訓練過濾圖片: {input_directory}")
    
    if not os.path.isdir(input_directory):
        return False, "輸入目錄不存在", {}
    
    # 讀取配置
    target_face_count = getattr(config, 'FACE_DETECTION_TARGET_FACE_COUNT', 1)
    filter_mode = getattr(config, 'FACE_DETECTION_FILTER_MODE', 'keep_target')
    excluded_dir_name = getattr(config, 'FACE_DETECTION_EXCLUDED_DIR', 'excluded_faces')
    training_dir_name = getattr(config, 'FACE_DETECTION_TRAINING_DIR', 'training_faces')
    
    logger.info(f"[FaceDetectionService] 目標人臉數量: {target_face_count}")
    logger.info(f"[FaceDetectionService] 過濾模式: {filter_mode}")
    
    # 創建輸出目錄
    base_dir = input_directory
    training_dir = os.path.join(base_dir, training_dir_name)
    excluded_dir = os.path.join(base_dir, excluded_dir_name)
    
    os.makedirs(training_dir, exist_ok=True)
    os.makedirs(excluded_dir, exist_ok=True)
    
    # 掃描圖片文件
    image_files = []
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp', '.tiff')
    
    for root, dirs, files in os.walk(input_directory):
        # 跳過已創建的訓練和排除目錄
        if training_dir_name in dirs:
            dirs.remove(training_dir_name)
        if excluded_dir_name in dirs:
            dirs.remove(excluded_dir_name)
            
        for filename in files:
            if filename.lower().endswith(supported_formats):
                image_files.append(os.path.join(root, filename))
    
    if not image_files:
        return False, "未找到圖片文件", {}
    
    logger.info(f"[FaceDetectionService] 找到 {len(image_files)} 個圖片文件")
    
    # 統計結果
    results = {
        "training_images": [],     # 符合訓練要求的圖片
        "excluded_images": [],     # 不符合要求的圖片
        "face_distribution": {},   # 人臉分佈統計
        "filter_stats": {
            "total_files": len(image_files),
            "processed": 0,
            "training_count": 0,
            "excluded_count": 0,
            "error_count": 0
        }
    }
    
    # 處理每個圖片
    for i, image_path in enumerate(image_files, 1):
        try:
            logger.info(f"[FaceDetectionService] 處理 ({i}/{len(image_files)}): {os.path.basename(image_path)}")
            
            # 載入圖片
            image_pil = Image.open(image_path)
            
            # 偵測人臉
            result_image, message, faces_data = detect_faces_service(image_pil, logger, config)
            
            if faces_data is not None:
                face_count = len(faces_data)
                
                # 更新人臉分佈統計
                results["face_distribution"][face_count] = results["face_distribution"].get(face_count, 0) + 1
                
                # 判斷是否符合訓練要求
                is_suitable_for_training = False
                
                if filter_mode == "keep_target":
                    # 保留目標人臉數量的圖片
                    if target_face_count == -1:  # 不限制人臉數量
                        is_suitable_for_training = True
                    else:
                        is_suitable_for_training = (face_count == target_face_count)
                        
                elif filter_mode == "exclude_target":
                    # 排除目標人臉數量的圖片
                    if target_face_count == -1:  # 不限制則都保留
                        is_suitable_for_training = True
                    else:
                        is_suitable_for_training = (face_count != target_face_count)
                        
                elif filter_mode == "classify_all":
                    # 只分類不過濾，都保留
                    is_suitable_for_training = True
                
                # 移動圖片到對應目錄
                filename = os.path.basename(image_path)
                
                if is_suitable_for_training:
                    target_path = os.path.join(training_dir, filename)
                    # 處理重名
                    if os.path.exists(target_path):
                        name, ext = os.path.splitext(filename)
                        counter = 1
                        while os.path.exists(target_path):
                            target_path = os.path.join(training_dir, f"{name}_{counter}{ext}")
                            counter += 1
                    
                    import shutil
                    shutil.move(image_path, target_path)
                    results["training_images"].append(target_path)
                    results["filter_stats"]["training_count"] += 1
                    
                    logger.info(f"[FaceDetectionService] ✓ 訓練圖片: {filename} ({face_count} 個人臉)")
                else:
                    target_path = os.path.join(excluded_dir, filename)
                    # 處理重名
                    if os.path.exists(target_path):
                        name, ext = os.path.splitext(filename)
                        counter = 1
                        while os.path.exists(target_path):
                            target_path = os.path.join(excluded_dir, f"{name}_{counter}{ext}")
                            counter += 1
                    
                    import shutil
                    shutil.move(image_path, target_path)
                    results["excluded_images"].append(target_path)
                    results["filter_stats"]["excluded_count"] += 1
                    
                    logger.info(f"[FaceDetectionService] ✗ 排除圖片: {filename} ({face_count} 個人臉)")
                
                results["filter_stats"]["processed"] += 1
            else:
                results["filter_stats"]["error_count"] += 1
                logger.error(f"[FaceDetectionService] ✗ 處理失敗: {os.path.basename(image_path)}")
            
            image_pil.close()
            
        except Exception as e:
            results["filter_stats"]["error_count"] += 1
            logger.error(f"[FaceDetectionService] ✗ 處理錯誤 {os.path.basename(image_path)}: {e}")
    
    # 生成摘要
    stats = results["filter_stats"]
    training_rate = (stats["training_count"] / stats["total_files"]) * 100 if stats["total_files"] > 0 else 0
    
    summary = (f"訓練圖片過濾完成. "
              f"處理: {stats['processed']}/{stats['total_files']}, "
              f"訓練用: {stats['training_count']}, "
              f"排除: {stats['excluded_count']}, "
              f"錯誤: {stats['error_count']}, "
              f"訓練可用率: {training_rate:.1f}%")
    
    logger.info(f"[FaceDetectionService] {summary}")
    logger.info(f"[FaceDetectionService] 人臉分佈: {results['face_distribution']}")
    logger.info(f"[FaceDetectionService] 訓練目錄: {training_dir} ({stats['training_count']} 個文件)")
    logger.info(f"[FaceDetectionService] 排除目錄: {excluded_dir} ({stats['excluded_count']} 個文件)")
    
    return True, summary, results

def detect_faces_batch_with_classification(input_directory, logger, config=None):
    """
    批量人臉偵測並按人臉數量自動分類
    
    Args:
        input_directory: 輸入圖片目錄
        logger: 日誌記錄器  
        config: 配置對象
    
    Returns:
        tuple: (成功標誌, 摘要訊息, 詳細結果)
    """
    logger.info(f"[FaceDetectionService] 開始批量人臉偵測和分類: {input_directory}")
    
    if not os.path.isdir(input_directory):
        return False, "輸入目錄不存在", {}
    
    # 掃描所有支援的圖片文件
    image_files = []
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp', '.tiff')
    
    for root, dirs, files in os.walk(input_directory):
        for filename in files:
            if filename.lower().endswith(supported_formats):
                image_files.append(os.path.join(root, filename))
    
    if not image_files:
        return False, "未找到圖片文件", {}
    
    logger.info(f"[FaceDetectionService] 找到 {len(image_files)} 個圖片文件")
    
    # 初始化結果統計
    classification_results = {
        "no_faces": [],       # 無人臉圖片列表
        "single_face": [],    # 單人臉圖片列表  
        "two_faces": [],      # 雙人臉圖片列表
        "multiple_faces": []  # 多人臉圖片列表
    }
    
    processing_stats = {
        "total_files": len(image_files),
        "processed": 0,
        "successful": 0,
        "failed": 0,
        "classification_enabled": getattr(config, 'FACE_DETECTION_AUTO_CLASSIFY', True),
        "face_distribution": {
            "0_faces": 0,
            "1_face": 0, 
            "2_faces": 0,
            "3_plus_faces": 0
        },
        "processing_details": []
    }
    
    # 處理每個圖片文件
    for i, image_path in enumerate(image_files, 1):
        try:
            logger.info(f"[FaceDetectionService] 處理 ({i}/{len(image_files)}): {os.path.basename(image_path)}")
            
            # 載入圖片
            image_pil = Image.open(image_path)
            
            # 執行人臉偵測
            result_image, message, faces_data = detect_faces_service(image_pil, logger, config)
            
            if faces_data is not None:
                face_count = len(faces_data)
                
                # 更新人臉分佈統計
                if face_count == 0:
                    processing_stats["face_distribution"]["0_faces"] += 1
                elif face_count == 1:
                    processing_stats["face_distribution"]["1_face"] += 1
                elif face_count == 2:
                    processing_stats["face_distribution"]["2_faces"] += 1
                else:
                    processing_stats["face_distribution"]["3_plus_faces"] += 1
                
                # 執行自動分類（如果啟用）
                classification_info = None
                if processing_stats["classification_enabled"]:
                    new_path, category_name, classification_stats = classify_by_face_count(
                        image_path, face_count, faces_data, logger, config
                    )
                    
                    classification_info = {
                        'new_path': new_path,
                        'category': category_name,
                        'stats': classification_stats
                    }
                    
                    # 記錄到對應分類結果
                    if face_count == 0:
                        classification_results["no_faces"].append(new_path)
                    elif face_count == 1:
                        classification_results["single_face"].append(new_path)
                    elif face_count == 2:
                        classification_results["two_faces"].append(new_path)
                    else:
                        classification_results["multiple_faces"].append(new_path)
                
                # 記錄處理詳情
                processing_detail = {
                    'original_path': image_path,
                    'face_count': face_count,
                    'faces_data': faces_data,
                    'classification': classification_info,
                    'message': message
                }
                processing_stats["processing_details"].append(processing_detail)
                
                processing_stats["successful"] += 1
                logger.info(f"[FaceDetectionService] ✓ {os.path.basename(image_path)}: "
                           f"{face_count} 個人臉" + 
                           (f" -> {classification_info['category']}" if classification_info else ""))
            else:
                processing_stats["failed"] += 1
                logger.error(f"[FaceDetectionService] ✗ 處理失敗: {os.path.basename(image_path)}")
            
            processing_stats["processed"] += 1
            image_pil.close()
            
        except Exception as e:
            processing_stats["failed"] += 1
            processing_stats["processed"] += 1
            logger.error(f"[FaceDetectionService] ✗ 處理錯誤 {os.path.basename(image_path)}: {e}")
    
    # 生成處理摘要
    success_rate = (processing_stats["successful"] / processing_stats["total_files"]) * 100
    
    summary_parts = [
        f"批量人臉偵測完成",
        f"處理: {processing_stats['processed']}/{processing_stats['total_files']}",
        f"成功: {processing_stats['successful']}",
        f"失敗: {processing_stats['failed']}",
        f"成功率: {success_rate:.1f}%"
    ]
    
    if processing_stats["classification_enabled"]:
        summary_parts.append("已啟用自動分類")
    
    summary = ". ".join(summary_parts)
    
    # 記錄詳細統計
    logger.info(f"[FaceDetectionService] {summary}")
    logger.info(f"[FaceDetectionService] 人臉分佈統計: {processing_stats['face_distribution']}")
    
    if processing_stats["classification_enabled"]:
        category_counts = {k: len(v) for k, v in classification_results.items()}
        logger.info(f"[FaceDetectionService] 分類結果統計: {category_counts}")
        
        # 顯示分類資料夾資訊
        for category, files in classification_results.items():
            if files:
                category_dir = os.path.dirname(files[0])
                logger.info(f"[FaceDetectionService] {category} 資料夾: {category_dir} ({len(files)} 個文件)")
    
    # 構建完整結果
    result_data = {
        "classification_results": classification_results,
        "processing_stats": processing_stats,
        "summary": {
            "total_processed": processing_stats["processed"],
            "success_count": processing_stats["successful"], 
            "error_count": processing_stats["failed"],
            "success_rate": success_rate,
            "face_distribution": processing_stats["face_distribution"],
            "classification_enabled": processing_stats["classification_enabled"]
        }
    }
    
    return True, summary, result_data

def detect_faces_service_entry(image_pil: Image.Image, logger, config=None):
    """
    Entry point for orchestrator - detects faces in PIL image.
    Returns: (pil_image, output_path, faces_data, message)
    """
    try:
        result_image, message, faces_data = detect_faces_service(image_pil, logger, config)
        # For orchestrator compatibility, return format: (pil_image, output_path, faces_data, message)
        return result_image, None, faces_data, message
    except Exception as e:
        logger.error(f"[FaceDetectionService] Error in detect_faces_service_entry: {e}")
        return image_pil, None, None, f"Face detection error: {str(e)}"
