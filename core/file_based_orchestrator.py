# core/file_based_orchestrator.py
"""
基於檔案路徑傳遞的工作流orchestrator
解決記憶體↔檔案轉換的根本性架構問題
"""

import os
import shutil
import tempfile
import logging
from typing import Optional, List, Dict, Tuple
from pathlib import Path

from config import settings as default_settings


class FileBasedOrchestrator:
    """
    基於檔案路徑傳遞的工作流orchestrator
    
    核心設計原則：
    1. 每個服務接受檔案路徑，返回檔案路徑
    2. 工作目錄統一管理中間檔案
    3. 數據流清晰：file_path → service → file_path
    4. 直接對應原始main.py的設計思路
    """
    
    def __init__(self, config=None, logger: Optional[logging.Logger] = None):
        self.config = config if config else default_settings
        
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
        
        # 定義基於檔案的服務映射
        self.step_definitions = {
            "validate": {
                "service": self._validate_file_service,
                "flag": "ENABLE_VALIDATION",
                "description": "驗證檔案完整性"
            },
            "transparency": {
                "service": self._transparency_file_service,
                "flag": "ENABLE_TRANSPARENCY_PROCESSING", 
                "description": "透明背景處理"
            },
            "face_detect": {
                "service": self._face_detect_file_service,
                "flag": "ENABLE_FACE_DETECTION",
                "description": "人臉檢測"
            },
            "cluster": {
                "service": self._cluster_file_service,
                "flag": "ENABLE_LPIPS_CLUSTERING",
                "description": "LPIPS聚類去重"
            },
            "crop": {
                "service": self._crop_file_service,
                "flag": "ENABLE_CROP",
                "description": "圖片裁切"
            },
            "classify": {
                "service": self._classify_file_service,
                "flag": "ENABLE_CLASSIFICATION",
                "description": "檔案分類"
            },
            "upscale": {
                "service": self._upscale_file_service,
                "flag": "ENABLE_UPSCALE",
                "description": "圖片放大"
            },
            "tag": {
                "service": self._tag_file_service,
                "flag": "ENABLE_TAGGING",
                "description": "圖片標記"
            }
        }
        
        # 完整的執行順序（對應原始main.py）
        self.step_execution_order = [
            "validate", "transparency", "face_detect", "cluster", 
            "crop", "classify", "upscale", "tag"
        ]
    
    def create_work_directory(self, base_name: str = "waifuc_work") -> str:
        """
        創建工作目錄用於中間檔案
        
        Returns:
            work_dir: 工作目錄路徑
        """
        work_dir = tempfile.mkdtemp(prefix=f"{base_name}_")
        self.logger.info(f"[FileOrchestrator] Created work directory: {work_dir}")
        return work_dir
    
    def cleanup_work_directory(self, work_dir: str):
        """清理工作目錄"""
        try:
            shutil.rmtree(work_dir)
            self.logger.info(f"[FileOrchestrator] Cleaned up work directory: {work_dir}")
        except Exception as e:
            self.logger.warning(f"[FileOrchestrator] Failed to cleanup work directory {work_dir}: {e}")
    
    def process_single_file(self, input_path: str, selected_steps: List[str] = None, 
                           output_dir: str = None, cleanup: bool = True) -> Dict:
        """
        處理單一檔案的完整工作流
        
        Args:
            input_path: 輸入檔案路徑
            selected_steps: 選擇的處理步驟
            output_dir: 輸出目錄（可選）
            cleanup: 是否清理中間檔案
            
        Returns:
            處理結果字典
        """
        if selected_steps is None:
            # 根據配置選擇啟用的步驟
            selected_steps = []
            for step_key, step_config in self.step_definitions.items():
                flag_name = step_config["flag"]
                if getattr(self.config, flag_name, False):
                    selected_steps.append(step_key)
        
        if output_dir is None:
            output_dir = getattr(self.config, 'OUTPUT_DIR', 'output_images')
        
        self.logger.info(f"[FileOrchestrator] Processing file: {input_path}")
        self.logger.info(f"[FileOrchestrator] Selected steps: {selected_steps}")
        
        # 創建工作目錄
        work_dir = self.create_work_directory()
        
        try:
            # 複製輸入檔案到工作目錄
            input_filename = os.path.basename(input_path)
            current_path = os.path.join(work_dir, f"input_{input_filename}")
            shutil.copy2(input_path, current_path)
            
            # 執行工作流
            step_results = {}
            pipeline_messages = []
            
            for step_key in self.step_execution_order:
                if step_key in selected_steps:
                    step_config = self.step_definitions[step_key]
                    
                    # 檢查步驟是否啟用
                    flag_name = step_config["flag"]
                    if not getattr(self.config, flag_name, False):
                        self.logger.info(f"[FileOrchestrator] {step_key} step skipped (disabled)")
                        continue
                    
                    self.logger.info(f"[FileOrchestrator] Executing step: {step_key}")
                    
                    try:
                        # 執行步驟服務
                        service_function = step_config["service"]
                        output_path, metadata = service_function(
                            input_path=current_path,
                            work_dir=work_dir,
                            step_name=step_key
                        )
                        
                        if output_path and os.path.exists(output_path):
                            current_path = output_path
                            step_results[step_key] = {
                                "output_path": output_path,
                                "metadata": metadata
                            }
                            
                            message = f"{step_key}: {metadata.get('message', 'Completed')}"
                            pipeline_messages.append(message)
                            self.logger.info(f"[FileOrchestrator] {step_key} completed: {output_path}")
                        else:
                            error_msg = f"{step_key}: Failed to produce output"
                            pipeline_messages.append(error_msg)
                            self.logger.error(f"[FileOrchestrator] {error_msg}")
                            
                    except Exception as e:
                        error_msg = f"{step_key}: Exception - {str(e)}"
                        pipeline_messages.append(error_msg)
                        self.logger.error(f"[FileOrchestrator] {error_msg}", exc_info=True)
            
            # 複製最終結果到輸出目錄
            final_output_path = None
            if current_path and os.path.exists(current_path):
                os.makedirs(output_dir, exist_ok=True)
                final_filename = f"processed_{input_filename}"
                final_output_path = os.path.join(output_dir, final_filename)
                
                # 避免檔名衝突
                counter = 1
                while os.path.exists(final_output_path):
                    name, ext = os.path.splitext(final_filename)
                    final_output_path = os.path.join(output_dir, f"{name}_{counter}{ext}")
                    counter += 1
                
                shutil.copy2(current_path, final_output_path)
                self.logger.info(f"[FileOrchestrator] Final output saved to: {final_output_path}")
            
            # 準備返回結果
            result = {
                "success": final_output_path is not None,
                "final_output_path": final_output_path,
                "work_directory": work_dir,
                "step_results": step_results,
                "pipeline_summary": "\\n".join(pipeline_messages),
                "selected_steps": selected_steps,
                "message": "Processing completed successfully" if final_output_path else "Processing failed"
            }
            
            return result
            
        finally:
            # 清理工作目錄（如果要求）
            if cleanup:
                self.cleanup_work_directory(work_dir)
    
    # =================== 檔案服務實現 ===================
    
    def _validate_file_service(self, input_path: str, work_dir: str, step_name: str) -> Tuple[str, Dict]:
        """檔案驗證服務 - 檢查檔案完整性"""
        try:
            from PIL import Image
            
            # 嘗試打開並驗證圖片
            with Image.open(input_path) as img:
                img.verify()
            
            # 重新打開確保可讀取
            with Image.open(input_path) as img:
                img.load()
            
            self.logger.info(f"[FileOrchestrator] File validation passed: {input_path}")
            
            # 驗證通過，返回原始檔案
            return input_path, {
                "status": "valid",
                "message": "File validation passed"
            }
            
        except Exception as e:
            self.logger.error(f"[FileOrchestrator] File validation failed: {e}")
            # 驗證失敗，但不刪除檔案（交由調用者決定）
            return None, {
                "status": "invalid",
                "message": f"File validation failed: {str(e)}"
            }
    
    def _transparency_file_service(self, input_path: str, work_dir: str, step_name: str) -> Tuple[str, Dict]:
        """透明背景處理服務"""
        try:
            from PIL import Image
            
            output_filename = f"{step_name}_{os.path.basename(input_path)}"
            output_path = os.path.join(work_dir, output_filename)
            
            with Image.open(input_path) as img:
                # 檢查是否有透明通道
                if img.mode in ('RGBA', 'LA') or 'transparency' in img.info:
                    # 轉換為白色背景
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'RGBA':
                        background.paste(img, mask=img.split()[-1])
                    else:
                        background.paste(img, (0, 0))
                    
                    background.save(output_path)
                    message = "Converted transparent background to white"
                else:
                    # 沒有透明通道，直接複製
                    shutil.copy2(input_path, output_path)
                    message = "No transparency found, image unchanged"
            
            return output_path, {
                "status": "processed",
                "message": message
            }
            
        except Exception as e:
            self.logger.error(f"[FileOrchestrator] Transparency processing failed: {e}")
            return input_path, {
                "status": "failed",
                "message": f"Transparency processing failed: {str(e)}"
            }
    
    def _face_detect_file_service(self, input_path: str, work_dir: str, step_name: str) -> Tuple[str, Dict]:
        """人臉檢測服務 - 基於檔案"""
        try:
            # 這裡直接對應原始face_detection.py的邏輯
            # 但為了簡化，這個版本只是檢測而不移動檔案
            from imgutils.detect import detect_faces
            
            result = detect_faces(input_path)
            face_count = len(result)
            
            # 在工作目錄中創建副本（可選：可以在圖片上畫框）
            output_filename = f"{step_name}_{os.path.basename(input_path)}"
            output_path = os.path.join(work_dir, output_filename)
            shutil.copy2(input_path, output_path)
            
            self.logger.info(f"[FileOrchestrator] Detected {face_count} faces in {input_path}")
            
            return output_path, {
                "status": "detected",
                "face_count": face_count,
                "faces_data": result,
                "message": f"Detected {face_count} faces"
            }
            
        except Exception as e:
            self.logger.error(f"[FileOrchestrator] Face detection failed: {e}")
            return input_path, {
                "status": "failed",
                "face_count": 0,
                "message": f"Face detection failed: {str(e)}"
            }
    
    def _cluster_file_service(self, input_path: str, work_dir: str, step_name: str) -> Tuple[str, Dict]:
        """LPIPS聚類服務 - 需要批量處理，單檔案版本暫時跳過"""
        # 聚類需要多個檔案，單檔案處理時跳過
        self.logger.info(f"[FileOrchestrator] Clustering skipped for single file processing")
        return input_path, {
            "status": "skipped",
            "message": "Clustering is only meaningful for batch processing"
        }
    
    def _crop_file_service(self, input_path: str, work_dir: str, step_name: str) -> Tuple[str, Dict]:
        """圖片裁切服務 - 直接使用waifuc"""
        try:
            from waifuc.action import ThreeStageSplitAction
            from waifuc.export import SaveExporter
            from waifuc.source import LocalSource
            
            # 創建輸出目錄
            crop_output_dir = os.path.join(work_dir, f"{step_name}_output")
            os.makedirs(crop_output_dir, exist_ok=True)
            
            # 將檔案複製到臨時目錄供waifuc處理
            temp_input_dir = os.path.join(work_dir, f"{step_name}_input")
            os.makedirs(temp_input_dir, exist_ok=True)
            temp_input_path = os.path.join(temp_input_dir, os.path.basename(input_path))
            shutil.copy2(input_path, temp_input_path)
            
            # 使用waifuc進行裁切
            source = LocalSource(temp_input_dir)
            source.attach(ThreeStageSplitAction()).export(SaveExporter(crop_output_dir))
            
            # 找到輸出檔案
            output_files = [f for f in os.listdir(crop_output_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if output_files:
                # 返回第一個裁切結果（可以改為返回最佳結果）
                primary_output = os.path.join(crop_output_dir, output_files[0])
                
                return primary_output, {
                    "status": "cropped", 
                    "crop_count": len(output_files),
                    "output_files": output_files,
                    "message": f"Generated {len(output_files)} crops"
                }
            else:
                # 沒有產生裁切結果，返回原檔案
                return input_path, {
                    "status": "no_crops",
                    "crop_count": 0,
                    "message": "No crops generated by waifuc"
                }
                
        except Exception as e:
            self.logger.error(f"[FileOrchestrator] Crop processing failed: {e}")
            return input_path, {
                "status": "failed",
                "message": f"Crop processing failed: {str(e)}"
            }
    
    def _classify_file_service(self, input_path: str, work_dir: str, step_name: str) -> Tuple[str, Dict]:
        """檔案分類服務 - 根據檔名分類"""
        try:
            filename = os.path.basename(input_path)
            
            # 根據檔名特徵決定分類（對應原始crop.py的邏輯）
            if "_person1_head" in filename:
                category = "head"
            elif "_person1_halfbody" in filename:
                category = "halfbody"
            elif "_person1" in filename:
                category = "person"
            else:
                category = "general"
            
            # 創建分類目錄
            category_dir = os.path.join(work_dir, f"{step_name}_{category}")
            os.makedirs(category_dir, exist_ok=True)
            
            # 移動檔案到分類目錄
            output_path = os.path.join(category_dir, filename)
            shutil.copy2(input_path, output_path)
            
            return output_path, {
                "status": "classified",
                "category": category,
                "message": f"Classified as {category}"
            }
            
        except Exception as e:
            self.logger.error(f"[FileOrchestrator] Classification failed: {e}")
            return input_path, {
                "status": "failed",
                "message": f"Classification failed: {str(e)}"
            }
    
    def _upscale_file_service(self, input_path: str, work_dir: str, step_name: str) -> Tuple[str, Dict]:
        """圖片放大服務 - 直接檔案處理"""
        try:
            from imgutils.upscale import upscale_with_cdc
            
            model_name = getattr(self.config, 'UPSCALE_MODEL_NAME', 'HGSR-MHR-anime-aug_X4_320')
            
            # 使用imgutils直接處理檔案
            upscaled_image = upscale_with_cdc(input_path, model_name)
            
            # 保存結果
            output_filename = f"{step_name}_{os.path.basename(input_path)}"
            output_path = os.path.join(work_dir, output_filename)
            upscaled_image.save(output_path)
            
            return output_path, {
                "status": "upscaled",
                "model": model_name,
                "original_size": f"{upscaled_image.size}",
                "message": f"Upscaled using {model_name}"
            }
            
        except Exception as e:
            self.logger.error(f"[FileOrchestrator] Upscale processing failed: {e}")
            return input_path, {
                "status": "failed",
                "message": f"Upscale processing failed: {str(e)}"
            }
    
    def _tag_file_service(self, input_path: str, work_dir: str, step_name: str) -> Tuple[str, Dict]:
        """圖片標記服務 - 生成標籤檔案"""
        try:
            from imgutils.tagging import get_wd14_tags
            
            model_name = getattr(self.config, 'TAG_MODEL_NAME', 'EVA02_Large')
            
            # 使用imgutils生成標籤
            rating, features, chars = get_wd14_tags(input_path, model_name=model_name)
            
            # 處理標籤（簡化版，可以增強）
            tag_parts = []
            if chars:
                tag_parts.extend(chars.keys())
            if features:
                # 過濾高信心度的特徵
                threshold = getattr(self.config, 'TAG_GENERAL_THRESHOLD', 0.35)
                filtered_features = [tag for tag, conf in features.items() if conf >= threshold]
                tag_parts.extend(filtered_features)
            
            tags_string = ", ".join(tag_parts)
            
            # 保存標籤檔案（對應原始tag.py的輸出）
            tag_filename = os.path.splitext(os.path.basename(input_path))[0] + ".txt"
            tag_output_path = os.path.join(work_dir, tag_filename)
            
            with open(tag_output_path, 'w', encoding='utf-8') as f:
                f.write(tags_string)
            
            # 複製原圖到工作目錄（標籤和圖片配對）
            image_output_filename = f"{step_name}_{os.path.basename(input_path)}"
            image_output_path = os.path.join(work_dir, image_output_filename)
            shutil.copy2(input_path, image_output_path)
            
            return image_output_path, {
                "status": "tagged",
                "tags": tags_string,
                "tag_file": tag_output_path,
                "tag_count": len(tag_parts),
                "model": model_name,
                "message": f"Generated {len(tag_parts)} tags using {model_name}"
            }
            
        except Exception as e:
            self.logger.error(f"[FileOrchestrator] Tagging failed: {e}")
            return input_path, {
                "status": "failed",
                "message": f"Tagging failed: {str(e)}"
            }