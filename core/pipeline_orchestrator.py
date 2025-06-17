#!/usr/bin/env python3
"""
智能管道編排器 - 處理步驟間的目錄依賴關係
解決人臉偵測後裁切應該從過濾結果讀取的問題
"""

import os
import logging
from typing import Optional, Dict, List
from services import (
    validator_service, 
    face_detection_service, 
    crop_service, 
    tag_service, 
    upscale_service, 
    lpips_clustering_service
)
from utils.file_utils import scan_directory_for_images
from utils.error_handler import safe_execute
from config import settings as default_settings

class PipelineOrchestrator:
    """
    智能管道編排器 - 處理步驟間的依賴關係
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
        
        # 步驟執行順序和依賴關係
        self.step_order = ["validate", "face_detect", "cluster", "crop", "tag", "upscale"]
        
        # 管道狀態：記錄每個步驟的輸出目錄
        self.pipeline_state = {
            "current_input_dir": None,
            "step_outputs": {},
            "processing_stats": {
                "total_steps": 0,
                "completed_steps": 0,
                "failed_steps": 0,
                "step_details": {}
            }
        }


    def _determine_next_working_directory(self, completed_step: str, step_result: Dict, current_dir: str) -> str:
        """
        根據完成的步驟結果決定下一個步驟的工作目錄
        
        Args:
            completed_step: 剛完成的步驟名稱
            step_result: 步驟執行結果
            current_dir: 當前工作目錄
            
        Returns:
            下一個步驟應該使用的工作目錄
        """
        # 人臉偵測步驟 - 如果有過濾，下一步使用訓練目錄
        if completed_step == "face_detect":
            if step_result.get("filter_applied", False) and step_result.get("training_directory"):
                training_dir = step_result["training_directory"]
                if os.path.exists(training_dir):
                    self.logger.info(f"[PipelineOrchestrator] 人臉過濾完成，後續步驟使用訓練目錄: {training_dir}")
                    return training_dir
        
        # 聚類步驟 - 通常在原目錄進行，但可能創建子目錄
        elif completed_step == "cluster":
            cluster_output = step_result.get("output_directory")
            if cluster_output and os.path.exists(cluster_output):
                return cluster_output
        
        # 裁切步驟 - 使用裁切輸出目錄
        elif completed_step == "crop":
            crop_output = step_result.get("output_directory")
            if crop_output and os.path.exists(crop_output):
                self.logger.info(f"[PipelineOrchestrator] 裁切完成，後續步驟使用裁切目錄: {crop_output}")
                return crop_output
        
        # 放大步驟 - 使用放大輸出目錄  
        elif completed_step == "upscale":
            upscale_output = step_result.get("output_directory")
            if upscale_output and os.path.exists(upscale_output):
                self.logger.info(f"[PipelineOrchestrator] 放大完成，後續步驟使用放大目錄: {upscale_output}")
                return upscale_output
        
        # 驗證和標記步驟通常不改變工作目錄
        # 如果沒有特殊輸出目錄，繼續使用當前目錄
        return current_dir

    def _execute_validation_step(self, input_dir: str) -> Dict:
        """執行圖片驗證步驟"""
        self.logger.info(f"[PipelineOrchestrator] 執行圖片驗證: {input_dir}")
        
        try:
            is_valid, message, valid_paths = validator_service.validate_image_service(
                input_dir, self.logger, self.config, is_directory=True
            )
            
            result = {
                "success": is_valid,
                "message": message,
                "valid_count": len(valid_paths) if valid_paths else 0,
                "output_directory": input_dir  # 驗證不改變目錄
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"[PipelineOrchestrator] 驗證步驟失敗: {e}")
            return {"success": False, "message": f"驗證失敗: {str(e)}"}

    def _execute_face_detection_step(self, input_dir: str) -> Dict:
        """執行人臉偵測步驟"""
        self.logger.info(f"[PipelineOrchestrator] 執行人臉偵測和過濾: {input_dir}")
        
        try:
            # 使用新的訓練導向過濾功能
            success, summary, results = face_detection_service.filter_images_for_training(
                input_dir, self.logger, self.config
            )
            
            if success and results:
                # 檢查是否有訓練圖片產生
                training_count = results.get("filter_stats", {}).get("training_count", 0)
                training_dir = None
                
                if training_count > 0 and results.get("training_images"):
                    # 獲取訓練目錄
                    first_training_image = results["training_images"][0]
                    training_dir = os.path.dirname(first_training_image)
                
                result = {
                    "success": True,
                    "message": summary,
                    "training_directory": training_dir,
                    "training_count": training_count,
                    "excluded_count": results.get("filter_stats", {}).get("excluded_count", 0),
                    "face_distribution": results.get("face_distribution", {}),
                    "filter_applied": True,
                    "output_directory": training_dir  # 後續步驟使用訓練目錄
                }
            else:
                result = {
                    "success": False,
                    "message": summary,
                    "filter_applied": False
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"[PipelineOrchestrator] 人臉偵測步驟失敗: {e}")
            return {"success": False, "message": f"人臉偵測失敗: {str(e)}"}

    def _execute_crop_step(self, input_dir: str) -> Dict:
        """執行圖片裁切步驟"""
        self.logger.info(f"[PipelineOrchestrator] 執行圖片裁切: {input_dir}")
        
        try:
            # 創建裁切輸出目錄
            crop_output_dir = os.path.join(os.path.dirname(input_dir), "cropped_images")
            
            success, summary, results = crop_service.crop_batch_with_categorization(
                input_dir, crop_output_dir, self.logger, self.config
            )
            
            result = {
                "success": success,
                "message": summary,
                "output_directory": crop_output_dir,
                "crop_categories": results.get("crop_categories", {}),
                "successful_crops": results.get("successful_crops", 0),
                "failed_crops": results.get("failed_crops", 0)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"[PipelineOrchestrator] 裁切步驟失敗: {e}")
            return {"success": False, "message": f"裁切失敗: {str(e)}"}

    def _execute_clustering_step(self, input_dir: str) -> Dict:
        """執行LPIPS聚類步驟"""
        self.logger.info(f"[PipelineOrchestrator] 執行LPIPS聚類: {input_dir}")
        
        try:
            # 掃描圖片文件
            image_files = scan_directory_for_images(input_dir, recursive=True)
            
            if not image_files:
                return {"success": False, "message": "未找到圖片文件進行聚類"}
            
            results, message = lpips_clustering_service.cluster_images_service_entry(
                image_files, self.logger, self.config
            )
            
            result = {
                "success": True,
                "message": message,
                "output_directory": input_dir,  # 聚類會修改原目錄結構
                "cluster_results": results
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"[PipelineOrchestrator] 聚類步驟失敗: {e}")
            return {"success": False, "message": f"聚類失敗: {str(e)}"}

    def _execute_tagging_step(self, input_dir: str) -> Dict:
        """執行圖片標記步驟"""
        self.logger.info(f"[PipelineOrchestrator] 執行圖片標記: {input_dir}")
        
        try:
            success, summary, results = tag_service.tag_batch_images(
                input_dir, self.logger, self.config
            )
            
            result = {
                "success": success,
                "message": summary,
                "output_directory": input_dir,  # 標記不改變目錄結構
                "successful_tags": results.get("successful_tags", 0),
                "failed_tags": results.get("failed_tags", 0),
                "total_tags_generated": results.get("total_tags_generated", 0)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"[PipelineOrchestrator] 標記步驟失敗: {e}")
            return {"success": False, "message": f"標記失敗: {str(e)}"}

    def _execute_upscale_step(self, input_dir: str) -> Dict:
        """執行圖片放大步驟"""
        self.logger.info(f"[PipelineOrchestrator] 執行圖片放大: {input_dir}")
        
        try:
            # 創建放大輸出目錄
            upscale_output_dir = os.path.join(os.path.dirname(input_dir), "upscaled_images")
            
            success, summary, results = upscale_service.upscale_batch_images(
                input_dir, upscale_output_dir, self.logger, self.config
            )
            
            result = {
                "success": success,
                "message": summary,
                "output_directory": upscale_output_dir,
                "successful_upscales": results.get("successful_upscales", 0),
                "failed_upscales": results.get("failed_upscales", 0),
                "skipped_files": results.get("skipped_files", 0)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"[PipelineOrchestrator] 放大步驟失敗: {e}")
            return {"success": False, "message": f"放大失敗: {str(e)}"}

    def process_pipeline(self, input_directory: str, selected_steps: List[str] = None) -> Dict:
        """
        執行完整的批量處理管道 - 正確的目錄級別步驟順序執行
        
        批量處理邏輯：
        1. 步驟1完成整個目錄處理 → 步驟2處理步驟1的輸出目錄 → ...
        2. 每個步驟處理完整個目錄後才進入下一個步驟
        3. 後續步驟自動使用前一步驟的輸出目錄作為輸入
        
        Args:
            input_directory: 初始輸入目錄
            selected_steps: 要執行的步驟列表
            
        Returns:
            處理結果字典
        """
        if selected_steps is None:
            selected_steps = self.step_order
        
        self.logger.info(f"[PipelineOrchestrator] 🚀 開始批量管道處理")
        self.logger.info(f"[PipelineOrchestrator] 📁 初始輸入目錄: {input_directory}")
        self.logger.info(f"[PipelineOrchestrator] 🔧 選定步驟: {selected_steps}")
        self.logger.info(f"[PipelineOrchestrator] 📋 批量處理模式: 目錄級別步驟順序執行")
        
        # 重置管道狀態
        self.pipeline_state = {
            "current_input_dir": input_directory,
            "step_outputs": {},
            "processing_stats": {
                "total_steps": len(selected_steps),
                "completed_steps": 0,
                "failed_steps": 0,
                "step_details": {}
            }
        }
        
        # 步驟執行器映射
        step_executors = {
            "validate": self._execute_validation_step,
            "face_detect": self._execute_face_detection_step,
            "crop": self._execute_crop_step,
            "cluster": self._execute_clustering_step,
            "tag": self._execute_tagging_step,
            "upscale": self._execute_upscale_step
        }
        
        # 🔄 按順序執行步驟 - 每個步驟完成整個目錄處理
        current_working_dir = input_directory
        
        for step_name in self.step_order:
            if step_name in selected_steps:
                try:
                    self.logger.info(f"[PipelineOrchestrator] 🔄 開始批量執行步驟: {step_name}")
                    self.logger.info(f"[PipelineOrchestrator] 📂 當前工作目錄: {current_working_dir}")
                    
                    # 執行步驟 - 處理整個目錄
                    executor = step_executors.get(step_name)
                    if executor:
                        step_result = executor(current_working_dir)
                        
                        # 記錄步驟結果
                        self.pipeline_state["step_outputs"][step_name] = step_result
                        self.pipeline_state["processing_stats"]["step_details"][step_name] = step_result
                        
                        if step_result.get("success", False):
                            self.pipeline_state["processing_stats"]["completed_steps"] += 1
                            self.logger.info(f"[PipelineOrchestrator] ✅ 步驟 {step_name} 完成: {step_result.get('message', '')}")
                            
                            # 🎯 更新下一個步驟的工作目錄
                            next_working_dir = self._determine_next_working_directory(step_name, step_result, current_working_dir)
                            if next_working_dir != current_working_dir:
                                self.logger.info(f"[PipelineOrchestrator] 📂 工作目錄更新: {current_working_dir} → {next_working_dir}")
                                current_working_dir = next_working_dir
                        else:
                            self.pipeline_state["processing_stats"]["failed_steps"] += 1
                            self.logger.error(f"[PipelineOrchestrator] ❌ 步驟 {step_name} 失敗: {step_result.get('message', '')}")
                            # 失敗時不更新工作目錄，後續步驟繼續使用當前目錄
                    else:
                        self.logger.error(f"[PipelineOrchestrator] ❌ 未知步驟: {step_name}")
                        self.pipeline_state["processing_stats"]["failed_steps"] += 1
                        
                except Exception as e:
                    self.pipeline_state["processing_stats"]["failed_steps"] += 1
                    self.logger.error(f"[PipelineOrchestrator] ❌ 步驟 {step_name} 執行異常: {e}")
        
        # 生成最終結果
        stats = self.pipeline_state["processing_stats"]
        success_rate = (stats["completed_steps"] / stats["total_steps"]) * 100 if stats["total_steps"] > 0 else 0
        
        final_result = {
            "success": stats["failed_steps"] == 0,
            "message": f"批量管道處理完成. 成功: {stats['completed_steps']}/{stats['total_steps']} ({success_rate:.1f}%)",
            "pipeline_state": self.pipeline_state,
            "step_outputs": self.pipeline_state["step_outputs"],
            "processing_stats": stats,
            "final_working_dir": current_working_dir  # 最終的工作目錄
        }
        
        self.logger.info(f"[PipelineOrchestrator] 🎉 {final_result['message']}")
        self.logger.info(f"[PipelineOrchestrator] 📂 最終輸出目錄: {current_working_dir}")
        
        return final_result