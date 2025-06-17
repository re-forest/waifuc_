# core/ui_adapter.py
"""
UI適配器 - 連接檔案導向orchestrator與Gradio UI
處理檔案上傳、下載和預覽功能
"""

import os
import tempfile
import shutil
from typing import Optional, List, Dict, Tuple
from pathlib import Path

from .file_based_orchestrator import FileBasedOrchestrator
from .pipeline_orchestrator import PipelineOrchestrator


class UIAdapter:
    """
    UI適配器類別
    
    負責：
    1. 處理Gradio檔案上傳/下載
    2. 調用檔案導向orchestrator
    3. 提供預覽功能
    4. 管理臨時檔案
    """
    
    def __init__(self, config=None, logger=None):
        self.orchestrator = FileBasedOrchestrator(config, logger)  # 用於單檔案處理
        self.pipeline_orchestrator = PipelineOrchestrator(config, logger)  # 用於批量處理
        self.config = config
        self.logger = logger or self.orchestrator.logger
        
        # 創建臨時目錄用於UI檔案交換
        self.temp_upload_dir = tempfile.mkdtemp(prefix="waifuc_ui_uploads_")
        self.temp_preview_dir = tempfile.mkdtemp(prefix="waifuc_ui_previews_")
        
        self.logger.info(f"[UIAdapter] Initialized with upload dir: {self.temp_upload_dir}")
        self.logger.info(f"[UIAdapter] Initialized with preview dir: {self.temp_preview_dir}")
    
    def __del__(self):
        """清理臨時目錄"""
        try:
            shutil.rmtree(self.temp_upload_dir, ignore_errors=True)
            shutil.rmtree(self.temp_preview_dir, ignore_errors=True)
        except:
            pass
    
    def process_uploaded_image(self, uploaded_file_path: str, selected_steps: List[str], 
                              preview_mode: bool = True) -> Dict:
        """
        處理UI上傳的圖片
        
        Args:
            uploaded_file_path: Gradio上傳的檔案路徑
            selected_steps: 選擇的處理步驟
            preview_mode: 是否為預覽模式
            
        Returns:
            處理結果字典，包含預覽路徑等
        """
        try:
            if not uploaded_file_path or not os.path.exists(uploaded_file_path):
                return {
                    "success": False,
                    "message": "沒有上傳檔案或檔案不存在",
                    "preview_image_path": None,
                    "detailed_log": "錯誤：檔案路徑無效"
                }
            
            self.logger.info(f"[UIAdapter] Processing uploaded file: {uploaded_file_path}")
            self.logger.info(f"[UIAdapter] Selected steps: {selected_steps}")
            
            # 複製上傳檔案到我們的管理目錄
            filename = os.path.basename(uploaded_file_path)
            managed_input_path = os.path.join(self.temp_upload_dir, filename)
            shutil.copy2(uploaded_file_path, managed_input_path)
            
            # 決定輸出目錄
            if preview_mode:
                output_dir = self.temp_preview_dir
                cleanup = False  # 預覽模式保留檔案供UI顯示
            else:
                output_dir = getattr(self.config, 'OUTPUT_DIR', 'output_images')
                cleanup = True   # 生產模式清理中間檔案
            
            # 調用檔案導向orchestrator
            result = self.orchestrator.process_single_file(
                input_path=managed_input_path,
                selected_steps=selected_steps,
                output_dir=output_dir,
                cleanup=cleanup
            )
            
            # 準備UI返回格式
            ui_result = {
                "success": result["success"],
                "message": result["message"],
                "pipeline_summary": result["pipeline_summary"],
                "detailed_log": self._format_detailed_log(result),
                "preview_image_path": result.get("final_output_path"),
                "final_image_path": result.get("final_output_path"),
                "step_results": result.get("step_results", {}),
                "tags": self._extract_tags_from_result(result)
            }
            
            # 如果是預覽模式，確保圖片在預覽目錄中
            if preview_mode and result["success"] and result.get("final_output_path"):
                preview_path = self._prepare_preview_image(result["final_output_path"])
                ui_result["preview_image_path"] = preview_path
            
            self.logger.info(f"[UIAdapter] Processing completed. Success: {result['success']}")
            return ui_result
            
        except Exception as e:
            self.logger.error(f"[UIAdapter] Error processing uploaded image: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"處理過程發生錯誤：{str(e)}",
                "preview_image_path": None,
                "detailed_log": f"錯誤詳情：{str(e)}",
                "tags": ""
            }
    
    def process_batch_directory(self, input_directory: str, output_directory: str,
                               selected_steps: List[str], recursive: bool = True,
                               preserve_structure: bool = True) -> Dict:
        """
        處理批量目錄 - 使用正確的目錄級別步驟順序執行
        
        Args:
            input_directory: 輸入目錄路徑
            output_directory: 輸出目錄路徑  
            selected_steps: 選擇的處理步驟
            recursive: 是否遞歸處理
            preserve_structure: 是否保持目錄結構
            
        Returns:
            批量處理結果
        """
        try:
            if not os.path.isdir(input_directory):
                return {
                    "success": False,
                    "message": f"輸入目錄不存在：{input_directory}",
                    "total_files": 0,
                    "successful_files": 0,
                    "failed_files": 0,
                    "errors": []
                }
            
            # 確保輸出目錄存在
            os.makedirs(output_directory, exist_ok=True)
            
            # 檢查目錄中是否有圖片檔案
            image_files = self._scan_image_files(input_directory, recursive)
            
            if not image_files:
                return {
                    "success": False,
                    "message": f"在目錄中沒有找到圖片檔案：{input_directory}",
                    "total_files": 0,
                    "successful_files": 0,
                    "failed_files": 0,
                    "errors": []
                }
            
            self.logger.info(f"[UIAdapter] 🚀 開始批量處理 - 目錄級別步驟順序執行")
            self.logger.info(f"[UIAdapter] 📁 輸入目錄: {input_directory}")  
            self.logger.info(f"[UIAdapter] 📁 輸出目錄: {output_directory}")
            self.logger.info(f"[UIAdapter] 🔧 選定步驟: {selected_steps}")
            self.logger.info(f"[UIAdapter] 📊 找到 {len(image_files)} 個圖片文件")
            self.logger.info(f"[UIAdapter] 📋 正確邏輯: 每個步驟完成整個目錄處理後再進入下一步")
            
            # 🎯 使用 PipelineOrchestrator 進行目錄級別的批量處理
            result = self.pipeline_orchestrator.process_pipeline(
                input_directory=input_directory,
                selected_steps=selected_steps
            )
            
            if result["success"]:
                # 計算處理統計（基於步驟結果估算）
                step_outputs = result.get("step_outputs", {})
                total_files = len(image_files)
                
                # 估算成功處理的檔案數量
                successful_files = 0
                failed_files = 0
                
                # 檢查各步驟的處理結果
                for step_name, step_result in step_outputs.items():
                    if step_result.get("success"):
                        if step_name == "face_detect":
                            # 人臉偵測步驟可能會過濾檔案
                            training_count = step_result.get("training_count", 0)
                            if training_count > 0:
                                successful_files = training_count
                        elif step_name in ["crop", "upscale"]:
                            # 裁切和放大步驟有具體的成功數量
                            step_success = step_result.get("successful_crops", step_result.get("successful_upscales", 0))
                            if step_success > 0:
                                successful_files = step_success
                
                # 如果沒有找到具體數量，使用總檔案數作為估算
                if successful_files == 0:
                    successful_files = total_files
                
                failed_files = total_files - successful_files
                success_rate = (successful_files / total_files * 100) if total_files > 0 else 0
                
                ui_result = {
                    "success": True,
                    "message": f"批量管道處理完成。{successful_files}/{total_files} 檔案處理成功 ({success_rate:.1f}%)",
                    "total_files": total_files,
                    "successful_files": successful_files,
                    "failed_files": failed_files,
                    "success_rate": success_rate,
                    "errors": [],
                    "pipeline_result": result,  # 包含完整的管道結果
                    "step_outputs": step_outputs,
                    "final_working_dir": result.get("final_working_dir", input_directory)
                }
                
                self.logger.info(f"[UIAdapter] ✅ 批量處理成功完成")
                self.logger.info(f"[UIAdapter] 📊 處理統計: {ui_result['message']}")
                
                # 記錄各步驟的結果
                for step_name, step_result in step_outputs.items():
                    if step_result.get("success"):
                        self.logger.info(f"[UIAdapter] ✅ 步驟 {step_name}: {step_result.get('message', '成功')}")
                    else:
                        self.logger.error(f"[UIAdapter] ❌ 步驟 {step_name}: {step_result.get('message', '失敗')}")
                
                return ui_result
            
            else:
                # 管道處理失敗
                return {
                    "success": False,
                    "message": f"批量管道處理失敗：{result.get('message', '未知錯誤')}",
                    "total_files": len(image_files),
                    "successful_files": 0,
                    "failed_files": len(image_files),
                    "success_rate": 0.0,
                    "errors": [result.get('message', '管道處理失敗')],
                    "pipeline_result": result
                }
            
        except Exception as e:
            self.logger.error(f"[UIAdapter] ❌ 批量處理過程發生異常: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"批量處理發生錯誤：{str(e)}",
                "total_files": 0,
                "successful_files": 0,
                "failed_files": 0,
                "errors": [str(e)]
            }
    
    def get_available_steps(self) -> List[Tuple[str, str]]:
        """
        獲取可用的處理步驟列表
        
        Returns:
            [(display_name, step_key), ...] 格式的列表
        """
        steps = []
        for step_key, step_config in self.orchestrator.step_definitions.items():
            display_name = f"{step_key.title()}: {step_config['description']}"
            steps.append((display_name, step_key))
        return steps
    
    def _scan_image_files(self, directory: str, recursive: bool = True) -> List[str]:
        """掃描目錄中的圖片檔案"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
        image_files = []
        
        if recursive:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        image_files.append(os.path.join(root, file))
        else:
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(file_path)
        
        return sorted(image_files)
    
    def _prepare_preview_image(self, image_path: str) -> str:
        """準備預覽圖片，確保在預覽目錄中"""
        if not image_path or not os.path.exists(image_path):
            return None
        
        # 如果已經在預覽目錄中，直接返回
        if os.path.abspath(os.path.dirname(image_path)) == os.path.abspath(self.temp_preview_dir):
            return image_path
        
        # 複製到預覽目錄
        filename = os.path.basename(image_path)
        preview_path = os.path.join(self.temp_preview_dir, filename)
        
        # 避免檔名衝突
        counter = 1
        while os.path.exists(preview_path):
            name, ext = os.path.splitext(filename)
            preview_path = os.path.join(self.temp_preview_dir, f"{name}_{counter}{ext}")
            counter += 1
        
        shutil.copy2(image_path, preview_path)
        return preview_path
    
    def _format_detailed_log(self, result: Dict) -> str:
        """格式化詳細日誌"""
        log_parts = []
        
        if result.get("pipeline_summary"):
            log_parts.append("=== 處理流程摘要 ===")
            log_parts.append(result["pipeline_summary"])
        
        if result.get("step_results"):
            log_parts.append("\\n=== 步驟詳情 ===")
            for step_key, step_result in result["step_results"].items():
                metadata = step_result.get("metadata", {})
                log_parts.append(f"{step_key}: {metadata.get('message', 'No details')}")
                if metadata.get("status"):
                    log_parts.append(f"  狀態: {metadata['status']}")
        
        return "\\n".join(log_parts)
    
    def _extract_tags_from_result(self, result: Dict) -> str:
        """從結果中提取標籤"""
        step_results = result.get("step_results", {})
        tag_result = step_results.get("tag", {})
        
        if tag_result:
            metadata = tag_result.get("metadata", {})
            return metadata.get("tags", "")
        
        return ""


# UI整合函數
def create_ui_adapter(config=None, logger=None) -> UIAdapter:
    """創建UI適配器實例"""
    return UIAdapter(config, logger)


# 為了向後相容性，提供舊的orchestrator介面模擬
class LegacyOrchestrator:
    """
    向後相容的orchestrator介面
    將舊的PIL Image調用轉換為新的檔案導向調用
    """
    
    def __init__(self, config=None, logger=None):
        self.ui_adapter = UIAdapter(config, logger)
        self.config = config
        self.logger = logger
    
    def process_single_image(self, image_path_or_url: str, output_filename_prefix: str = None) -> Dict:
        """
        模擬舊的process_single_image介面
        """
        try:
            # 根據配置選擇啟用的步驟
            selected_steps = []
            for step_key, step_config in self.ui_adapter.orchestrator.step_definitions.items():
                flag_name = step_config["flag"]
                if getattr(self.config, flag_name, False):
                    selected_steps.append(step_key)
            
            # 調用新的檔案導向處理
            result = self.ui_adapter.process_uploaded_image(
                uploaded_file_path=image_path_or_url,
                selected_steps=selected_steps,
                preview_mode=True
            )
            
            # 轉換為舊格式
            return {
                "success": result["success"],
                "final_image_path": result.get("final_image_path"),
                "preview_image_path": result.get("preview_image_path"),
                "message": result["message"],
                "pipeline_summary": result.get("pipeline_summary", ""),
                "detailed_log": result.get("detailed_log", ""),
                "tags": result.get("tags", "")
            }
            
        except Exception as e:
            return {
                "success": False,
                "final_image_path": None,
                "preview_image_path": None,
                "message": f"Processing failed: {str(e)}",
                "pipeline_summary": f"Error: {str(e)}",
                "detailed_log": f"Exception: {str(e)}",
                "tags": ""
            }