#!/usr/bin/env python3
"""
WaifuC 增強版批處理器
集成所有AI圖片處理功能
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 添加項目路徑
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def setup_argument_parser():
    """設置命令行參數解析器"""
    parser = argparse.ArgumentParser(
        description='WaifuC 增強版AI圖片批處理工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  # 基本批處理 (所有功能)
  python batch_processor_enhanced.py input_images/ output_images/

  # 只進行人臉偵測分類
  python batch_processor_enhanced.py input_images/ output_images/ --only-face-detection

  # 只進行重複圖片淘汰
  python batch_processor_enhanced.py input_images/ output_images/ --only-clustering

  # 只進行圖片裁切
  python batch_processor_enhanced.py input_images/ output_images/ --only-crop

  # 只進行圖片標記
  python batch_processor_enhanced.py input_images/ output_images/ --only-tagging

  # 只進行圖片放大
  python batch_processor_enhanced.py input_images/ output_images/ --only-upscale

  # 自定義處理流程
  python batch_processor_enhanced.py input_images/ output_images/ \
    --steps validate,face_detect,tag,upscale

新增功能說明:
  📋 圖片驗證: 檢查圖片完整性，自動隔離損壞文件
  👥 人臉分類: 按人臉數量自動分類 (無臉/單臉/雙臉/多臉)
  🔄 智能去重: 使用LPIPS算法淘汰重複圖片，保留最佳品質
  ✂️  智能裁切: 自動檢測並裁切頭部/上半身/全身，分類存儲
  🏷️  自動標記: 使用WD14模型生成精確標籤，支持批量保存
  🔍 智能放大: AI放大到指定尺寸，適合模型訓練使用
        """)
    
    parser.add_argument('input_dir', help='輸入圖片目錄')
    parser.add_argument('output_dir', help='輸出目錄')
    
    # 處理模式選項
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--only-validation', action='store_true',
                           help='只執行圖片驗證')
    mode_group.add_argument('--only-face-detection', action='store_true',
                           help='只執行人臉偵測分類')
    mode_group.add_argument('--only-clustering', action='store_true',
                           help='只執行LPIPS聚類去重')
    mode_group.add_argument('--only-crop', action='store_true',
                           help='只執行圖片裁切')
    mode_group.add_argument('--only-tagging', action='store_true',
                           help='只執行圖片標記')
    mode_group.add_argument('--only-upscale', action='store_true',
                           help='只執行圖片放大')
    
    # 自定義步驟
    parser.add_argument('--steps', type=str,
                       help='自定義處理步驟 (用逗號分隔): validate,face_detect,cluster,crop,tag,upscale')
    
    # 配置選項
    parser.add_argument('--face-threshold', type=float, default=0.3,
                       help='人臉偵測閾值 (默認: 0.3)')
    parser.add_argument('--target-face-count', type=int, default=1,
                       help='訓練需要的人臉數量 (默認: 1, -1=不限制)')
    parser.add_argument('--face-filter-mode', choices=['keep_target', 'exclude_target', 'classify_all'],
                       default='keep_target', help='人臉過濾模式 (默認: keep_target)')
    parser.add_argument('--clustering-threshold', type=float, default=0.3,
                       help='聚類相似度閾值 (默認: 0.3)')
    parser.add_argument('--upscale-size', type=str, default='2048x2048',
                       help='放大目標尺寸 (默認: 2048x2048)')
    parser.add_argument('--tag-threshold', type=float, default=0.35,
                       help='標籤生成閾值 (默認: 0.35)')
    
    # 輸出控制
    parser.add_argument('--preserve-structure', action='store_true',
                       help='保持原始目錄結構')
    parser.add_argument('--recursive', action='store_true', default=True,
                       help='遞歸處理子目錄 (默認開啟)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='日誌級別')
    
    return parser

def main():
    """主函數"""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # 檢查輸入目錄
    if not os.path.isdir(args.input_dir):
        print(f"❌ 錯誤: 輸入目錄不存在: {args.input_dir}")
        return 1
    
    # 創建輸出目錄
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🚀 WaifuC 增強版批處理器啟動")
    print("=" * 60)
    print(f"📁 輸入目錄: {args.input_dir}")
    print(f"📁 輸出目錄: {args.output_dir}")
    
    # 確定處理步驟
    if args.steps:
        selected_steps = [s.strip() for s in args.steps.split(',')]
    elif args.only_validation:
        selected_steps = ['validate']
    elif args.only_face_detection:
        selected_steps = ['face_detect']
    elif args.only_clustering:
        selected_steps = ['cluster']
    elif args.only_crop:
        selected_steps = ['crop']
    elif args.only_tagging:
        selected_steps = ['tag']
    elif args.only_upscale:
        selected_steps = ['upscale']
    else:
        selected_steps = ['validate', 'face_detect', 'cluster', 'crop', 'tag', 'upscale']
    
    print(f"🔧 處理步驟: {', '.join(selected_steps)}")
    
    # 顯示人臉偵測配置（如果包含此步驟）
    if 'face_detect' in selected_steps:
        print(f"👥 目標人臉數量: {args.target_face_count}")
        print(f"👥 過濾模式: {args.face_filter_mode}")
    
    print("=" * 60)
    
    try:
        # 導入必要模塊 (如果可用)
        print("📦 正在載入處理模塊...")
        
# 嘗試導入智能編排器
        try:
            from core.pipeline_orchestrator import PipelineOrchestrator
            from config import settings
            from utils.logger_config import setup_logging
            
            print("✅ 所有模塊載入成功")
            print("\n🎯 開始智能管道處理...")
            
            # 設置日誌
            logger = setup_logging("pipeline", "logs/", args.log_level)
            
            # 應用命令行參數到配置
            if hasattr(settings, 'FACE_DETECTION_TARGET_FACE_COUNT'):
                settings.FACE_DETECTION_TARGET_FACE_COUNT = args.target_face_count
            if hasattr(settings, 'FACE_DETECTION_FILTER_MODE'):
                settings.FACE_DETECTION_FILTER_MODE = args.face_filter_mode
            if hasattr(settings, 'FACE_DETECTION_CONFIDENCE_THRESHOLD'):
                settings.FACE_DETECTION_CONFIDENCE_THRESHOLD = args.face_threshold
            
            # 創建智能編排器
            orchestrator = PipelineOrchestrator(config=settings, logger=logger)
            
            # 執行處理管道
            result = orchestrator.process_pipeline(args.input_dir, selected_steps)
            
            if result["success"]:
                print("\n✨ 處理完成！")
                print(f"\n📊 {result['message']}")
                
                # 顯示詳細結果
                step_outputs = result.get("step_outputs", {})
                for step_name, step_result in step_outputs.items():
                    if step_result.get("success"):
                        print(f"  ✅ {step_name}: {step_result.get('message', '')}")
                    else:
                        print(f"  ❌ {step_name}: {step_result.get('message', '')}")
                
                # 特別顯示人臉偵測結果
                if 'face_detect' in step_outputs:
                    face_result = step_outputs['face_detect']
                    if face_result.get("filter_applied"):
                        print(f"\n👥 人臉過濾結果:")
                        print(f"  • 訓練可用圖片: {face_result.get('training_count', 0)} 個")
                        print(f"  • 排除圖片: {face_result.get('excluded_count', 0)} 個")
                        print(f"  • 訓練目錄: {face_result.get('training_directory', 'N/A')}")
                
                # 顯示裁切結果
                if 'crop' in step_outputs:
                    crop_result = step_outputs['crop']
                    if crop_result.get("success"):
                        print(f"\n✂️ 裁切結果:")
                        print(f"  • 成功裁切: {crop_result.get('successful_crops', 0)} 個")
                        print(f"  • 裁切輸出: {crop_result.get('output_directory', 'N/A')}")
                
                print("\n🎉 智能管道處理完成！")
                return 0
            else:
                print(f"\n❌ 處理失敗: {result['message']}")
                return 1
                
        except ImportError:
            print("⚠️ 無法載入完整功能模塊，顯示功能預覽...")
            
            # 顯示會執行的處理流程
            step_descriptions = {
                'validate': '📋 圖片驗證 - 檢查完整性並隔離無效圖片',
                'face_detect': f'👥 人臉偵測 - 保留{args.target_face_count}個人臉的圖片',
                'cluster': '🔄 智能去重 - 使用LPIPS淘汰重複圖片',
                'crop': '✂️ 智能裁切 - 從過濾後的圖片進行裁切分類',
                'tag': '🏷️ 自動標記 - 生成精確標籤並保存',
                'upscale': '🔍 智能放大 - AI放大到訓練尺寸'
            }
            
            print("\n📋 將執行的處理流程:")
            for step in selected_steps:
                if step in step_descriptions:
                    print(f"  {step_descriptions[step]}")
            
            if len(selected_steps) > 1:
                print("\n🔗 批量處理管道 (目錄級別步驟順序執行):")
                print("  ✅ 正確邏輯: 每個步驟完成整個目錄處理後，再進入下一個步驟")
                print("  ❌ 錯誤邏輯: 每張圖片走完整個管道")
                print("\n📋 執行順序:")
                
                current_dir = args.input_dir
                for i, step in enumerate(selected_steps, 1):
                    step_name = {
                        'validate': '圖片驗證',
                        'face_detect': '人臉偵測與過濾', 
                        'cluster': 'LPIPS聚類去重',
                        'crop': '圖片裁切分類',
                        'tag': '自動標籤生成',
                        'upscale': 'AI圖片放大'
                    }.get(step, step)
                    
                    print(f"  {i}. {step_name}: 處理整個 {current_dir} 目錄")
                    
                    # 預測下一個目錄
                    if step == 'face_detect' and args.face_filter_mode in ['keep_target', 'exclude_target']:
                        current_dir = f"{args.input_dir}/training_faces"
                    elif step == 'crop':
                        current_dir = f"{current_dir.rstrip('/')}_parent/cropped_images"
                    elif step == 'upscale':
                        current_dir = f"{current_dir.rstrip('/')}_parent/upscaled_images"
            
            print("\n✅ 功能預覽完成！請安裝依賴以使用完整功能。")
            return 0
        
    except ImportError as e:
        print(f"❌ 模塊導入失敗: {e}")
        print("💡 請確保已安裝所有依賴項目:")
        print("   pip install pillow waifuc[gpu] dghs-imgutils gradio==4.15.0 onnxruntime-gpu==1.21.0")
        return 1
    except Exception as e:
        print(f"❌ 處理過程中發生錯誤: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
