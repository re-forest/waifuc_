#!/usr/bin/env python3
"""
批量圖片處理命令列工具

這個工具可以批量處理資料夾中的所有圖片，支援遞歸掃描子資料夾。

使用方式:
    python batch_processor.py input_dir output_dir [options]

範例:
    # 基本批量處理
    python batch_processor.py /path/to/input /path/to/output
    
    # 遞歸處理所有子資料夾，保持目錄結構
    python batch_processor.py /path/to/input /path/to/output --recursive --preserve-structure
    
    # 只啟用特定處理步驟
    python batch_processor.py /path/to/input /path/to/output --steps validate upscale tag
"""

import argparse
import sys
import os
from pathlib import Path

# 添加項目根目錄到 Python 路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.orchestrator import Orchestrator
from config import settings
from utils.logger_config import setup_logging


def parse_arguments():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(
        description="批量圖片處理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "input_dir",
        help="輸入目錄路徑"
    )
    
    parser.add_argument(
        "output_dir", 
        help="輸出目錄路徑"
    )
    
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        default=True,
        help="遞歸處理子目錄 (預設: True)"
    )
    
    parser.add_argument(
        "--no-recursive",
        action="store_false",
        dest="recursive",
        help="不遞歸處理子目錄"
    )
    
    parser.add_argument(
        "--preserve-structure", "-p",
        action="store_true", 
        default=True,
        help="保持原有目錄結構 (預設: True)"
    )
    
    parser.add_argument(
        "--no-preserve-structure",
        action="store_false",
        dest="preserve_structure",
        help="不保持目錄結構，所有文件輸出到同一目錄"
    )
    
    parser.add_argument(
        "--steps", "-s",
        nargs="+",
        default=None,
        choices=["validate", "face_detect", "upscale", "crop", "tag", "cluster"],
        help="要執行的處理步驟 (預設: 使用配置文件中啟用的步驟)"
    )
    
    parser.add_argument(
        "--log-level", "-l",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="日誌級別 (預設: INFO)"
    )
    
    parser.add_argument(
        "--config-file", "-c",
        help="配置文件路徑 (預設使用內建配置)"
    )
    
    return parser.parse_args()


def setup_batch_config(args):
    """基於命令列參數設置批量處理配置"""
    # 如果指定了步驟，則只啟用指定的步驟
    if args.steps:
        # 先關閉所有步驟
        settings.ENABLE_VALIDATION = False
        settings.ENABLE_FACE_DETECTION = False
        settings.ENABLE_UPSCALE = False
        settings.ENABLE_CROP = False
        settings.ENABLE_TAGGING = False
        settings.ENABLE_LPIPS_CLUSTERING = False
        
        # 啟用指定的步驟
        step_mapping = {
            "validate": "ENABLE_VALIDATION",
            "face_detect": "ENABLE_FACE_DETECTION", 
            "upscale": "ENABLE_UPSCALE",
            "crop": "ENABLE_CROP",
            "tag": "ENABLE_TAGGING",
            "cluster": "ENABLE_LPIPS_CLUSTERING"
        }
        
        for step in args.steps:
            if step in step_mapping:
                setattr(settings, step_mapping[step], True)
    
    return settings


def main():
    """主函數"""
    args = parse_arguments()
    
    # 設置日誌
    logger = setup_logging(
        "batch_processor", 
        settings.LOG_DIR, 
        args.log_level
    )
    
    logger.info("=== 批量圖片處理工具啟動 ===")
    logger.info(f"輸入目錄: {args.input_dir}")
    logger.info(f"輸出目錄: {args.output_dir}")
    logger.info(f"遞歸處理: {args.recursive}")
    logger.info(f"保持結構: {args.preserve_structure}")
    logger.info(f"處理步驟: {args.steps or '使用配置預設'}")
    
    # 驗證輸入目錄
    if not os.path.exists(args.input_dir):
        logger.error(f"輸入目錄不存在: {args.input_dir}")
        sys.exit(1)
    
    if not os.path.isdir(args.input_dir):
        logger.error(f"輸入路徑不是目錄: {args.input_dir}")
        sys.exit(1)
    
    # 創建輸出目錄
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"輸出目錄已準備: {args.output_dir}")
    
    try:
        # 設置配置
        config = setup_batch_config(args)
        
        # 創建 Orchestrator 實例
        orchestrator = Orchestrator(config=config, logger=logger)
        
        # 執行批量處理
        logger.info("開始批量處理...")
        result = orchestrator.process_batch(
            input_directory=args.input_dir,
            output_directory=args.output_dir,
            recursive=args.recursive,
            preserve_structure=args.preserve_structure,
            selected_steps=args.steps
        )
        
        # 輸出結果
        logger.info("=== 批量處理完成 ===")
        logger.info(f"處理結果: {result['message']}")
        logger.info(f"總文件數: {result['total_files']}")
        logger.info(f"成功處理: {result['successful_files']}")
        logger.info(f"失敗數量: {result['failed_files']}")
        logger.info(f"成功率: {result.get('success_rate', 0):.1f}%")
        
        if result.get('errors'):
            logger.warning(f"發生 {len(result['errors'])} 個錯誤:")
            for i, error in enumerate(result['errors'][:10], 1):  # 只顯示前10個錯誤
                logger.warning(f"  {i}. {error}")
            if len(result['errors']) > 10:
                logger.warning(f"  ... 還有 {len(result['errors']) - 10} 個錯誤，請查看詳細日誌")
        
        # 根據結果設置退出碼
        if result['success']:
            logger.info("批量處理成功完成")
            sys.exit(0)
        else:
            logger.error("批量處理失敗")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("用戶中斷處理")
        sys.exit(130)
    except Exception as e:
        logger.error(f"批量處理過程中發生錯誤: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()