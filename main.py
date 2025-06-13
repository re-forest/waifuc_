import os
import sys
import shutil
from concurrent.futures import ProcessPoolExecutor
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from tqdm import tqdm

# 導入日誌系統
from logger_config import get_logger

# 從本地模組導入功能
from validate_image import validate_and_remove_invalid_images
from face_detection import detect_faces_in_directory
from lpips_clustering import process_lpips_clustering
from crop import classify_files_in_directory, process_single_folder
from tag import tag_image
from upscale import upscale_images_in_directory

def main():
    """主程式執行流程"""
    # 初始化日誌系統
    logger = get_logger('main')
    
    try:
        # 從.env檔案中讀取設定
        load_dotenv()
        logger.info("成功載入環境變數設定")
        
        # 檔案路徑
        directory = os.getenv("directory")
        logger.debug(f"從環境變數讀取的目錄: {directory}")
        
        if not directory or not os.path.isdir(directory):
            error_msg = f"目錄 '{directory}' 不存在或不是有效目錄"
            logger.error(error_msg)
            print(f"錯誤: {error_msg}")
            return 1

        logger.info(f"開始處理目錄: {directory}")
        print(f"開始處理目錄: {directory}")
        
        # 驗證圖片完整性
        # 目的: 移除損壞或不完整的圖像，確保後續處理不會因為圖像問題而中斷
        if os.getenv("enable_validation", "true").lower() == "true":
            logger.info("開始執行圖片驗證步驟")
            print("------- 開始驗證圖片完整性 -------")
            try:
                validate_and_remove_invalid_images(directory)
                logger.info("圖片驗證步驟完成")
            except Exception as e:
                logger.error(f"圖片驗證過程中發生錯誤: {str(e)}", exc_info=True)
                print(f"警告: 圖片驗證過程中發生錯誤: {str(e)}")
        else:
            logger.info("圖片驗證步驟已跳過（根據環境變數設定）")
        
        # 人臉檢測
        # 目的: 篩選出單一人臉的圖像，這對於訓練生成特定人物的AI模型非常重要
        # 單一人臉的圖像能讓模型更專注於學習特定人物特徵，避免多人圖像造成的特徵混淆
        if os.getenv("enable_face_detection", "true").lower() == "true":
            logger.info("開始執行人臉檢測步驟")
            print("------- 開始人臉檢測 -------")
            try:
                min_face_count = int(os.getenv("min_face_count", 1))  # 預設最小人臉數量為 1
                face_output_directory = os.getenv("face_output_directory", "face_out")  # 從環境變數讀取輸出目錄
                logger.debug(f"人臉檢測參數 - 最小人臉數量: {min_face_count}, 輸出目錄: {face_output_directory}")
                detect_faces_in_directory(directory, min_face_count, face_output_directory)
                logger.info("人臉檢測步驟完成")
            except Exception as e:
                logger.error(f"人臉檢測過程中發生錯誤: {str(e)}", exc_info=True)
                print(f"警告: 人臉檢測過程中發生錯誤: {str(e)}")
        else:
            logger.info("人臉檢測步驟已跳過（根據環境變數設定）")
          # 重新取得資料夾內所有檔案路徑 
        file_paths = [os.path.join(directory, f) for f in os.listdir(directory) 
                     if os.path.isfile(os.path.join(directory, f))]
        logger.debug(f"在目錄中找到 {len(file_paths)} 個檔案")
        
        # 圖片去重複
        # 目的: 通過LPIPS聚類找出相似的圖像，避免訓練資料中存在過多相似圖片
        # 相似度過高的圖像不僅浪費訓練資源，還可能導致模型過度擬合某些特定場景
        if os.getenv("enable_lpips_clustering", "true").lower() == "true":
            logger.info("開始執行 LPIPS 圖片去重複步驟")
            print("------- 開始圖片去重複 -------")
            try:
                lpips_output_directory = os.getenv("lpips_output_directory", "lpips_output")
                lpips_batch_size = int(os.getenv("lpips_batch_size", 100))
                logger.debug(f"LPIPS 參數 - 輸出目錄: {lpips_output_directory}, 批次大小: {lpips_batch_size}")
                
                print(f"圖片去重複設定 - 輸出目錄: {lpips_output_directory}, 批次大小: {lpips_batch_size}")
                lpips_result_dir = process_lpips_clustering(file_paths, lpips_output_directory, lpips_batch_size)
                logger.info(f"LPIPS 圖片去重複完成，結果保存在: {lpips_result_dir}")
                print(f"圖片去重複處理完成，結果保存在: {lpips_result_dir}")
            except Exception as e:
                logger.error(f"LPIPS 去重複過程中發生錯誤: {str(e)}", exc_info=True)
                print(f"警告: LPIPS 去重複過程中發生錯誤: {str(e)}")
        else:
            logger.info("LPIPS 圖片去重複步驟已跳過（根據環境變數設定）")
        
        # 裁切檔案
        # 目的: 自動將人物圖像切割成全身像、半身像和頭像，使模型可以從不同尺度學習人物特徵
        # 多尺度學習可以顯著提升模型對人物特徵的理解和生成能力
        if os.getenv("enable_cropping", "true").lower() == "true":
            logger.info("開始執行圖片裁切步驟")
            print("------- 開始裁切圖片 -------")
            try:
                output_directory = os.getenv("output_directory")
                if not output_directory:
                    output_directory = os.path.join(os.path.dirname(directory), "cropped")
                    logger.warning(f"未設定 output_directory 環境變數，使用預設值: {output_directory}")
                    print("警告: 未設定 output_directory 環境變數，使用預設值")
                
                logger.debug(f"圖片裁切參數 - 輸出目錄: {output_directory}")
                process_single_folder(directory, output_directory)
                logger.info("圖片裁切步驟完成")
            except Exception as e:
                logger.error(f"圖片裁切過程中發生錯誤: {str(e)}", exc_info=True)
                print(f"警告: 圖片裁切過程中發生錯誤: {str(e)}")
            
            process_single_folder(directory, output_directory)
        
        # 檔案分類
        # 目的: 將裁切後的圖像分類到不同資料夾，便於後續處理和訓練
        # 分類後的資料結構更清晰，也便於針對不同類型圖像應用不同處理策略        # 檔案分類
        # 目的: 將裁切後的圖像分類到不同資料夾，便於後續處理和訓練
        # 分類後的資料結構更清晰，也便於針對不同類型圖像應用不同處理策略
        if os.getenv("enable_classification", "true").lower() == "true":
            logger.info("開始執行圖片分類步驟")
            print("------- 開始分類圖片 -------")
            try:
                output_directory = os.getenv("output_directory")
                if not output_directory:
                    logger.warning("未設定 output_directory 環境變數，跳過分類步驟")
                    print("警告: 未設定 output_directory 環境變數，跳過分類步驟")
                else:
                    logger.debug(f"圖片分類參數 - 目錄: {output_directory}")
                    classify_files_in_directory(output_directory)
                    logger.info("圖片分類步驟完成")
            except Exception as e:
                logger.error(f"圖片分類過程中發生錯誤: {str(e)}", exc_info=True)
                print(f"警告: 圖片分類過程中發生錯誤: {str(e)}")
        else:
            logger.info("圖片分類步驟已跳過（根據環境變數設定）")

        # 放大圖片
        # 目的: 對低解析度圖像進行超解析度處理，提高整體資料集的品質
        # 高品質的訓練資料能夠讓模型學習到更細緻的特徵，提升生成結果的品質
        if os.getenv("enable_upscaling", "true").lower() == "true":
            logger.info("開始執行圖片放大步驟")
            print("------- 開始放大圖片 -------")
            try:
                output_directory = os.getenv("output_directory")
                if not output_directory:
                    logger.warning("未設定 output_directory 環境變數，跳過放大步驟")
                    print("警告: 未設定 output_directory 環境變數，跳過放大步驟")
                else:
                    target_width = int(os.getenv("upscale_target_width", 1024))
                    target_height = int(os.getenv("upscale_target_height", 1024))
                    upscale_model = os.getenv("upscale_model", "HGSR-MHR-anime-aug_X4_320")
                    min_size = int(os.getenv("upscale_min_size", 800)) if os.getenv("upscale_min_size") else None
                    
                    logger.debug(f"放大參數 - 目標尺寸: {target_width}x{target_height}, 模型: {upscale_model}, 最小尺寸: {min_size}")
                    print(f"放大設定 - 目標尺寸: {target_width}x{target_height}, 模型: {upscale_model}")
                    if min_size:
                        print(f"只處理小於 {min_size}x{min_size} 的圖片")
                        
                    upscale_images_in_directory(
                        output_directory,
                        target_width=target_width,
                        target_height=target_height,
                        model=upscale_model,
                        overwrite=True,
                        min_size=min_size,
                        recursive=True
                    )
                    logger.info("圖片放大步驟完成")
            except Exception as e:
                logger.error(f"圖片放大過程中發生錯誤: {str(e)}", exc_info=True)
                print(f"警告: 圖片放大過程中發生錯誤: {str(e)}")
        else:
            logger.info("圖片放大步驟已跳過（根據環境變數設定）")
        
        # 標記圖片
        # 目的: 自動為每張圖像生成描述性標籤，用於條件式生成模型的訓練
        # 這些標籤可以幫助模型學習圖像內容與文字描述之間的關聯，提升控制生成結果的能力
        if os.getenv("enable_tagging", "true").lower() == "true":
            logger.info("開始執行圖片標記步驟")
            print("------- 開始標記圖片 -------")
            try:
                output_directory = os.getenv("output_directory")
                if not output_directory:
                    logger.warning("未設定 output_directory 環境變數，跳過標記步驟")
                    print("警告: 未設定 output_directory 環境變數，跳過標記步驟")
                else:
                    logger.debug(f"圖片標記參數 - 目錄: {output_directory}")
                    tag_image(output_directory)
                    logger.info("圖片標記步驟完成")
            except Exception as e:
                logger.error(f"圖片標記過程中發生錯誤: {str(e)}", exc_info=True)
                print(f"警告: 圖片標記過程中發生錯誤: {str(e)}")
        else:
            logger.info("圖片標記步驟已跳過（根據環境變數設定）")
            logger.info("所有處理步驟已完成")
        print("所有處理已完成!")
        return 0
        
    except Exception as e:
        logger.critical(f"主程式執行過程中發生嚴重錯誤: {str(e)}", exc_info=True)
        print(f"嚴重錯誤: {str(e)}")
        return 1
    finally:
        logger.info("程式結束")

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger = get_logger('main')
        logger.warning("程式被使用者中斷")
        print("\n程式被使用者中斷")
        sys.exit(1)
    except Exception as e:
        logger = get_logger('main')
        logger.critical(f"程式執行失敗: {str(e)}", exc_info=True)
        print(f"\n執行過程中發生錯誤: {str(e)}")
        sys.exit(1)
