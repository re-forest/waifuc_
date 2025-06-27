import os
import shutil
from imgutils.metrics import lpips_clustering
from tqdm import tqdm

def batch_generator(lst, batch_size):
    """生成器，對文件路徑列表進行批次處理"""
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

def process_lpips_clustering(file_paths, output_directory="lpips_output", batch_size=100):
    """
    根據LPIPS聚類結果處理圖像文件
    
    Parameters:
    - file_paths: 圖像文件路徑列表
    - output_directory: 輸出目錄
    - batch_size: 批處理大小
    
    Returns:
    - output_directory: 聚類結果的輸出目錄
    """
    os.makedirs(output_directory, exist_ok=True)  # 建立輸出資料夾，如果不存在的話
    lpips_counts = {}

    # 添加進度條以跟踪處理過程
    total_batches = (len(file_paths) + batch_size - 1) // batch_size
    with tqdm(total=len(file_paths), desc="LPIPS聚類進度") as pbar:
        for batch in batch_generator(file_paths, batch_size):
            lpips_batch = lpips_clustering(batch)
            
            # 使用 zip 將批次的檔案路徑和 lpips 結果結合起來
            for source_file, lpips_result in zip(batch, lpips_batch):
                # 確保 lpips_result 不是 -1（噪音樣本）
                if lpips_result != -1:
                    # 建立存放該群組檔案的資料夾
                    group_directory = os.path.join(output_directory, f"group_{lpips_result}")
                    os.makedirs(group_directory, exist_ok=True)

                    # 使用 get 方法獲取 lpips_result 對應的計數，如果不存在則預設為 0
                    lpips_count = lpips_counts.get(lpips_result, 0)

                    # 如果 lpips_result 已經出現過1次或更多，移動檔案
                    if lpips_count > 0:
                        destination_file = os.path.join(group_directory, os.path.basename(source_file))
                        shutil.move(source_file, destination_file)

                    # 更新 lpips_counts 字典中 lpips_result 的計數
                    lpips_counts[lpips_result] = lpips_count + 1
                
                # 更新進度條
                pbar.update(1)

    print("檔案已根據 LPIPS 結果分類並放入各自的資料夾中。")
    return output_directory

def process_lpips_clustering_with_summary(file_paths, output_directory="lpips_output", batch_size=100):
    """
    根據LPIPS聚類結果處理圖像文件並返回詳細摘要
    
    Parameters:
    - file_paths: 圖像文件路徑列表
    - output_directory: 輸出目錄
    - batch_size: 批處理大小
    
    Returns:
    - dict: 包含處理結果摘要的字典
    """
    logs = []
    try:
        # 驗證輸入
        if not file_paths:
            return {
                "success": False,
                "error": "沒有提供文件路徑",
                "logs": ["沒有提供文件路徑"],
                "total_files": 0,
                "processed_files": 0,
                "groups_created": 0,
                "output_directory": None
            }
        
        # 驗證文件是否存在
        valid_files = []
        for file_path in file_paths:
            if os.path.isfile(file_path):
                valid_files.append(file_path)
            else:
                logs.append(f"警告: 文件不存在 - {file_path}")
        
        if not valid_files:
            return {
                "success": False,
                "error": "沒有找到有效的圖像文件",
                "logs": logs,
                "total_files": len(file_paths),
                "processed_files": 0,
                "groups_created": 0,
                "output_directory": None
            }
        
        logs.append(f"開始處理 {len(valid_files)} 個文件，批次大小: {batch_size}")
        logs.append(f"輸出目錄: {output_directory}")
        
        # 執行LPIPS聚類處理
        result_dir = process_lpips_clustering(valid_files, output_directory, batch_size)
        
        # 統計結果
        groups_created = 0
        processed_files = 0
        
        if os.path.exists(result_dir):
            # 計算創建的群組數
            for item in os.listdir(result_dir):
                item_path = os.path.join(result_dir, item)
                if os.path.isdir(item_path) and item.startswith("group_"):
                    groups_created += 1
                    # 計算該群組中的文件數
                    group_files = [f for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f))]
                    processed_files += len(group_files)
                    logs.append(f"群組 {item} 包含 {len(group_files)} 個文件")
        
        logs.append(f"處理完成，共創建 {groups_created} 個群組，移動 {processed_files} 個文件")
        
        return {
            "success": True,
            "error": None,
            "logs": logs,
            "total_files": len(file_paths),
            "valid_files": len(valid_files),
            "processed_files": processed_files,
            "groups_created": groups_created,
            "output_directory": result_dir
        }
        
    except Exception as e:
        error_msg = f"LPIPS聚類處理失敗: {str(e)}"
        logs.append(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "logs": logs,
            "total_files": len(file_paths) if file_paths else 0,
            "processed_files": 0,
            "groups_created": 0,
            "output_directory": None
        }

if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    
    # 先讀取.env檔案的設定
    load_dotenv()
    
    # 從環境變數獲取預設值
    default_directory = os.getenv("directory", "")
    default_output = os.getenv("lpips_output_directory", "lpips_output")
    default_batch_size = int(os.getenv("lpips_batch_size", 100))
    
    parser = argparse.ArgumentParser(description='使用LPIPS進行圖像聚類')
    parser.add_argument('directory', type=str, nargs='?', default=default_directory, 
                       help='圖像文件目錄 (預設: 從.env讀取)')
    parser.add_argument('--output', type=str, default=default_output, 
                       help=f'輸出目錄 (預設: {default_output})')
    parser.add_argument('--batch-size', type=int, default=default_batch_size, 
                       help=f'處理批次大小 (預設: {default_batch_size})')
    
    args = parser.parse_args()
    
    # 確認目錄存在
    if not args.directory or not os.path.isdir(args.directory):
        print(f"錯誤: 指定的目錄 '{args.directory}' 不存在或不是有效目錄")
        exit(1)
    
    # 取得資料夾內所有檔案路徑
    file_paths = [os.path.join(args.directory, f) for f in os.listdir(args.directory) 
                 if os.path.isfile(os.path.join(args.directory, f))]
    
    if not file_paths:
        print(f"警告: 在目錄 '{args.directory}' 中未找到檔案")
        exit(0)
    
    print(f"找到 {len(file_paths)} 個檔案，開始進行LPIPS聚類...")
    print(f"輸出目錄: {args.output}, 批次大小: {args.batch_size}")
    
    # 執行LPIPS聚類處理
    output_dir = process_lpips_clustering(file_paths, args.output, args.batch_size)
    
    print(f"處理完成，結果已儲存至: {output_dir}")
