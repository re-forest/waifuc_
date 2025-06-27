from PIL import Image
from imgutils.upscale import upscale_with_cdc
import os
from tqdm import tqdm
import time

def upscale_to_target_size(image, target_width=None, target_height=None, model='HGSR-MHR-anime-aug_X4_320', 
                          tile_size=512, tile_overlap=64, batch_size=1, silent=False, preserve_aspect_ratio=True):
    """
    將圖像放大到指定的目標尺寸
    
    Parameters:
        image (PIL.Image.Image): 輸入圖像。
        target_width (int, optional): 目標寬度。如果為None，則根據高度按比例計算。
        target_height (int, optional): 目標高度。如果為None，則根據寬度按比例計算。
        model (str): 使用的CDC模型名稱。(預設: 'HGSR-MHR-anime-aug_X4_320')
        tile_size (int): 每個分塊的大小。(預設: 512)
        tile_overlap (int): 分塊之間的重疊區域大小。(預設: 64)
        batch_size (int): 批次處理大小。(預設: 1)
        silent (bool): 是否禁止顯示進度訊息。(預設: False)
        preserve_aspect_ratio (bool): 是否保持原始寬高比。(預設: True)
        
    Returns:
        PIL.Image.Image: 放大到目標尺寸的圖像。
    """
    # 先使用CDC模型放大
    upscaled_image = upscale_with_cdc(
        image,
        model=model,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        batch_size=batch_size,
        silent=silent
    )
    
    # 如果沒有指定目標尺寸，直接返回CDC放大的結果
    if target_width is None and target_height is None:
        return upscaled_image
    
    # 獲取當前尺寸
    current_width, current_height = upscaled_image.size
    
    # 計算目標尺寸（處理寬高比）
    if target_width is None:
        # 只指定了目標高度，按比例計算寬度
        scale = target_height / current_height
        target_width = int(current_width * scale)
    elif target_height is None:
        # 只指定了目標寬度，按比例計算高度
        scale = target_width / current_width
        target_height = int(current_height * scale)
    elif preserve_aspect_ratio:
        # 同時指定了寬高，但需要保持比例
        # 計算哪個維度是限制因素（確保寬高都至少達到目標值）
        width_scale = target_width / current_width
        height_scale = target_height / current_height
        
        # 選擇較大的縮放比例，確保兩個維度都達到或超過目標尺寸
        scale = max(width_scale, height_scale)
        target_width = int(current_width * scale)
        target_height = int(current_height * scale)
    
    # 如果目標尺寸小於當前尺寸，使用LANCZOS採樣方式（適合縮小）
    # 如果目標尺寸大於當前尺寸，使用BICUBIC採樣方式（適合放大）
    resample_method = Image.LANCZOS if (target_width < current_width or target_height < current_height) else Image.BICUBIC
    
    # 調整到目標尺寸
    resized_image = upscaled_image.resize((target_width, target_height), resample=resample_method)
    
    return resized_image

def upscale_and_center_crop(image, target_width, target_height, model='HGSR-MHR-anime-aug_X4_320', 
                           tile_size=512, tile_overlap=64, batch_size=1, silent=False):
    """
    將圖像依比例放大到長寬都大於指定目標尺寸，然後以圖片中心進行裁剪
    
    Parameters:
        image (PIL.Image.Image): 輸入圖像
        target_width (int): 目標寬度
        target_height (int): 目標高度
        model (str): 使用的CDC模型名稱 (預設: 'HGSR-MHR-anime-aug_X4_320')
        tile_size (int): 每個分塊的大小 (預設: 512)
        tile_overlap (int): 分塊之間的重疊區域大小 (預設: 64)
        batch_size (int): 批次處理大小 (預設: 1)
        silent (bool): 是否禁止顯示進度訊息 (預設: False)
    
    Returns:
        PIL.Image.Image: 裁剪後的目標尺寸圖像
    """
    # 先將圖像依比例放大，確保長寬都大於目標尺寸
    upscaled_image = upscale_to_target_size(
        image,
        target_width=target_width,
        target_height=target_height,
        model=model,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        batch_size=batch_size,
        silent=silent,
        preserve_aspect_ratio=True  # 必須保持原比例
    )
    
    # 取得放大後的尺寸
    upscaled_width, upscaled_height = upscaled_image.size
    
    # 計算裁剪區域（從中心開始）
    left = (upscaled_width - target_width) // 2
    top = (upscaled_height - target_height) // 2
    right = left + target_width
    bottom = top + target_height
    
    # 進行裁剪
    cropped_image = upscaled_image.crop((left, top, right, bottom))
    
    return cropped_image

def upscale_images_in_directory(directory, target_width=1024, target_height=1024, 
                                model='HGSR-MHR-anime-aug_X4_320', overwrite=True,
                                min_size=None, recursive=True):
    """
    處理目錄中的所有圖像，支援多層目錄結構，放大並裁剪圖像到目標尺寸
    
    Parameters:
        directory (str): 包含圖像的目錄路徑
        target_width (int): 目標寬度 (預設: 1024)
        target_height (int): 目標高度 (預設: 1024)
        model (str): 使用的CDC模型名稱 (預設: 'HGSR-MHR-anime-aug_X4_320')
        overwrite (bool): 是否覆蓋原圖像 (預設: True)
        min_size (int, optional): 最小寬度或高度閾值，小於此值的圖像才會被放大
        recursive (bool): 是否遞迴處理子目錄 (預設: True)
    
    Returns:
        tuple: (處理的圖像數, 放大的圖像數)
    """
    if not os.path.isdir(directory):
        print(f"錯誤: 目錄 '{directory}' 不存在或不是有效目錄")
        return 0, 0
    
    # 支援的圖像格式
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    
    # 收集需要處理的圖像
    image_files = []
    
    if recursive:
        # 遞迴處理所有子目錄
        for root, _, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in supported_formats):
                    image_files.append(os.path.join(root, file))
    else:
        # 只處理當前目錄
        for file in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, file)) and any(file.lower().endswith(ext) for ext in supported_formats):
                image_files.append(os.path.join(directory, file))
    
    if not image_files:
        print(f"在目錄 '{directory}' 中未找到支援的圖像文件")
        return 0, 0
    
    upscaled_count = 0
    total_time = 0
    
    # 處理每個圖像
    for img_path in tqdm(image_files, desc="放大圖像"):
        try:
            # 打開圖像
            with Image.open(img_path) as img:
                # 確保圖像模式為RGB
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    
                width, height = img.size
                
                # 如果設定了最小尺寸閾值，檢查是否需要放大
                if min_size is not None and width >= min_size and height >= min_size:
                    continue
                
                # 記錄開始時間
                start_time = time.time()
                
                # 放大並裁剪圖像
                upscaled_img = upscale_and_center_crop(
                    img, 
                    target_width=target_width, 
                    target_height=target_height,
                    model=model
                )
                
                # 計算處理時間
                elapsed_time = time.time() - start_time
                total_time += elapsed_time
                
                # 決定保存路徑
                if overwrite:
                    # 覆蓋原圖像
                    save_path = img_path
                else:
                    # 在原目錄添加前綴
                    base_dir = os.path.dirname(img_path)
                    base_name = os.path.basename(img_path)
                    save_path = os.path.join(base_dir, f"upscaled_{base_name}")
                
                # 保存圖像
                upscaled_img.save(save_path)
                upscaled_count += 1
                print(f"已放大: {img_path} ({width}x{height} -> {upscaled_img.size[0]}x{upscaled_img.size[1]}) 耗時: {elapsed_time:.2f}秒")
        except Exception as e:
            print(f"處理 {img_path} 時出錯: {str(e)}")
    
    avg_time = total_time / upscaled_count if upscaled_count > 0 else 0
    print(f"共處理 {len(image_files)} 張圖像，放大 {upscaled_count} 張圖像")
    print(f"總耗時: {total_time:.2f}秒，平均每張圖像處理時間: {avg_time:.2f}秒")
    
    return len(image_files), upscaled_count

def upscale_images_with_summary(directory, target_width=1024, target_height=1024, 
                               model='HGSR-MHR-anime-aug_X4_320', recursive=True, min_size=0):
    """
    處理目錄中的圖像放大並返回詳細摘要
    
    Parameters:
    - directory: 包含圖像的目錄路徑
    - target_width: 目標寬度
    - target_height: 目標高度
    - model: 使用的CDC模型名稱
    - recursive: 是否遞迴處理子目錄
    - min_size: 最小寬度或高度閾值，小於此值的圖像才會被放大
    
    Returns:
    - dict: 包含處理結果摘要的字典
    """
    logs = []
    try:
        # 驗證輸入目錄
        if not directory or not os.path.isdir(directory):
            return {
                "success": False,
                "error": f"目錄不存在或不是有效目錄: {directory}",
                "logs": [f"目錄不存在或不是有效目錄: {directory}"],
                "directory": directory,
                "total_files": 0,
                "upscaled_files": 0,
                "skipped_files": 0,
                "failed_files": 0,
                "target_size": f"{target_width}x{target_height}",
                "model": model,
                "processing_time": 0
            }
        
        logs.append(f"開始處理目錄: {directory}")
        logs.append(f"目標尺寸: {target_width}x{target_height}")
        logs.append(f"使用模型: {model}")
        logs.append(f"遞迴處理: {recursive}")
        logs.append(f"最小尺寸閾值: {min_size}")
        
        # 記錄開始時間
        import time
        start_time = time.time()
        
        # 支援的圖像格式
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        
        # 收集需要處理的圖像
        image_files = []
        
        if recursive:
            # 遞迴處理所有子目錄
            for root, _, files in os.walk(directory):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in supported_formats):
                        image_files.append(os.path.join(root, file))
        else:
            # 只處理當前目錄
            for file in os.listdir(directory):
                if os.path.isfile(os.path.join(directory, file)) and any(file.lower().endswith(ext) for ext in supported_formats):
                    image_files.append(os.path.join(directory, file))
        
        if not image_files:
            return {
                "success": True,
                "error": None,
                "logs": logs + [f"在目錄 '{directory}' 中未找到支援的圖像文件"],
                "directory": directory,
                "total_files": 0,
                "upscaled_files": 0,
                "skipped_files": 0,
                "failed_files": 0,
                "target_size": f"{target_width}x{target_height}",
                "model": model,
                "processing_time": 0
            }
        
        logs.append(f"找到 {len(image_files)} 個圖像文件")
        
        upscaled_count = 0
        skipped_count = 0
        failed_count = 0
        
        # 處理每個圖像
        for img_path in tqdm(image_files, desc="放大圖像"):
            try:
                # 打開圖像
                with Image.open(img_path) as img:
                    # 確保圖像模式為RGB
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                        
                    width, height = img.size
                    
                    # 如果設定了最小尺寸閾值，檢查是否需要放大
                    if min_size > 0 and width >= min_size and height >= min_size:
                        skipped_count += 1
                        logs.append(f"跳過 {img_path} (尺寸已足夠: {width}x{height})")
                        continue
                    
                    # 放大並裁剪圖像
                    upscaled_img = upscale_and_center_crop(
                        img, 
                        target_width=target_width, 
                        target_height=target_height,
                        model=model
                    )
                    
                    # 保存圖像（覆蓋原圖像）
                    upscaled_img.save(img_path)
                    upscaled_count += 1
                    
                    logs.append(f"已放大: {os.path.basename(img_path)} ({width}x{height} -> {upscaled_img.size[0]}x{upscaled_img.size[1]})")
                    
            except Exception as e:
                failed_count += 1
                error_msg = f"處理 {img_path} 時出錯: {str(e)}"
                logs.append(error_msg)
        
        # 計算總處理時間
        total_time = time.time() - start_time
        
        logs.append(f"處理完成: 總文件 {len(image_files)}, 已放大 {upscaled_count}, 跳過 {skipped_count}, 失敗 {failed_count}")
        logs.append(f"總耗時: {total_time:.2f}秒")
        
        return {
            "success": True,
            "error": None,
            "logs": logs,
            "directory": directory,
            "total_files": len(image_files),
            "upscaled_files": upscaled_count,
            "skipped_files": skipped_count,
            "failed_files": failed_count,
            "target_size": f"{target_width}x{target_height}",
            "model": model,
            "processing_time": total_time
        }
        
    except Exception as e:
        error_msg = f"圖像放大處理失敗: {str(e)}"
        logs.append(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "logs": logs,
            "directory": directory,
            "total_files": 0,
            "upscaled_files": 0,
            "skipped_files": 0,
            "failed_files": 0,
            "target_size": f"{target_width}x{target_height}",
            "model": model,
            "processing_time": 0
        }

if __name__ == "__main__":
    import argparse
    
    # 建立命令列參數解析器
    parser = argparse.ArgumentParser(description='圖像放大工具')
    parser.add_argument('directory', type=str, help='要處理的圖像目錄')
    parser.add_argument('--width', type=int, default=1024, help='目標寬度 (預設: 1024)')
    parser.add_argument('--height', type=int, default=1024, help='目標高度 (預設: 1024)')
    parser.add_argument('--model', type=str, default='HGSR-MHR-anime-aug_X4_320', help='使用的CDC模型名稱')
    parser.add_argument('--no-overwrite', action='store_false', dest='overwrite', help='不覆蓋原始圖像')
    parser.add_argument('--min-size', type=int, help='最小寬度或高度閾值，小於此值的圖像才會被放大')
    parser.add_argument('--no-recursive', action='store_false', dest='recursive', help='不遞迴處理子目錄')
    
    # 解析命令列參數
    args = parser.parse_args()
    
    # 執行批次處理
    print(f"開始處理目錄: {args.directory}")
    print(f"目標尺寸: {args.width}x{args.height}")
    print(f"使用模型: {args.model}")
    print(f"{'覆蓋' if args.overwrite else '不覆蓋'}原始圖像")
    print(f"{'遞迴' if args.recursive else '不遞迴'}處理子目錄")
    if args.min_size:
        print(f"只處理小於 {args.min_size} 像素的圖像")
    
    # 呼叫處理函數
    total, upscaled = upscale_images_in_directory(
        args.directory,
        target_width=args.width,
        target_height=args.height,
        model=args.model,
        overwrite=args.overwrite,
        min_size=args.min_size,
        recursive=args.recursive
    )
    
    print(f"處理完成! 共處理: {total} 張圖像，其中 {upscaled} 張已放大")
