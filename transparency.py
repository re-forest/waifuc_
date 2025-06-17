#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
from PIL import Image
import sys
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

def has_transparency(img_path):
    """
    檢查圖片是否有透明圖層
    """
    try:
        # 打開圖片
        img = Image.open(img_path)
        
        # 檢查圖片模式，如果包含'A'代表有alpha通道
        if img.mode == 'RGBA' or img.mode == 'LA' or 'A' in img.mode:
            # 進一步檢查是否有非完全不透明的像素
            if 'A' in img.getbands():
                # 獲取alpha通道
                alpha = img.getchannel('A')
                # 檢查是否有小於255的值 (非完全不透明)
                return min(alpha.getdata()) < 255
        
        # 檢查PNG特殊透明像素
        if img.format == 'PNG' and img.info.get('transparency', None) is not None:
            return True
        
        # 檢查GIF特殊透明色
        if img.format == 'GIF' and img.info.get('transparency', None) is not None:
            return True
        
        return False
    except Exception as e:
        print(f"處理 {img_path} 時出錯: {str(e)}")
        return False

def convert_transparent_to_white(img_path):
    """
    將透明背景轉換為白色背景
    """
    try:
        # 打開原始圖片
        img = Image.open(img_path)
        original_format = img.format  # 保留原始格式
        
        # 如果圖片有透明度通道
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            # 創建一個白色背景圖片
            background = Image.new('RGBA', img.size, (255, 255, 255, 255))
            # 將原圖片貼到白色背景上
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, (0, 0), img)
            # 轉換為RGB模式 (沒有透明度)
            background = background.convert('RGB')
            # 保存圖片
            background.save(img_path, original_format)
            print(f"已將 {img_path} 轉換為白色背景")
            return True
        return False
    except Exception as e:
        print(f"轉換 {img_path} 時出錯: {str(e)}")
        return False

def scan_directory(directory, max_workers=8):
    """
    多線程掃描目錄中的所有圖片文件，檢查是否有透明圖層
    """
    results = []
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.tiff')
    file_list = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(image_extensions):
                file_path = os.path.join(root, file)
                file_list.append(file_path)

    def check_transparency(file_path):
        transparent = has_transparency(file_path)
        print(f"{'✓' if transparent else '✗'} {file_path}")
        return {
            'file_path': file_path,
            'has_transparency': transparent
        }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(check_transparency, fp) for fp in file_list]
        for future in as_completed(futures):
            results.append(future.result())
    return results

def batch_convert_transparent_to_white(results, max_workers=8):
    """
    多線程將透明背景轉換為白色背景
    """
    converted_count = 0
    def convert_and_update(result):
        if result['has_transparency']:
            if convert_transparent_to_white(result['file_path']):
                result['has_transparency'] = False
                result['converted_to_white'] = True
                return 1
        result['converted_to_white'] = False
        return 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(convert_and_update, r) for r in results]
        for future in as_completed(futures):
            converted_count += future.result()
    return converted_count

def main():
    load_dotenv()
    root_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"正在掃描目錄: {root_dir}")
    # 從環境變數讀取線程數，預設8
    max_workers = int(os.getenv("transparency_max_workers", 8))
    # 多線程掃描
    results = scan_directory(root_dir, max_workers=max_workers)
    transparent_count = sum(1 for r in results if r['has_transparency'])
    total_count = len(results)
    print(f"\n掃描完成!")
    print(f"圖片總數: {total_count}")
    print(f"包含透明層的圖片: {transparent_count}")
    print(f"不包含透明層的圖片: {total_count - transparent_count}")
    # 多線程轉換
    if transparent_count > 0:
        converted_count = batch_convert_transparent_to_white(results, max_workers=max_workers)
        print(f"\n已將 {converted_count} 個透明圖層的圖片轉換為白色背景")
    # 將結果保存到CSV文件
    csv_path = os.path.join(root_dir, "transparency_results.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['file_path', 'has_transparency', 'converted_to_white']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            if 'converted_to_white' not in result:
                result['converted_to_white'] = False
            writer.writerow(result)
    print(f"\n結果已保存至: {csv_path}")

if __name__ == "__main__":
    main()