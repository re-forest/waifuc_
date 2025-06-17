#!/usr/bin/env python3
"""
展示修復後的人臉偵測功能 - 真正為訓練目的過濾圖片
"""

def demo_training_oriented_face_detection():
    """展示面向訓練的人臉偵測功能"""
    
    print("=" * 70)
    print("WaifuC 訓練導向人臉偵測功能")
    print("=" * 70)
    
    print("\n❌ 之前的問題:")
    print("1. 沒有提供設定目標人臉數量的選項")
    print("2. 只是分類了圖片，但沒有為訓練目的進行過濾")
    print("3. 無法指定要保留或排除特定人臉數量的圖片")
    
    print("\n✅ 現在的修復:")
    print("1. 增加了完整的配置選項")
    print("2. 實現了真正的訓練導向過濾邏輯")  
    print("3. 提供了靈活的過濾模式")
    
    print("\n⚙️ 新增配置選項:")
    print("""
# config/settings.py
FACE_DETECTION_TARGET_FACE_COUNT = 1        # 訓練需要的人臉數量
FACE_DETECTION_FILTER_MODE = "keep_target"  # 過濾模式
FACE_DETECTION_TRAINING_DIR = "training_faces"  # 訓練圖片目錄
FACE_DETECTION_EXCLUDED_DIR = "excluded_faces"  # 排除圖片目錄
    """)
    
    print("\n🎯 過濾模式說明:")
    print("• keep_target: 保留目標人臉數量的圖片，排除其他")
    print("• exclude_target: 排除目標人臉數量的圖片，保留其他")  
    print("• classify_all: 只分類不過濾，保留所有圖片")
    
    print("\n📋 使用範例:")
    
    print("\n1️⃣ 只要單人圖片進行訓練 (最常見):")
    print("""
# 配置
target_face_count = 1
filter_mode = "keep_target"

# 結果: 只保留有1個人臉的圖片用於訓練，其他都排除
    """)
    
    print("2️⃣ 排除多人圖片，保留單人和無人臉圖片:")
    print("""
# 配置  
target_face_count = 2  # 排除2人以上的圖片
filter_mode = "exclude_target"

# 結果: 排除有2個人臉的圖片，保留0人、1人、3人以上的圖片
    """)
    
    print("3️⃣ 不限制人臉數量:")
    print("""
# 配置
target_face_count = -1  # 不限制
filter_mode = "keep_target"

# 結果: 保留所有圖片，不過濾
    """)
    
    print("\n🚀 命令行使用:")
    print("""
# 只保留單人圖片用於訓練
python batch_processor_final.py input_images/ output_images/ \\
    --only-face-detection \\
    --target-face-count 1 \\
    --face-filter-mode keep_target

# 排除多人圖片  
python batch_processor_final.py input_images/ output_images/ \\
    --only-face-detection \\
    --target-face-count 2 \\
    --face-filter-mode exclude_target

# 只分類不過濾
python batch_processor_final.py input_images/ output_images/ \\
    --only-face-detection \\
    --face-filter-mode classify_all
    """)
    
    print("\n💻 程式中調用:")
    print("""
from services.face_detection_service import filter_images_for_training
from config import settings
from utils.logger_config import setup_logging

# 設定配置
settings.FACE_DETECTION_TARGET_FACE_COUNT = 1  # 只要單人圖片
settings.FACE_DETECTION_FILTER_MODE = "keep_target"

# 執行過濾
logger = setup_logging("face_filter", settings.LOG_DIR, "INFO")
success, summary, results = filter_images_for_training(
    "input_images/", logger, settings
)

print(f"過濾結果: {summary}")
print(f"訓練圖片: {len(results['training_images'])} 個")
print(f"排除圖片: {len(results['excluded_images'])} 個")
print(f"人臉分佈: {results['face_distribution']}")
    """)
    
    print("\n📂 處理後的目錄結構:")
    print("""
input_images/
├── training_faces/     # 符合訓練要求的圖片
│   ├── single1.jpg    # 1個人臉 ✓
│   ├── single2.jpg    # 1個人臉 ✓
│   └── single3.jpg    # 1個人臉 ✓
└── excluded_faces/     # 不符合要求的圖片
    ├── landscape.jpg  # 0個人臉 ✗
    ├── couple.jpg     # 2個人臉 ✗
    └── group.jpg      # 5個人臉 ✗
    """)
    
    print("\n📊 詳細統計報告:")
    print("""
{
    "filter_stats": {
        "total_files": 100,
        "processed": 100,
        "training_count": 65,    # 符合訓練要求
        "excluded_count": 35,    # 不符合要求
        "error_count": 0
    },
    "face_distribution": {
        0: 15,  # 15張無人臉圖片
        1: 65,  # 65張單人臉圖片 (保留)
        2: 15,  # 15張雙人臉圖片 (排除)
        3: 5    # 5張多人臉圖片 (排除)
    },
    "training_images": [...],   # 訓練圖片路徑列表
    "excluded_images": [...]    # 排除圖片路徑列表
}
    """)

def show_ui_integration():
    """展示UI集成的可能性"""
    
    print("\n🖥️ UI界面整合建議:")
    print("""
在Gradio UI中可以新增以下控制項:

1. 人臉數量設定:
   • 下拉選單: 0, 1, 2, 3+, 不限制(-1)
   • 預設值: 1 (最適合大多數訓練用途)

2. 過濾模式設定:
   • 單選按鈕: 
     - "保留指定人臉數量" (keep_target)
     - "排除指定人臉數量" (exclude_target)  
     - "只分類不過濾" (classify_all)

3. 結果顯示:
   • 訓練可用圖片數量
   • 排除圖片數量
   • 人臉分佈統計圖表
   • 處理進度條

4. 輸出目錄設定:
   • 訓練圖片資料夾名稱
   • 排除圖片資料夾名稱
    """)

def main():
    """主函數"""
    demo_training_oriented_face_detection()
    show_ui_integration()
    
    print("\n" + "=" * 70)
    print("✅ 人臉偵測功能已修復為真正的訓練導向!")
    print("=" * 70)
    
    print("\n🎯 核心改進:")
    print("1. ✅ 新增目標人臉數量配置選項")
    print("2. ✅ 實現真正的過濾邏輯，而不只是分類")
    print("3. ✅ 提供靈活的過濾模式選擇")
    print("4. ✅ 自動分離訓練圖片和排除圖片")
    print("5. ✅ 詳細的統計報告和處理日誌")
    
    print("\n💡 現在可以:")
    print("• 指定只保留單人圖片用於人物訓練")
    print("• 排除多人圖片避免訓練混亂")
    print("• 根據具體需求靈活調整過濾條件")
    print("• 獲得完整的處理統計和品質報告")

if __name__ == "__main__":
    main()