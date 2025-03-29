# Waifuc 圖像處理工具箱

這是一個基於 waifuc 函式庫的綜合圖像處理工具箱，專為動漫圖像的自動化處理而設計。此工具可以執行圖像檢驗、人臉偵測、聚類去重、裁切分類、標籤產生以及圖像放大等多種功能。

## 功能特色

- **圖像檢驗**：偵測並移除損壞的圖像檔案
- **人臉偵測**：根據圖像中的人臉數量進行分類
- **LPIPS聚類**：基於LPIPS（Learned Perceptual Image Patch Similarity）進行圖像去重
- **自動裁切**：自動裁切圖像成完整人像、半身像、頭像格式
- **分類整理**：根據裁切結果分類整理到不同的資料夾
- **標籤產生**：使用深度學習模型自動為圖像產生描述性標籤
- **圖像放大**：使用先進的超解析度模型放大低解析度圖像

## 安裝需求

1. 安裝所需的Python套件：

```bash
pip install -r requirements.txt
```

## 設定與使用

1. 編輯 `.env` 檔案設定參數：

```properties
# 圖片來源資料夾路徑
directory="圖片來源路徑"

# 圖片處理後的輸出資料夾路徑
output_directory="輸出路徑"

# 啟用或停用特定功能
enable_validation=true
enable_cropping=true
enable_classification=true
enable_tagging=true
enable_face_detection=true
enable_lpips_clustering=true
enable_upscaling=true

# 更多設定請參考 .env 檔案中的註解
```

2. 執行主程式：

```bash
python main.py
```

## 單獨執行各功能模組

除了使用主程式整合所有功能外，也可以單獨執行各個模組：

### 圖像檢驗

```bash
python validate_image.py
```

### 人臉偵測

```bash
python face_detection.py
```

### LPIPS聚類去重

```bash
python lpips_clustering.py [目錄路徑] --output [輸出目錄] --batch-size [批次大小]
```

### 圖像裁切與分類

```bash
python crop.py --input_path [輸入目錄] --output_path [輸出目錄] --include_subfolders
```

### 標籤產生

```bash
python tag.py
```

### 圖像放大

```bash
python upscale.py [目錄路徑] --width [目標寬度] --height [目標高度] --model [模型名稱]
```

## 環境變數說明

| 變數名稱 | 說明 | 預設值 |
|---------|------|-------|
| directory | 圖片來源資料夾路徑 | 無，必須指定 |
| output_directory | 處理後圖片輸出路徑 | 無，必須指定 |
| enable_validation | 啟用圖片完整性檢驗 | true |
| enable_cropping | 啟用圖片裁切功能 | true |
| enable_classification | 啟用圖片分類功能 | true |
| enable_tagging | 啟用圖片標籤功能 | true |
| enable_face_detection | 啟用人臉偵測功能 | true |
| enable_lpips_clustering | 啟用LPIPS聚類功能 | true |
| enable_upscaling | 啟用圖片放大功能 | true |
| num_threads | 使用的執行緒數量 | 2 |

## 致謝

本專案基於以下函式庫：
- [waifuc](https://github.com/deepghs/waifuc)
- [imgutils](https://github.com/deepghs/imgutils)
- [onnxruntime](https://onnxruntime.ai/)
