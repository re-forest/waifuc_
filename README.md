# Waifuc 圖像處理工具箱

這是一個基於 waifuc 函式庫的綜合圖像處理工具箱，此工具可以執行圖像檢驗、人臉偵測、聚類去重、裁切分類、標籤產生以及圖像放大等功能。

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

## 工作流程說明

此工具設計了一個完整的圖像處理流程，每個步驟都有明確的目的，特別適合於AI模型訓練前的資料預處理：

1. **圖像檢驗**：首先移除所有損壞或不完整的圖像，確保後續處理不會因為圖像問題而中斷，也避免將損壞的圖像送入訓練流程。

2. **人臉偵測**：將圖像按照人臉數量進行分類，特別是將單一人臉的圖像分離出來。對於AI模型訓練，單一人臉的圖像能夠讓模型更專注於學習特定人物特徵，避免多人圖像造成的特徵混淆。

3. **LPIPS聚類去重**：使用感知相似度指標對圖像進行去重分類，避免在訓練集中包含過多相似圖像。相似度過高的圖像不僅浪費訓練資源，還可能導致模型過度學習某些特定場景或姿勢，降低模型的泛化能力。

4. **自動裁切與分類**：將人物圖像自動裁切為全身像、半身像和頭像三種格式，並分類整理。這一步驟能讓模型從不同尺度學習人物特徵：
   - 頭像：專注於面部細節和表情特徵
   - 半身像：學習上半身比例和姿態特徵
   - 全身像：學習完整的人物體態和姿勢特徵
   
   多尺度學習可以顯著提升模型對人物特徵的理解和生成能力。

5. **圖像放大**：對低解析度圖像進行超解析度處理，確保所有訓練圖像達到一定的質量標準。高品質的訓練資料能夠讓模型學習到更細緻的特徵，提升生成結果的品質。

6. **標籤產生**：自動為每張圖像生成描述性標籤，這些標籤不僅可以用於訓練條件式生成模型，還能夠幫助組織和檢索圖像資料集。準確的標籤對於控制生成結果的特定屬性至關重要。

這個工作流程設計為漸進式的，每一步都建立在前一步的基礎上，形成一個完整的資料預處理管道，為AI模型訓練提供高品質且結構良好的圖像資料集。

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
| lpips_output_directory | LPIPS聚類輸出目錄 | 空 |
| face_output_directory | 人臉檢測輸出目錄 | 空 |
| min_face_count | 人臉檢測的最小人臉數量 | 2 |
| custom_character_tag | 自訂角色標籤 | 空 |
| custom_artist_name | 自訂繪師名稱 | Pas6telIl98lust |
| enable_wildcard | 啟用wildcard功能 | true |
| lpips_batch_size | LPIPS聚類批次處理大小 | 100 |
| upscale_target_width | 放大後的目標寬度 | 1024 |
| upscale_target_height | 放大後的目標高度 | 1024 |
| upscale_model | 放大使用的模型 | HGSR-MHR-anime-aug_X4_320 |
| upscale_min_size | 最小尺寸閾值 | 800 |

## Gradio 介面預設值

執行 `gradio_app.py` 可使用圖形介面處理圖片。若 `.env` 檔案中設置了
`directory` 或 `output_directory` 等環境變數，這些值會自動填入對應的輸入框，
若未設定則輸入框會顯示如 `/path/to/images` 的提示文字。

## 致謝

本專案基於以下函式庫：
- [waifuc](https://github.com/deepghs/waifuc)
- [imgutils](https://github.com/deepghs/imgutils)
- [onnxruntime](https://onnxruntime.ai/)
