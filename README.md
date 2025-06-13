# 圖像自動化預處理專案

這是一個圖像處理專案，旨在自動化動漫風格圖像的預處理流程，為 AI 模型訓練做準備。其主要功能包括圖像驗證、人臉偵測、圖像去重複、圖像裁切、圖像分類、圖像放大以及圖像標記。

## 架構說明

專案採用模組化設計，各功能由獨立的 Python 腳本實現：

*   `main.py`: 主執行腳本，協調各個處理模組的執行順序。
*   `validate_image.py`: 驗證圖像檔案的完整性，移除損壞檔案。
*   `face_detection.py`: 偵測圖像中的人臉，並可根據人臉數量篩選圖像。
*   `lpips_clustering.py`: 使用 LPIPS (Learned Perceptual Image Patch Similarity) 演算法對圖像進行聚類，以去除重複或高度相似的圖像。
*   `crop.py`: 將圖像裁切成不同尺寸，例如全身、半身、頭像。
*   `tag.py`: 自動為圖像生成描述性標籤。
*   `upscale.py`: 對低解析度圖像進行超解析度放大。
*   `gradio_app.py`: 提供一個 Gradio Web UI 介面，方便使用者操作。
*   `logger_config.py`: 統一的日誌系統配置，提供結構化的日誌記錄和檔案輸出。
*   `error_handler.py`: 統一的錯誤處理系統，提供專案自定義異常和安全執行包裝函數。

## 技術棧

*   Python
*   ONNX Runtime (用於深度學習模型推論)
*   imgutils (圖像處理工具庫)
*   Waifuc (動漫圖像處理工具庫)
*   Dotenv (環境變數管理)
*   Matplotlib (圖像顯示)
*   Boto3 (AWS SDK，可能用於雲端儲存)
*   Gradio (快速建立機器學習 Web UI)
*   Tqdm (進度條顯示)

## 核心功能特色

### 🛡️ 完善的錯誤處理
- **統一異常系統**: 所有模組使用一致的錯誤分類和處理邏輯
- **安全執行機制**: 單一處理失敗不會影響整個流程
- **友好錯誤訊息**: 將技術錯誤轉換為使用者可理解的說明
- **詳細錯誤日誌**: 提供完整的錯誤追蹤和診斷資訊

### 📊 結構化日誌系統
- **多級別記錄**: 支援 DEBUG、INFO、WARNING、ERROR、CRITICAL 級別
- **雙重輸出**: 同時輸出到控制台和檔案，按模組和日期分類
- **環境變數控制**: 透過 `.env` 檔案靈活配置日誌行為
- **UTF-8 支援**: 完整支援中文日誌記錄

### 🌐 直觀的 Web 介面
- **參數整合控制**: 在 UI 中直接配置所有處理參數
- **即時進度顯示**: 為所有長時間操作提供進度回饋
- **視覺化預覽**: 即時查看處理結果的樣本圖像
- **整合式流程**: 提供統一介面執行完整預處理管線

### 🧪 全面的測試系統 (100% 通過率)
- **完整測試覆蓋**: 所有核心模組都有自動化測試，150個測試案例全部通過 ✅
- **錯誤模擬**: 測試各種異常情況和邊界條件處理
- **穩定可靠**: 100% 測試通過率確保程式碼變更後的穩定性
- **測試隔離**: 使用臨時環境確保測試間的獨立性

## 安裝指南

1.  複製專案：`git clone <repository_url>`
2.  進入專案目錄：`cd waifuc_`
3.  安裝依賴：`pip install -r requirements.txt`
4.  設定環境變數：複製 `.env.example` (如果有的話) 為 `.env`，並填寫必要的路徑和參數。

## 執行指南

1.  確保已完成安裝步驟並設定好環境變數。
2.  執行主程式：`python main.py`
3.  (如果 `gradio_app.py` 存在且已設定) 執行 Gradio 應用程式：`python gradio_app.py`

## 測試

### 執行測試
```bash
# 執行所有測試
python -m pytest tests/ -v

# 執行特定測試文件
python -m pytest tests/test_validate_image.py -v

# 執行測試並查看覆蓋率
python -m pytest tests/ --cov=. --cov-report=term-missing
```

### 測試狀態 (🎉 完美成績)
- **總測試數**: 150個
- **通過率**: 100% (150/150) 🏆
- **測試覆蓋率**: 95%+ (所有核心功能已覆蓋)
- **所有模組狀態**: 全部優秀 ✅
  - error_handler.py: 17/17 (100%)
  - validate_image.py: 10/10 (100%)  
  - lpips_clustering.py: 18/18 (100%)
  - logger_config.py: 17/17 (100%)
  - face_detection.py: 14/14 (100%)
  - tag.py: 22/22 (100%)
  - crop.py: 18/18 (100%)
  - upscale.py: 34/34 (100%)
