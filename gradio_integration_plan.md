## Gradio 整合計畫

### 1. 專案目標

將 Gradio 使用者介面整合到現有的圖片處理專案中，提供兩種主要的使用方式：
1.  一個可以執行 `main.py` 中定義的完整圖片處理流程的介面。
2.  為選定的核心圖片處理單元提供獨立的 Gradio 操作介面。

### 2. 現有功能分析

專案目前包含以下主要功能模組，由 `main.py` 進行串接：

*   **圖片驗證 (`validate_image.py`)**: 移除損壞或不完整的圖片。
*   **透明通道處理 (`transparency.py`)**: 檢查圖片透明度，並可將透明背景轉換為白色背景。
*   **人臉偵測 (`face_detection.py`)**: 偵測圖片中的人臉，並根據設定篩選（例如排除特定數量以上人臉的資料）。
*   **圖片去重複 (`lpips_clustering.py`)**: 使用 LPIPS 進行聚類，找出並處理相似度過高的圖片。
*   **圖片裁切 (`crop.py`)**: 將人物圖像切割成全身、半身、頭像等。
*   **檔案分類 (`crop.py`)**: 將裁切後的圖像分類到不同資料夾。
*   **圖片放大 (`upscale.py`)**: 對低解析度圖像進行超解析度處理。 (在完整流程中為可選)
*   **圖片標記 (`tag.py`)**: 自動為圖像生成描述性標籤。

所有功能目前主要以處理指定目錄下的檔案為主。

### 3. Gradio 介面設計

#### 3.1. 完整流程介面 (`main.py` 流程)

*   **輸入元件**:
    *   `gr.Textbox`：用於指定主要處理目錄路徑 (對應 `.env` 中的 `directory`)。
    *   `gr.CheckboxGroup`：用於選擇要執行的主要處理步驟 (例如：\"驗證圖片\", \"透明度處理\", \"人臉偵測\", \"圖片去重複\", \"裁切與分類\", \"圖片標記\")。預設選擇核心步驟。
    *   `gr.Checkbox`：\"執行圖片放大 (可選)\"。預設不勾選。
    *   `gr.Button`：\"開始完整處理流程\"。
    *   (可選) 各步驟的進階設定，例如：
        *   `gr.Number`：人臉偵測的最小人臉數。
        *   `gr.Textbox`：人臉偵測輸出目錄。
        *   `gr.Textbox`：LPIPS 輸出目錄。
        *   `gr.Number`：LPIPS 批次大小。
        *   `gr.Textbox`：裁切與分類的輸出目錄。
        *   如果勾選了 \"執行圖片放大\":
            *   `gr.Number`：放大目標寬度/高度。
            *   `gr.Textbox`：放大模型名稱。
*   **輸出元件**:
    *   `gr.Textbox` (多行)：顯示處理日誌與進度。
    *   `gr.Markdown`：顯示最終結果摘要，例如處理圖片數量、錯誤訊息、輸出檔案路徑等。

#### 3.2. 單元功能介面 (個別模組)

為選定的核心功能建立獨立的 Gradio 頁籤或區塊。

*   **A. 圖片驗證 (`validate_image.py`)**
    *   輸入: `gr.Textbox` (目錄路徑)。
    *   輸出: `gr.Textbox` (顯示驗證結果，如移除的檔案列表或摘要)。
*   **B. 透明度處理 (`transparency.py`)**
    *   輸入: `gr.Textbox` (目錄路徑)。
    *   按鈕1: \"掃描透明圖片\"。
    *   輸出1: `gr.DataFrame` 或 `gr.JSON` (顯示掃描結果，包含檔案路徑、是否透明)。
    *   按鈕2: \"轉換透明圖片為白色背景\" (基於掃描結果或直接處理目錄)。
    *   輸出2: `gr.Textbox` (顯示轉換日誌與摘要)。
*   **C. 人臉偵測 (`face_detection.py`)**
    *   輸入:
        *   `gr.Textbox` (輸入目錄路徑)。
        *   `gr.Number` (用以排除人臉數量 **大於等於** 此設定值的圖片，例如設為 2 表示只保留單一人臉圖片)。
        *   `gr.Textbox` (輸出目錄路徑)。
    *   輸出: `gr.Textbox` (顯示處理日誌與摘要)。
*   **D. 圖片去重複 (`lpips_clustering.py`)**
    *   輸入:
        *   `gr.Textbox` (輸入目錄路徑，或讓使用者上傳檔案列表)。
        *   `gr.Textbox` (輸出目錄路徑)。
        *   `gr.Number` (批次大小)。
        *   `gr.Slider` (相似度閾值，用於判斷是否為重複圖片)。
    *   輸出: `gr.Textbox` (顯示處理日誌與結果儲存路徑)。
*   **E. 圖片裁切與分類 (`crop.py`)**
    *   輸入:
        *   `gr.Textbox` (輸入目錄路徑)。
        *   `gr.Textbox` (輸出目錄路徑, 裁切後的圖片會儲存於此，並在此目錄下進行分類)。
    *   按鈕: \"執行裁切與分類\"。
    *   輸出: `gr.Textbox` (顯示處理日誌與摘要)。
*   **F. 圖片標記 (`tag.py`)**
    *   輸入: `gr.Textbox` (輸入目錄路徑，通常是已處理完成的圖片目錄)。
    *   輸出: `gr.Textbox` (顯示處理日誌與摘要，標籤檔案已儲存於圖片旁)。

### 4. 程式碼重構計畫

為了方便 Gradio 呼叫，需要對現有模組進行適度重構：

*   **通用原則**:
    *   每個 `.py` 檔案中的主要功能應封裝成可獨立呼叫的函式。
    *   函式應接受明確的參數 (如輸入路徑、輸出路徑、設定值)，避免過度依賴全域變數或固定的 `.env` 設定 (Gradio 介面中可以提供這些設定的輸入)。
    *   函式應回傳有意義的結果，例如處理狀態、日誌訊息、輸出檔案路徑等，以便在 Gradio 中顯示。
    *   `.env` 檔案的載入 (`load_dotenv()`) 可以在 Gradio 應用程式啟動時執行一次，或者讓函式接受一個包含設定的字典。

*   **`main.py`**:
    *   將 `main()` 函式的主體邏輯提取到一個新的函式，例如 `run_full_pipeline(config: dict)`。
    *   `config` 字典可以包含所有原先從 `os.getenv()` 讀取的設定。
    *   此函式應回傳一個包含執行日誌和結果摘要的字串或字典。

*   **各單元模組 (`validate_image.py`, `transparency.py`, etc.)**:
    *   **`validate_image.py`**: `validate_and_remove_invalid_images(directory_path: str) -> dict` 回傳處理摘要。
    *   **`transparency.py`**:
        *   `scan_directory(directory_path: str, max_workers: int = 8) -> list[dict]`
        *   `batch_convert_transparent_to_white(results: list[dict], max_workers: int = 8) -> int`
        *   可以考慮新增一個整合函式 `process_transparency(directory_path: str, convert: bool = True, max_workers: int = 8) -> dict` 回傳掃描與轉換的完整報告。
    *   **`face_detection.py`**: `detect_faces_in_directory(input_dir: str, min_face_count: int, output_dir: str) -> dict` 回傳處理摘要。
    *   **`lpips_clustering.py`**: `process_lpips_clustering(file_paths: list[str], output_directory: str, batch_size: int) -> str` 回傳結果目錄路徑。
    *   **`crop.py`**:
        *   `process_single_folder(input_folder: str, output_folder: str) -> dict` 回傳處理摘要。
        *   `classify_files_in_directory(directory_path: str) -> dict` 回傳處理摘要。
        *   可以考慮新增一個整合函式 `process_cropping_and_classification(input_dir: str, output_dir: str) -> dict`，該函式內部呼叫 `process_single_folder` 和 `classify_files_in_directory`，並回傳整體摘要。
    *   **`upscale.py`**: `upscale_images_in_directory(directory: str, target_width: int, target_height: int, model: str, recursive: bool, min_size: int = 0) -> dict` 回傳處理摘要。
    *   **`tag.py`**: `tag_image(image_path_or_directory: str) -> int` 回傳處理的圖片數量。

### 5. 開發步驟

1.  **環境準備**:
    *   安裝 Gradio: `pip install gradio`
    *   確保所有現有依賴已安裝並在虛擬環境中。
2.  **模組重構**:
    *   依照「4. 程式碼重構計畫」修改各 `.py` 檔案，使其函式更易於被外部呼叫。
    *   優先重構 `main.py` 使其流程可被函式呼叫。
3.  **建立 Gradio 應用程式 (`gradio_app.py`)**:
    *   建立新的 `gradio_app.py` 檔案。
    *   匯入重構後的模組函式。
4.  **實現完整流程介面**:
    *   在 `gradio_app.py` 中，使用 Gradio 元件建立「3.1. 完整流程介面」。
    *   連接介面元件到重構後的 `run_full_pipeline` 函式。
    *   進行初步測試。
5.  **實現單元功能介面**:
    *   逐步為每個功能模組建立 Gradio 介面 (如「3.2. 單元功能介面」所述)。
    *   連接介面元件到對應的重構函式。
    *   逐個測試每個單元功能。
6.  **日誌與回饋**:
    *   確保所有操作都有清晰的日誌輸出到 Gradio 介面。
    *   對於長時間執行的任務，研究 Gradio 的進度更新機制 (例如 `gr.Progress`)。
7.  **錯誤處理**:
    *   在 Gradio 呼叫的函式中加入適當的錯誤處理 (try-except)，並將錯誤訊息回饋到 UI。
8.  **測試與調整**:
    *   全面測試所有功能和介面。
    *   根據使用體驗調整介面佈局和互動。

### 6. 注意事項

*   **檔案路徑**: Gradio 的 `gr.File` 或 `gr.Textbox` 用於路徑輸入時，需確保後端函式能正確處理這些路徑字串。
*   **執行緒與效能**: 部分圖片處理任務可能耗時較長。Gradio 預設在執行緒中執行函式，但對於 CPU 密集型任務，需注意不要阻塞 Gradio 主執行緒太久。考慮非同步操作或更細緻的進度回饋。
*   **`.env` 管理**:
    *   可以在 `gradio_app.py` 啟動时 `load_dotenv()`。
    *   或者，將必要的設定值作為 Gradio 介面的輸入項，然後傳遞給後端函式。後者更靈活。
*   **輸出顯示**: 處理結果可能是檔案、目錄或文字報告。選擇合適的 Gradio 元件來顯示 (例如 `gr.Image` 預覽單張圖片，`gr.Files` 顯示檔案列表，`gr.Textbox` 或 `gr.Markdown` 顯示文字報告)。

此計畫將作為後續開發的指導。
