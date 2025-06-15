import sys
import os
import logging # 導入標準 logging 模組

# --- 設定 Python Path ---
# 確保專案根目錄 (waifuc_) 在 sys.path 中，以便進行絕對導入
# __file__ 是 waifuc_/main.py
# current_dir 是 waifuc_
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
# --- 設定 Python Path 結束 ---

# 現在可以安全地從專案內部導入模組
try:
    from config import settings
    from ui.app import create_ui
    from utils.logger_config import setup_logging
    from utils.error_handler import handle_exception
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure that the project structure is correct and all __init__.py files are in place.")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)

# 全局 logger 實例
# 注意：logger 的初始化需要在 settings 模組成功導入之後
main_logger = None

def initialize_global_services():
    global main_logger
    # 初始化主日誌記錄器
    main_logger = setup_logging(
        module_name="app_main", # 或者 settings.APP_NAME
        log_dir=settings.LOG_DIR,
        log_level_str=settings.LOG_LEVEL,
        max_bytes=settings.LOG_ROTATION_MAX_BYTES,
        backup_count=settings.LOG_ROTATION_BACKUP_COUNT
    )
    main_logger.info(f"日誌系統初始化完成。日誌級別: {settings.LOG_LEVEL}, 日誌目錄: {settings.LOG_DIR}")

    # 設定全域未捕獲異常處理器
    # lambda 函數會捕獲定義時的 main_logger
    sys.excepthook = lambda exc_type, exc_value, exc_traceback: \
        handle_exception(exc_type, exc_value, exc_traceback, main_logger, "Global Unhandled Exception")
    main_logger.info("全域異常處理器已設定。")

def start_application():
    if not main_logger:
        print("Error: Logger not initialized. Call initialize_global_services() first.")
        # 嘗試進行一次初始化
        try:
            initialize_global_services()
            if not main_logger: # 再次檢查
                 print("Logger initialization failed. Exiting.")
                 sys.exit(1)
        except Exception as e:
            print(f"Critical error during logger initialization: {e}")
            sys.exit(1)

    main_logger.info(f"準備啟動 {settings.GRADIO_TITLE}...")
    try:
        app_ui = create_ui() # create_ui 應使用 settings 中的配置
        
        main_logger.info(f"Gradio 應用程式即將在 {settings.GRADIO_SERVER_NAME}:{settings.GRADIO_SERVER_PORT} 啟動。")
        if settings.GRADIO_SHARE:
            main_logger.info("Gradio 'share' 功能已啟用。將生成公開鏈接。")

        app_ui.launch(
            server_name=settings.GRADIO_SERVER_NAME,
            server_port=settings.GRADIO_SERVER_PORT,
            share=settings.GRADIO_SHARE,
            # prevent_thread_lock=True # 在 Windows 上如果遇到問題可以嘗試啟用
        )
        main_logger.info("Gradio 應用程式已成功關閉。") # launch() 是阻塞的，這行在關閉後執行
    except ImportError as e:
        # 這種錯誤通常在 create_ui 內部發生，如果 ui.app 又嘗試導入但路徑有問題
        main_logger.error(f"導入錯誤導致 Gradio 啟動失敗: {e}", exc_info=True)
        print(f"Import error during Gradio UI creation or launch: {e}")
    except Exception as e:
        main_logger.error(f"啟動 Gradio 應用程式時發生未預期錯誤: {e}", exc_info=True)
        # 也可以使用 handle_exception，但此處 logger 已捕獲
        # handle_exception(type(e), e, e.__traceback__, main_logger, "Gradio Launch Error")
        print(f"An unexpected error occurred while starting the Gradio application: {e}")

if __name__ == "__main__":
    try:
        initialize_global_services() # 首先初始化日誌和全域異常處理
        start_application()          # 然後啟動應用程式
    except Exception as e:
        # 這是最後的防線，如果 initialize_global_services 本身出錯且未被捕獲
        print(f"CRITICAL ERROR in main execution: {e}")
        if main_logger: # 如果 logger 初始化成功了，嘗試記錄
            main_logger.critical(f"CRITICAL ERROR in main execution: {e}", exc_info=True)
        sys.exit(1) # 嚴重錯誤，退出
