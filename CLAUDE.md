# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WaifuC is an AI-powered image processing application specialized for anime/manga-style images. It supports both single image and batch processing modes with features including validation, face detection, upscaling, tagging, cropping, and clustering. The application provides both a Gradio web UI and command-line interface for batch processing entire directories with recursive subdirectory support.

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run main application (Web UI)
python main.py

# Run UI directly for testing
python ui/app.py

# Batch processing via command line
python batch_processor.py input_dir output_dir [options]

# Run tests
python -m pytest tests/
python -m pytest tests/test_specific_service.py  # Single test file
```

## Architecture Overview

The application follows a **Service-Oriented Architecture** with an **Orchestrator Pattern**:

- **Core Layer**: `core/orchestrator.py` coordinates all processing steps
- **Service Layer**: Independent services in `services/` handle specific processing tasks
- **UI Layer**: `ui/app.py` provides Gradio web interface with comprehensive error handling
- **Utils Layer**: `utils/` contains shared utilities, logging, and error handling

### Key Services

- `validator_service.py` - Image integrity validation
- `face_detection_service.py` - Anime face detection using imgutils
- `upscale_service.py` - AI upscaling with multiple models
- `tag_service.py` - Auto-tagging using WD14 models
- `crop_service.py` - Face-based cropping
- `lpips_clustering_service.py` - Image similarity clustering
- `file_service.py` - File I/O operations

### Configuration System

- `config/settings.py` - Central configuration with all processing parameters
- `.env` file - Runtime configuration and feature flags
- Auto-creates required directories: `input_images/`, `output_images/`, `logs/`, `models/`, etc.

### Processing Flow

1. **Orchestrator** reads configuration and initializes enabled services
2. **Processing Pipeline** executes steps based on configuration flags
3. **Services** operate independently with shared error handling
4. **UI** provides real-time feedback and detailed logging

### Batch Processing Features

- **Directory Scanning**: Recursive subdirectory support with configurable file filtering
- **Output Structure**: Preserve original directory structure or centralized output
- **Progress Tracking**: Detailed statistics and error reporting
- **Flexible Configuration**: Step-by-step processing control via UI or command line
- **Dual Interface**: Web UI for interactive use, CLI for automation/scripting

## Testing Structure

- `tests/test_base.py` - Common test utilities and TestConfig class
- Individual service tests follow naming pattern `test_{service_name}.py`
- Mock objects and test data are centralized in test base
- Error handling is extensively tested

## Key Dependencies

- `waifuc[gpu]` - Core image processing library
- `dghs-imgutils` - Computer vision utilities for anime images
- `gradio==4.15.0` - Web UI framework
- `onnxruntime-gpu==1.21.0` - AI model runtime

## Error Handling

- Custom exception hierarchy with `WaifucBaseError` base class
- All services use safe execution wrappers
- UI layer has comprehensive fallback mechanisms
- Detailed logging with rotation in `logs/` directory

## Configuration Notes

- Services can be individually enabled/disabled via config flags
- Model parameters (thresholds, batch sizes) are configurable
- Supports both legacy function-based and modern class-based orchestration
- Multi-language support (Chinese/Traditional Chinese UI elements)