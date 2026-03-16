# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Video frame extraction tool that generates image datasets from TV channel recordings. It reads ad metadata from an Excel file (`metadata.xlsx`), locates corresponding video files, and extracts frames as JPEG images with optional cropping.

## Commands

```bash
# Install dependencies (use virtual environment)
source ./unhealthy-dataset-generator-env/bin/activate
pip install -r requirements.txt

# Run the main pipeline
python main.py

# Test crop parameters for a video
python test_custom_crop_params.py --video_path <path> --top <px> --bottom <px> --left <px> --right <px>
```

There are no test or lint commands configured.

## Architecture

**Data flow:** `config.json` → `main.py` → `generator.py` → `processor.py` → extracted JPEG frames

- **main.py** — Entry point. Loads and validates config (user `config.json` or `default_config.json`), then calls the generator.
- **src/generator.py** — Reads `metadata.xlsx`, filters rows by `tip_values` config, delegates each row to `processor.py`.
- **src/processor.py** — Maps Excel row data (date, channel, times) to a video file, calculates frame positions, extracts and saves frames. Caches `cv2.VideoCapture` objects in module-level globals to avoid reopening the same video.
- **src/time_calculator.py** — Converts ad start/end times relative to video start time (seconds offsets). Also has `extract_datetime()` OCR utility using pytesseract.

## Configuration

Config is a JSON file with: video/dataset paths, `tip_values` filter list, `tv_channels_mapping` (channel number → filename suffix like DN, CN, canal_6, canal_7), and per-video `videos_metadata` entries specifying `start_time` and optional `crop` parameters.

## Key Conventions

- Excel metadata columns: `fec` (date), `can` (channel), `hin` (start time), `hfi` (end time), `tip` (ad type), `cod` (identifier)
- Output frame naming: `{video_name}_{cod}_{counter}.jpg` saved under `{dataset_path}/result/ad/`
- Date parsing supports 6 formats for flexibility across different Excel sources
