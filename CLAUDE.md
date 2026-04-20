# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Activate the virtualenv before any Python command:
- Windows: `unhealthy-dataset-generator-env\Scripts\activate.bat`
- macOS: `source ./unhealthy-dataset-generator-env/bin/activate`

- Install deps: `pip install -r requirements.txt`
- Run the generator: `python main.py` (from the repo root — paths in the config are resolved relative to CWD)
- Inspect/tune crop params for a video: `python test_custom_crop_params.py --video_path <path> --top <n> --bottom <n> --left <n> --right <n>`. This opens a window showing the cropped frame and prints a `"crop": {...}` snippet to paste into `videos_metadata.<filename>.crop` in the config.

`pytesseract` requires the Tesseract OCR binary to be installed and on PATH (only exercised by `src/time_calculator.extract_datetime`).

There is no test suite, linter, or formatter configured.

## Configuration model

`main.py:load_config` looks for `config.json` in CWD and falls back to `default_config.json`. `validate_config` walks `default_config.json` recursively and requires every key to be present in `config.json` with a matching Python type — so `default_config.json` is the schema source of truth. Any new config key must be added there first (with a realistic default) or validation will reject user configs that include it.

Key config fields and how they drive behavior:
- `path.dataset`: root that must contain `metadata.xlsx`; output is written to `{dataset}/result/ad/` (wiped on each run — see `generator.py`).
- `path.videos`: directory where the `.mp4` files live.
- `tip_values`: integers. A metadata row is processed only if its `tip` column value *starts with* `"{n}="` for some `n` in this list (e.g. `tip_values=[2,3]` matches `"2=foo"` but not `"1=foo"`).
- `tv_channels_mapping`: maps the first character of the `can` column to the filename suffix (e.g. `"1" → "DN"` means `can` starting with `1` reads `{date}_DN.mp4`).
- `videos_metadata[filename].start_time` (`HH:MM:SS`): wall-clock time the video recording begins, used to convert the ad's `hin`/`hfi` times into video offsets.
- `videos_metadata[filename].crop` (`top`/`bottom`/`left`/`right`): pixel crop applied to every extracted frame. Optional — if omitted, frames are saved uncropped.

## Architecture

The pipeline is a straight line: `main.py` → `src/generator.py` → `src/processor.py` → `src/time_calculator.py`.

1. **`generator.generate_dataset`** reads `{dataset}/metadata.xlsx` (first sheet) into a DataFrame, deletes any prior `{dataset}/result/`, then iterates rows and dispatches matching ones to `process_row`.
2. **`processor.process_row`** is where per-row business logic lives. It resolves the video filename from the row's date (`fec`) and channel (`can`), validates that file + `videos_metadata` entry exist, computes the ad's start/end offsets, and calls `extract_frames`. Row-level errors are logged and the row is skipped — one bad row never aborts the whole run.
3. **`processor.extract_frames`** pulls one frame per second of the ad window (stepping `frame_count` by `fps`), optionally crops it, and writes `{custom_name}_{counter}.jpg`. It keeps `video_capture`, `fps`, `total_frames`, and `previous_video_path` as **module-level globals** so consecutive rows from the same video reuse the open `cv2.VideoCapture` — do not refactor this into per-call state without replacing it with an equivalent cache, or throughput will collapse on large metadata files.
4. **`time_calculator.get_times`** just subtracts `datetime`s to get `(start_seconds, end_seconds)` offsets into the video. `extract_datetime` (OCR via pytesseract) is defined but not wired into the main pipeline.

### Metadata schema (columns consumed from `metadata.xlsx`)
- `tip` — ad type tag, filtered by `tip_values`
- `can` — channel code, mapped via `tv_channels_mapping`
- `fec` — date (accepts `datetime` or any of the formats listed in `processor.formats`)
- `hin` / `hfi` — ad start/end `datetime.time`
- `cod` — row id, embedded into every output filename (`{video}_{cod}_{counter}.jpg`)

## Expected on-disk layout

```
{path.dataset}/
  metadata.xlsx
  result/ad/            # generated, wiped each run
{path.videos}/
  YYYY-MM-DD_{channel}.mp4
```
