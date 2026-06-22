# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Activate the virtualenv before any Python command:
- Windows: `unhealthy-dataset-generator-env\Scripts\activate.bat`
- macOS: `source ./unhealthy-dataset-generator-env/bin/activate`

- Install deps: `pip install -r requirements.txt`
- Run the generator: `python -m src.dataset_generator_pipeline.main` (from the repo root — paths in the config are resolved relative to CWD)
- Review generated frames: `python src/misc/review_dataset.py` — opens a Tkinter GUI that browses `result/ad/` by ad batch and `result/non_ad/` by page. Keyboard shortcuts: `A` select all, `D` discard & advance, `H`/`L` or arrow keys to navigate, `Ctrl+Z` undo last discard. Discarded frames are moved (not deleted) to `result/discarded/ad/` or `result/discarded/non_ad/`.
- Inspect/tune crop params for a video: `python src/misc/test_custom_crop_params.py --video_path <path> --top <n> --bottom <n> --left <n> --right <n>`. This opens a window showing the cropped frame and prints a `"crop": {...}` snippet to paste into `videos_metadata.<filename>.crop` in the config.
- Remove near-duplicate frames: `python src/duplicate_images_removal_pipeline/image_similarity.py [folder] [--dry-run] [--combined-threshold N] [--clip-weight N] [--hash-size N] [--batch-size N] [--chunk-size N]`. Defaults to scanning `{dataset}/result`; CLI flags override the `duplicate_images_removal` config block. Moves duplicates to `result/duplicates/` (preserving relative paths) and writes `duplicates_manifest.csv`.
- Rebalance class counts: `python src/rebalance_pipeline/rebalance.py [result] [--dry-run] [--target-ratio N] [--excess-dir DIR]`. Defaults to `{dataset}/result`; trims the majority of `ad`/`non_ad` (loss-only, never duplicates) to hit `non_ad:ad == target_ratio`, moving surplus to `result/excess/` with `rebalance_manifest.csv`.

These two pipelines and `review_dataset.py` are **post-processing stages run manually after the generator**, in any order. Each one only ever *moves* frames into a sibling folder under `result/` (`discarded/`, `duplicates/`, `excess/`) and skips those folders on subsequent scans (`EXCLUDED_DIRS`), so the operations are reversible and idempotent. They load config the same way as `main.py` (`config.json` in CWD, else `default_config.json`), so run them from the repo root.

OCR uses `tesserocr` (in-process libtesseract binding) — `pytesseract` was replaced because it spawned `tesseract.exe` per call, which dominated runtime on Windows. `time_calculator._get_api()` lazy-initializes a singleton `PyTessBaseAPI` that lives for the process lifetime; do not create per-call instances. The Tesseract binary is still required:
- Windows: [UB Mannheim build](https://github.com/UB-Mannheim/tesseract/wiki); add the install dir (e.g. `C:\Program Files\Tesseract-OCR`) to `PATH`. `time_calculator._resolve_tessdata_path` looks at `TESSDATA_PREFIX` first, then falls back to `C:\Program Files\Tesseract-OCR\tessdata`.
- macOS: `brew install tesseract`.
- Linux: `sudo apt install tesseract-ocr libtesseract-dev libleptonica-dev` (the dev packages are needed because `tesserocr` builds from source on non-Windows).

Verify with `tesseract --version`.

On Windows, `tesserocr` has no PyPI wheel and the UB Mannheim install lacks the dev libs to build from source — install a prebuilt wheel matching your Python version from [simonflueckiger/tesserocr-windows_build](https://github.com/simonflueckiger/tesserocr-windows_build/releases) **before** `pip install -r requirements.txt`.

The duplicate-removal pipeline pulls in `sentence-transformers` + `torch` (for CLIP embeddings, model `clip-ViT-B-32` downloaded on first run). On Windows, plain `pip install torch` is CPU-only — install the matching CUDA wheel first (e.g. `pip install torch --index-url https://download.pytorch.org/whl/cu124`) **before** `pip install -r requirements.txt` to use the GPU. The core generator (`main.py`) does not need torch.

There is no test suite, linter, or formatter configured.

## Import style

The pipeline modules use bare imports (`from generator import generate_dataset`, `from processor import process_row`, etc.) — not relative or absolute package imports. This works because `python -m src.dataset_generator_pipeline.main` puts the package directory on `sys.path`. Do not convert these to relative imports (e.g. `from .generator import …`) without testing; `-m` and relative imports interact in non-obvious ways with this layout.

## Configuration model

`src/dataset_generator_pipeline/main.py:load_config` looks for `config.json` in CWD and falls back to `default_config.json`. `validate_config` walks `default_config.json` recursively and requires every key to be present in `config.json` with a matching Python type — so `default_config.json` is the schema source of truth. Any new config key must be added there first (with a realistic default) or validation will reject user configs that include it.


Key config fields and how they drive behavior:
- `path.dataset`: root that must contain `metadata.xlsx`; output is written to `{dataset}/result/ad/` and `{dataset}/result/non_ad/` (wiped on each run).
- `path.videos`: directory where the `.mp4` files live.
- `tip_values`: integers. A metadata row is processed only if its `tip` column value *starts with* `"{n}="` for some `n` in this list (e.g. `tip_values=[2,3]` matches `"2=foo"` but not `"1=foo"`).
- `tv_channels_mapping`: maps the first character of the `can` column to the filename suffix (e.g. `"1" → "DN"` means `can` starting with `1` reads `{date}_DN_processed.mp4`).
- `videos_metadata[filename].start_time` (`HH:MM:SS`): wall-clock time the video recording begins, used to convert the ad's `hin`/`hfi` times into video offsets.
- `videos_metadata[filename].crop` (`top`/`bottom`/`left`/`right`): pixel crop applied to every extracted frame. Optional — if omitted, frames are saved uncropped.
- `non_ad_ratio`: float, default `1.0`. Number of non-ad frames to generate as a multiple of the total ad frame count.
- `non_ad_gap_seconds`: int, default `30`. Seconds before/after each ad window that are also excluded when selecting non-ad candidates.
- `duplicate_images_removal`: defaults for the dedup pipeline — `hash_size` (perceptual-hash grid), `combined_threshold` (score at/above which two frames are duplicates), `clip_weight` (split between CLIP cosine and hash similarity, 0–1), `batch_size`, `chunk_size`. All overridable per-run via CLI flags.
- `rebalance.target_ratio`: float, default `1.0`. Desired `non_ad:ad` ratio the rebalance pipeline trims toward.

## Architecture

The pipeline is a straight line: `src/dataset_generator_pipeline/main.py` → `src/dataset_generator_pipeline/generator.py` → `src/dataset_generator_pipeline/processor.py` → `src/dataset_generator_pipeline/time_calculator.py`, with `src/dataset_generator_pipeline/non_ad_generator.py` running after all ad rows are processed.

1. **`generator.generate_dataset`** reads `{dataset}/metadata.xlsx` (first sheet) into a DataFrame, deletes any prior `{dataset}/result/`, iterates rows dispatching matching ones to `process_row`, **accumulates the ad-frame count returned by each `process_row` call**, then passes that total to `non_ad_generator.generate_non_ad_images`. The running total replaced an earlier `glob.glob('result/ad/*.jpg')` rescan that walked thousands of files.
2. **`processor.process_row`** is where per-row business logic lives. It resolves the video filename from the row's date (`fec`) and channel (`can`), validates that file + `videos_metadata` entry exist, computes the ad's start/end offsets, calls `extract_frames`, and **returns the number of frames written** (0 on validation failure). Row-level errors are logged and the row is skipped — one bad row never aborts the whole run.
3. **`processor.extract_frames`** has a two-phase design:
   - **Phase 1 — boundary OCR.** Read the first and last sampled frames, OCR their burned-in clocks. If both timestamps land inside `[ad_start_time, ad_end_time]`, the seek math from `videos_metadata.start_time + hin/hfi` is correct and middle frames are guaranteed in range, so phase 2 skips OCR entirely (the fast path). If either boundary is missing or out of range, phase 2 falls back to OCR'ing every frame — same semantics as the original code (None means kept).
   - **Phase 2 — sequential extraction.** One `cap.set(POS_FRAMES, …)` to the window start, then advance one second at a time with `cap.grab() × (fps-1)` + `cap.read()`. The grab calls are required because video frames depend on prior frames between keyframes, but they're much cheaper than `set()` per second (which forces a keyframe seek + decode every iteration).

   The function keeps `video_capture`, `fps`, `total_frames`, and `previous_video_path` as **module-level globals** so consecutive rows from the same video reuse the open `cv2.VideoCapture` — do not refactor this into per-call state without replacing it with an equivalent cache, or throughput will collapse on large metadata files. Returns `extracted - removed` (the count of JPEGs written). Counter increments for every considered frame so dropped frames in the fallback path leave numbering gaps.
4. **`time_calculator.get_times`** subtracts `datetime`s to get `(start_seconds, end_seconds)` offsets into the video. `time_calculator.extract_datetime` OCRs the lower-right ROI of an in-memory BGR frame using a **singleton `tesserocr.PyTessBaseAPI`** (psm=SINGLE_LINE, whitelist `0123456789:`) and returns a parsed `datetime.time` (or `None`). The API is lazy-initialized on first call and reused for the program's lifetime — do not instantiate per call.
5. **`non_ad_generator.generate_non_ad_images`** collects every second in each video that is not within `non_ad_gap_seconds` of any ad window, then evenly samples `round(ad_count * non_ad_ratio)` of those candidates and extracts one frame per candidate to `{dataset}/result/non_ad/`. Like `processor`, it uses module-level globals to cache the open `cv2.VideoCapture` across consecutive frames from the same video. It also keeps a **`_video_meta_cache: dict[path, (fps, total_frames)]`** populated by `_get_video_meta` so each video's container header is parsed once per run instead of twice (once in `collect_non_ad_candidates`, once in `extract_non_ad_frame`).

### Post-processing pipelines (standalone, run after the generator)

Each is a single self-contained script with its own `argparse`/`load_config`, not part of the `dataset_generator_pipeline` package. They share a convention: scan `{dataset}/result`, ignore the `EXCLUDED_DIRS` sibling folders, move (never delete) rejected frames into a sibling under `result/`, and write a CSV manifest there for auditing/reversal.

- **`duplicate_images_removal_pipeline/image_similarity.py`** — for each frame computes a perceptual difference-hash (`perceptual_dhash`) and a CLIP embedding (`clip-ViT-B-32`, normalized so dot product = cosine similarity). `find_duplicate_groups` does a chunked matmul over embeddings and, for every pair scoring `>= combined_threshold` on the `clip_weight`-blended (CLIP + hash) score, unions them in a `UnionFind`. Each resulting group keeps its lexicographically-first frame; the rest move to `result/duplicates/`. The chunked matmul + union-find avoids ever materializing the full O(n²) pair list.
- **`rebalance_pipeline/rebalance.py`** — counts `result/ad` vs `result/non_ad`, `plan_trim` picks the majority class and a target count for `non_ad:ad == target_ratio` (loss-only — it never duplicates the minority), and `evenly_spaced_keep_indices` keeps frames spread across the (time-ordered) sequence rather than clustering. Surplus moves to `result/excess/`.

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
  result/
    ad/                 # generated, wiped each run
    non_ad/             # generated, wiped each run
    discarded/          # created by review_dataset.py, never wiped automatically
      ad/
      non_ad/
    duplicates/         # created by image_similarity.py; + duplicates_manifest.csv
    excess/             # created by rebalance.py; + rebalance_manifest.csv
{path.videos}/
  YYYY-MM-DD_{channel}_processed.mp4
```
