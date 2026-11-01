# Copilot instructions — unhealthy-dataset-generator

Builds a binary image dataset (junk-food `ad` vs `non_ad`) from recorded TV videos plus a `metadata.xlsx` ad log, then post-processes it. This file is the single source of agent instructions for the repo.

## Setup

Activate the venv before any Python command: `unhealthy-dataset-generator-env\Scripts\activate.bat` (Windows) or `source ./unhealthy-dataset-generator-env/bin/activate` (macOS). Then `pip install -r requirements.txt`.

OCR uses `tesserocr` (in-process libtesseract binding). The Tesseract binary is required — verify with `tesseract --version`:
- Windows: [UB Mannheim build](https://github.com/UB-Mannheim/tesseract/wiki), add its install dir to `PATH`. `tesserocr` has no PyPI wheel, so install a prebuilt one matching your Python from [simonflueckiger/tesserocr-windows_build](https://github.com/simonflueckiger/tesserocr-windows_build/releases) **before** `pip install -r requirements.txt`. `time_calculator._resolve_tessdata_path` checks `TESSDATA_PREFIX`, then `C:\Program Files\Tesseract-OCR\tessdata`.
- macOS: `brew install tesseract`.
- Linux: `sudo apt install tesseract-ocr libtesseract-dev libleptonica-dev` (dev headers needed; `tesserocr` builds from source).

The dedup pipeline needs `sentence-transformers` + `torch` (CLIP `clip-ViT-B-32` downloads on first run); the core generator does not. On Windows `pip install torch` is CPU-only — for GPU, install the CUDA wheel first (e.g. `pip install torch --index-url https://download.pytorch.org/whl/cu124`).

## Commands

**Run everything from the repo root** — `config.json`/`default_config.json` and relative dataset paths resolve against CWD.

- Generator: `python src/dataset_generator_pipeline/main.py`
  - Don't use `python -m src.dataset_generator_pipeline.main` (still shown in README). Since the module moved into its folder, `-m` puts only the CWD on `sys.path`, so the bare `from generator import …` fails with `ModuleNotFoundError`.
- Post-processing (run manually after the generator, in any order):
  - Review GUI: `python src/misc/review_dataset.py` — browses `result/ad/` by ad batch and `result/non_ad/` by page. Keys: `A` select all, `D` discard & advance, `H`/`L` or arrows navigate, `Ctrl+Z` undo.
  - Dedup: `python src/duplicate_images_removal_pipeline/image_similarity.py [folder] [--dry-run] [--combined-threshold N] [--clip-weight N] [--hash-size N] [--batch-size N] [--chunk-size N]` (defaults to `{dataset}/result`; flags override the `duplicate_images_removal` config block).
  - Rebalance: `python src/rebalance_pipeline/rebalance.py [result] [--dry-run] [--target-ratio N] [--excess-dir DIR]`
  - Multilabel CSV: `python -m src.multilabel_dataset_generator.main [dataset_root]` — pass the dataset root (not `result/`); defaults to `<repo>/dataset`.
  - Check multilabel CSV: `python -m src.multilabel_dataset_generator.check_dataset <dataset_root>/result/marketing_multilabel_dataset.csv <dataset_root>/result` — always pass both; the defaults (`<repo>/marketing_multilabel_dataset.csv`, `<repo>/dataset`) don't match the generator's output.
- Crop tuning: `python src/misc/test_custom_crop_params.py --video_path <path> --top N --bottom N --left N --right N` — shows the cropped frame and prints a `"crop": {...}` snippet for `videos_metadata.<file>.crop`.

There is no test suite, linter, or formatter. Validate changes by running the affected script (`--dry-run` for dedup/rebalance).

## Architecture

**Main generator** (`src/dataset_generator_pipeline/`): `main.py` → `generator.py` → `processor.py` → `time_calculator.py`, then `non_ad_generator.py` after all ad rows.

1. `generator.generate_dataset` reads `{dataset}/metadata.xlsx` (first sheet), **deletes `{dataset}/result/`**, dispatches matching rows to `process_row`, sums the ad-frame counts it returns, and passes the total to `non_ad_generator.generate_non_ad_images`. (The running total replaced a slow `glob` rescan of `result/ad/`.)
2. `processor.process_row` holds the per-row business logic: resolves the video from `fec` + `can`, validates the file and its `videos_metadata` entry, computes ad offsets via `time_calculator.get_times`, calls `extract_frames`, and returns frames written (0 on failure). Row errors are logged and skipped — one bad row never aborts the run.
3. `processor.extract_frames` is two-phase:
   - **Boundary OCR**: OCR the burned-in clock on the first and last sampled frames. If both fall inside `[ad_start, ad_end]`, the seek math is trusted and Phase 2 skips OCR (fast path). Otherwise every frame is OCR'd and out-of-window frames dropped (`None` = keep).
   - **Sequential extraction**: one `cap.set(POS_FRAMES)` to the window start, then `cap.grab() × (fps-1)` + `cap.read()` per second. Don't switch to per-second `set()` — it forces a keyframe seek + decode every iteration.
   - Returns `extracted - removed`. The counter increments for every considered frame, so dropped frames leave numbering gaps.
4. `time_calculator.extract_datetime` OCRs the lower-right ROI of an in-memory BGR frame using a lazily-initialized **singleton** `tesserocr.PyTessBaseAPI` (`_get_api()`, psm SINGLE_LINE, whitelist `0123456789:`) and returns a `datetime.time` or `None`. Never instantiate per call — `pytesseract` was removed because spawning `tesseract.exe` per call dominated runtime.
5. `non_ad_generator` collects every second not within `non_ad_gap_seconds` of any ad window, evenly samples `round(ad_count * non_ad_ratio)` candidates, and writes one frame each to `result/non_ad/`. `_video_meta_cache` (`path → (fps, total_frames)`) avoids parsing each container header twice.

**Post-processing scripts** are standalone (own `argparse` + `load_config`), not part of the generator package. They operate on `{dataset}/result`, only **move** files (never delete) into sibling folders, and write CSV manifests, so they're reversible and idempotent:
- `review_dataset.py` → `result/discarded/{ad,non_ad}/`.
- `image_similarity.py` → `result/duplicates/` (relative paths preserved) + `duplicates_manifest.csv`. Computes a perceptual dHash and a normalized CLIP embedding per frame, blends them by `clip_weight`, and runs a chunked matmul + `UnionFind` over pairs scoring ≥ `combined_threshold` (never materializes all O(n²) pairs). Keeps the lexicographically-first frame per group.
- `rebalance.py` → `result/excess/` + `rebalance_manifest.csv`. `plan_trim` trims only the majority class (never duplicates) to hit `non_ad:ad == target_ratio`; `evenly_spaced_keep_indices` keeps survivors spread across the time-ordered sequence.
- `multilabel_dataset_generator/` — `main.py` checks the dataset root has `metadata.xlsx` and `result/{ad,non_ad}/`. Then `generator.generate_csv` reads `result/ad/*.jpg`, extracts the 5-digit `cod` via `_(\d{5})_\d+\.jpg$`, and joins it with `metadata.xlsx` (sheet `Base combinada`, else the first sheet). It writes `image_name` + every column from `rpr` through `rie` (inclusive, by position) to `result/marketing_multilabel_dataset.csv`. Non-matching files are skipped and images are never moved. `check_dataset.py` reports images missing from disk or from the CSV, duplicate rows, and absent `rpr`/`rie` columns.

`EXCLUDED_DIRS` differs per script: `image_similarity.py` skips `discarded`/`duplicates`; `rebalance.py` also skips `excess`. Add any new sibling output folder under `result/` to each set.

## Conventions

- **Config schema**: `default_config.json` is the source of truth. `main.validate_config` recursively requires every default key in `config.json` with the same Python type, so add any new key there first with a realistic default. All scripts load `config.json` from CWD, falling back to `default_config.json`.
- **Bare imports** inside `src/dataset_generator_pipeline/` (`from processor import process_row`). They rely on the script's own directory being on `sys.path`. Don't convert to relative imports without testing the entrypoint.
- **Module-level `cv2.VideoCapture` cache** (`video_capture`, `fps`, `total_frames`, `previous_video_path` in `processor.py` and `non_ad_generator.py`) reuses the open video across consecutive rows. Keep an equivalent cache if refactoring, or throughput collapses on large metadata files.
- **Output filenames** `{video}_{cod}_{counter}.jpg` are parsed downstream (multilabel `cod` regex, review GUI grouping by ad batch) — keep the format stable.

### Config keys
- `path.dataset`: root containing `metadata.xlsx`; output goes to `{dataset}/result/`. `path.videos`: folder of `.mp4` files.
- `tip_values`: ints; a row is processed only if `tip` *starts with* `"{n}="` (e.g. `[2,3]` matches `"2=foo"`, not `"1=foo"`).
- `tv_channels_mapping`: first char of `can` → filename channel (`"1" → "DN"` reads `{date}_DN_processed.mp4`).
- `videos_metadata[filename]`: `start_time` (`HH:MM:SS`, recording wall-clock start used to convert `hin`/`hfi` into offsets) and optional `crop` (`top`/`bottom`/`left`/`right` px).
- `non_ad_ratio` (float, 1.0): non-ad frames as a multiple of ad frames. `non_ad_gap_seconds` (int, 30): exclusion margin around each ad window.
- `duplicate_images_removal`: `hash_size`, `combined_threshold`, `clip_weight` (0–1, CLIP vs hash), `batch_size`, `chunk_size`.
- `rebalance.target_ratio` (float, 1.0): desired `non_ad:ad`.

### Metadata columns (`metadata.xlsx`)
`tip` (type filter), `can` (channel code), `fec` (date; `datetime` or a format in `processor.formats`), `hin`/`hfi` (ad start/end `datetime.time`), `cod` (row id embedded in filenames), `rpr`…`rie` (multilabel marketing labels).

### On-disk layout
```
{path.dataset}/
  metadata.xlsx
  result/                           # deleted and regenerated by each generator run
    ad/  non_ad/
    discarded/{ad,non_ad}/          # review_dataset.py
    duplicates/ + duplicates_manifest.csv   # image_similarity.py
    excess/ + rebalance_manifest.csv        # rebalance.py
    marketing_multilabel_dataset.csv        # multilabel_dataset_generator
{path.videos}/
  YYYY-MM-DD_{channel}_processed.mp4
```
