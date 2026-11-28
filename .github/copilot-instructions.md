# Copilot instructions — unhealthy-dataset-generator

Builds a binary image dataset (junk-food `ad` vs `non_ad`) from recorded TV videos plus a `metadata.xlsx` ad log, then post-processes it. This file is the single source of agent instructions for the repo.

## Setup

Activate the venv before any Python command: `unhealthy-dataset-generator-env\Scripts\activate.bat` (Windows) or `source ./unhealthy-dataset-generator-env/bin/activate` (macOS). Then `pip install -r requirements.txt`.

OCR uses `tesserocr` (in-process libtesseract binding). The Tesseract binary is required — verify with `tesseract --version`:
- Windows: [UB Mannheim build](https://github.com/UB-Mannheim/tesseract/wiki), add its install dir to `PATH`. `tesserocr` has no PyPI wheel, so install a prebuilt one matching your Python from [simonflueckiger/tesserocr-windows_build](https://github.com/simonflueckiger/tesserocr-windows_build/releases) **before** `pip install -r requirements.txt`. `time_calculator._resolve_tessdata_path` checks `TESSDATA_PREFIX`, then `C:\Program Files\Tesseract-OCR\tessdata`.
- macOS: `brew install tesseract`.
- Linux: `sudo apt install tesseract-ocr libtesseract-dev libleptonica-dev` (dev headers needed; `tesserocr` builds from source).

Dedup and grouping need `sentence-transformers` + `torch` (dedup downloads CLIP `clip-ViT-B-32`; grouping downloads DINOv2 `facebook/dinov2-base` via `transformers` into `grouping.model_cache_dir` = `./models`, gitignored; first run needs internet); the core generator does not. torchvision isn't installed, so `group_frames.embed_frames` preprocesses manually (392x224 resize, ImageNet norm) instead of `AutoImageProcessor`.

Copy `default_config.json` to `config.json` and set `path.videos`, `path.dataset`, and a `videos_metadata` entry per video. To test a single video, filter `metadata.xlsx` to its `can` + `fec`. On Windows `pip install torch` is CPU-only — for GPU, install the CUDA wheel first (e.g. `pip install torch --index-url https://download.pytorch.org/whl/cu124`).

## Commands

**Run everything from the repo root** — `config.json`/`default_config.json` and relative dataset paths resolve against CWD.

- Generator: `python src/dataset_generator_pipeline/main.py`
  - Don't use `python -m src.dataset_generator_pipeline.main`. Since the module moved into its folder, `-m` puts only the CWD on `sys.path`, so the bare `from generator import …` fails with `ModuleNotFoundError`.
- Post-processing (run manually after the generator, in the README order: review → dedup → group → rebalance → multiclass/multilabel):
  - Review GUI: `python src/misc/review_dataset.py` — browses `result/ad/` by ad batch and `result/non_ad/` by page. Keys: `A` select all, `D` discard & advance, `H`/`L` or arrows navigate, `Ctrl+Z` undo.
  - Dedup: `python src/duplicate_images_removal_pipeline/image_similarity.py [folder] [--dry-run] [--combined-threshold N] [--clip-weight N] [--hash-size N] [--batch-size N] [--chunk-size N]` (defaults to `{dataset}/result`; flags override the `duplicate_images_removal` config block).
  - Rebalance: `python src/rebalance_pipeline/rebalance.py [result] [--dry-run] [--target-ratio N] [--excess-dir DIR]`
  - Leakage grouping + per-creative cap: `python src/grouping_pipeline/group_frames.py [result] [--dry-run] [--max-frames N] [--centroid-threshold N] [--frame-threshold N] [--non-ad-group-by {channel,video}] [--batch-size N]` — run after dedup/review, before rebalance and the multiclass generator.
  - Multilabel dataset: `python -m src.marketing_strategies_multilabel_dataset_generator.main [binary_root] [--output DIR]` — input defaults to `path.dataset` (pass the root, not `result/`), output to `path.multilabel_dataset`.
  - Check multilabel dataset: `python -m src.marketing_strategies_multilabel_dataset_generator.check_dataset [multilabel_root]` — defaults to `path.multilabel_dataset`. Train-only augmentation: `python -m src.marketing_strategies_multilabel_dataset_generator.augment [multilabel_root] [--dry-run] [--min-train-share N] [--target-train-share N] [--max-multiplier N] [--seed N]`.
  - Tip multiclass dataset: `python -m src.tip_multiclass_dataset_generator.main [binary_root] [--output DIR] [--non-ad-ratio N]` (output defaults to `path.multiclass_dataset`); check with `python -m src.tip_multiclass_dataset_generator.check_dataset [multiclass_root]`. Train-only augmentation: `python -m src.tip_multiclass_dataset_generator.augment [multiclass_root] [--dry-run] [--target-ratio N] [--max-multiplier N] [--seed N]`.
- Zips: `python src/misc/make_zips.py` — writes uncompressed `binary-dataset.zip` (`ad/`, `non_ad/`, `groups.csv`), `tip-multiclass-dataset.zip` and `multi-strategies-dataset.zip` (`images/`, CSV, plus `augmented/` + augmented CSV if present) to the repo root.
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
- `grouping_pipeline/group_frames.py` → `result/groups.csv` (`image_name`, `label`, `group`, `creative`, `capped`) + `result/capped/ad/` + `capped_manifest.csv`. Rows ≠ ads: one creative airs under many `cod`s, so ad groups key on normalized `nom` (via `cod`), then `merge_creatives` unions creatives (DINOv2 `pooler_output` embeddings) whose centroid cosine ≥ `centroid_threshold`, or whose best frame match ≥ `frame_threshold` AND centroid ≥ `frame_match_min_centroid` (the centroid floor stops a black frame/logo from chaining unrelated creatives). Same-brand spots that share no shots stay separate by design. Non-ad group = channel (`non_ad_{channel}`, date prefix stripped) or source video (`non_ad_{video}`) per `non_ad_group_by`; filenames carry a counter, not the second. Ad groups over `max_frames_per_creative` keep `evenly_spaced_keep_indices` survivors in airing order (`cod`, counter). Already-capped frames still count toward their group, so re-runs are stable.
- `marketing_strategies_multilabel_dataset_generator/` — `main.py` checks the binary root has `metadata.xlsx` and `result/ad/`, and refuses an output equal to the input. Then `generator.generate_dataset` reads `result/ad/*.jpg`, extracts the 5-digit `cod` via `_(\d{5})_\d+\.jpg$`, and joins it with `metadata.xlsx` (sheet `Base combinada`, else the first sheet). It wipes and refills `{multilabel_dataset}/images/` with **copies** of matched frames, skipping frames whose metadata row has every `rpr`–`rie` cell blank (unannotated non-food ads, e.g. tip 6), and writes `image_name` + every column from `rpr` through `rie` (inclusive, by position) + `group` (from the binary `result/groups.csv`) + `split` to `{multilabel_dataset}/marketing_multilabel_dataset.csv`. `split.multilabel_group_split` assigns whole groups 70/15/15, balancing each label's positives (label cells are text/numbers with blanks and NBSP; `split.positives` treats only numeric 1 as positive). `augment.py` (optional, train only) runs only if the real train share < `min_train_share`, then adds copies until train = `target_train_share`, drawing sources by rare-label weight and reusing the multiclass `augment_image`; output `augmented/` + `marketing_multilabel_augmented.csv`. Non-matching files are skipped; the binary dataset is never modified. `check_dataset.py` reports images missing from `images/` or from the CSV, duplicate rows, absent `rpr`/`rie` columns, and fails on a missing/empty `group`, a group in several splits, or a bad augmented set.
- `tip_multiclass_dataset_generator/` — same structure and input checks as the multilabel package, but the single class per frame is the integer prefix of `tip` (`"2=..."` → `2`), keeping only frames whose tip is in `tip_values` (so `tip=1` is excluded). It also adds a synthetic class `0` (`NON_AD_TIP`) from `result/non_ad/`, sized `round(multiclass.non_ad_ratio × largest ad class)` and evenly sampled in time order (`--non-ad-ratio` overrides; `0` disables). It writes `images/` + `tip_multiclass_dataset.csv` (`image_name`, `tip`, `tip_label`, `group`, `split`) under `path.multiclass_dataset`; `group` comes from the binary `result/groups.csv`, and `split` (train/val/test) from `split.stratified_group_split` — a verbatim port of the junk-food-ads-classifiers-v2 notebook function (seed 42, 70/15/15); keep both copies in sync. `augment.py` (optional) writes train-only augmented copies to `augmented/` + `tip_multiclass_augmented.csv` (main columns + `source_image`), topping each train class up to `target_ratio × largest train class`, capped at `max_multiplier × original`; no flips, small hue shift (logos/brand colours carry signal); it never touches val/test, `images/` or the main CSV. `check_dataset.py` also prints the class distribution and frames/groups per class and per split, notes groups spanning several tips, and fails on empty `group`, a group in several splits, or augmented rows that are not train or whose source is not train.

`EXCLUDED_DIRS` differs per script: `image_similarity.py` skips `discarded`/`duplicates`/`capped`; `rebalance.py` also skips `excess`. Add any new sibling output folder under `result/` to each set.

## Conventions

- **Config schema**: `default_config.json` is the source of truth. `main.validate_config` recursively requires every default key in `config.json` with the same Python type, so add any new key there first with a realistic default. All scripts load `config.json` from CWD, falling back to `default_config.json`.
- **Bare imports** inside `src/dataset_generator_pipeline/` (`from processor import process_row`). They rely on the script's own directory being on `sys.path`. Don't convert to relative imports without testing the entrypoint.
- **Module-level `cv2.VideoCapture` cache** (`video_capture`, `fps`, `total_frames`, `previous_video_path` in `processor.py` and `non_ad_generator.py`) reuses the open video across consecutive rows. Keep an equivalent cache if refactoring, or throughput collapses on large metadata files.
- **Output filenames** are parsed downstream — keep the formats stable:
  - Ad: `{video}_{cod}_{counter}.jpg`; `cod` must be 5 digits or downstream regexes (`_(\d{5})_(\d+)\.jpg$` in multilabel/multiclass/grouping) skip the frame. The review GUI groups by ad batch from this name.
  - Non-ad: `{date}_{channel}_non_ad_{counter}.jpg`; `NON_AD_PATTERN` (`^(.*)_non_ad_(\d+)\.jpg$`) derives the non-ad group (`non_ad_{channel}` or `non_ad_{video}`).
- **Cross-script imports**: `group_frames.py` inserts `src/` into `sys.path` and imports `evenly_spaced_keep_indices` from `rebalance_pipeline.rebalance` plus `load_image`/`UnionFind` from `duplicate_images_removal_pipeline.image_similarity`. Keep those names/signatures stable when editing either script.
- **Excel sheet selection**: grouping and both dataset generators read sheet `Base combinada` if present, else the first sheet (`ValueError` fallback); the main generator always reads the first sheet.
- **Leakage-safe splits**: train/test splits must be by `group` (e.g. `GroupShuffleSplit`/`StratifiedGroupKFold`). The multiclass and multilabel CSVs both carry `group` (from `result/groups.csv`).

### Config keys
- `path.dataset` (`./binary-dataset`): root containing `metadata.xlsx`; binary output goes to `{dataset}/result/`. `path.multilabel_dataset` (`./multi-strategies-dataset`): multilabel output root. `path.multiclass_dataset` (`./tip-multiclass-dataset`): tip multiclass output root. `path.videos`: folder of `.mp4` files.
- `tip_values`: ints; a row is processed only if `tip` *starts with* `"{n}="` (e.g. `[2,3]` matches `"2=foo"`, not `"1=foo"`).
- `tv_channels_mapping`: first char of `can` → filename channel (`"1" → "DN"` reads `{date}_DN_processed.mp4`).
- `videos_metadata[filename]`: `start_time` (`HH:MM:SS`, recording wall-clock start used to convert `hin`/`hfi` into offsets) and optional `crop` (`top`/`bottom`/`left`/`right` px).
- `non_ad_ratio` (float, 1.0): non-ad frames as a multiple of ad frames. `non_ad_gap_seconds` (int, 30): exclusion margin around each ad window.
- `duplicate_images_removal`: `hash_size`, `combined_threshold`, `clip_weight` (0–1, CLIP vs hash), `batch_size`, `chunk_size`.
- `rebalance.target_ratio` (float, 1.0): desired `non_ad:ad`.
- `multiclass.non_ad_ratio` (float, 0.5): size of the tip-multiclass non-ad class (`tip 0`) as a multiple of the largest ad class.
- `multiclass.augmentation`: `target_ratio` (float, 0.5; per-class train target vs the largest train class), `max_multiplier` (float, 5.0; cap vs the class's original train frames), `seed` (int, 42).
- `multilabel.augmentation`: `min_train_share` (float, 0.70; augment only below this), `target_train_share` (float, 0.75), `max_multiplier` (float, 5.0; max uses per source frame), `seed` (int, 42).
- `grouping`: `max_frames_per_creative` (int, 50; 0 = no cap), `model` (`facebook/dinov2-base`), `model_cache_dir` (`./models`), `centroid_threshold` (0.85), `frame_threshold` (0.90), `frame_match_min_centroid` (0.45 — calibrated: real same-brand pairs had best frame ≥0.90 with centroid ≥0.45; spurious logo/black-frame matches had lower centroids), `non_ad_group_by` (`channel`), `batch_size`.

### Metadata columns (`metadata.xlsx`)
`tip` (type filter), `can` (channel code), `fec` (date; `datetime` or a format in `processor.formats`), `hin`/`hfi` (ad start/end `datetime.time`), `cod` (row id embedded in filenames), `nom` (ad name, used for leakage grouping), `rpr`…`rie` (multilabel marketing labels).

### On-disk layout
```
{path.dataset}/
  metadata.xlsx
  result/                           # deleted and regenerated by each generator run
    ad/  non_ad/
    discarded/{ad,non_ad}/          # review_dataset.py
    duplicates/ + duplicates_manifest.csv   # image_similarity.py
    excess/ + rebalance_manifest.csv        # rebalance.py
    capped/ad/ + capped_manifest.csv        # group_frames.py
    groups.csv                              # group_frames.py
{path.multilabel_dataset}/                  # marketing_strategies_multilabel_dataset_generator
  images/                                   # copies of labelled ad frames, wiped each run
  marketing_multilabel_dataset.csv
  augmented/ + marketing_multilabel_augmented.csv  # augment.py, train split only, wiped each run
{path.multiclass_dataset}/                  # tip_multiclass_dataset_generator
  images/                                   # copies of labelled ad frames, wiped each run
  tip_multiclass_dataset.csv
  augmented/ + tip_multiclass_augmented.csv # augment.py, train split only, wiped each run
{path.videos}/
  YYYY-MM-DD_{channel}_processed.mp4
```
