# Unhealthy Dataset Generator

A pipeline to generate a binary dataset with the following two classes:

- Junk food ad
- Non junk food ad

The pipeline iterates through a metadata sheet, takes the start and end times, extracts one frame per second, and saves it in an "ad" directory. For the "non_ad" directory, it takes frames outside the start and end times of the metadata.

## Get started

- Initiate a [virtual environment](https://www.freecodecamp.org/news/how-to-setup-virtual-environments-in-python/):

```
python -m venv unhealthy-dataset-generator-env
```

Windows:

```
unhealthy-dataset-generator-env\Scripts\activate.bat
```

MacOS:

```
source ./unhealthy-dataset-generator-env/bin/activate
```

- Install the Tesseract OCR binary. The pipeline uses it (via the in-process `tesserocr` binding) to read the timestamp overlay on each extracted frame and discard frames whose on-screen time falls outside the ad window.
  - Windows: install from the [UB Mannheim build](https://github.com/UB-Mannheim/tesseract/wiki). Add the install directory (e.g. `C:\Program Files\Tesseract-OCR`) to `PATH`. The pipeline also looks for `tessdata` at `C:\Program Files\Tesseract-OCR\tessdata` (the default install location); if you installed elsewhere, set the `TESSDATA_PREFIX` environment variable to your `tessdata` directory.
  - macOS: `brew install tesseract`
  - Linux: `sudo apt install tesseract-ocr libtesseract-dev libleptonica-dev` (or the equivalent for your distro — `tesserocr` needs the dev headers to install).
  - Verify with `tesseract --version`.

- Install Python dependencies: `pip install -r requirements.txt`

  On **Windows**, `tesserocr` does not have a wheel on PyPI and needs the Tesseract dev libraries to build from source, which the UB Mannheim binary install doesn't ship. Use the prebuilt wheel matching your Python version from [simonflueckiger/tesserocr-windows_build](https://github.com/simonflueckiger/tesserocr-windows_build/releases) **before** running `pip install -r requirements.txt`. For Python 3.14:

  ```
  pip install https://github.com/simonflueckiger/tesserocr-windows_build/releases/download/tesserocr-v2.10.0-tesseract-5.5.2/tesserocr-2.10.0-cp314-cp314-win_amd64.whl
  pip install -r requirements.txt
  ```

  On **macOS / Linux**, `pip install -r requirements.txt` builds `tesserocr` from source against the Tesseract dev headers installed in the previous step.

- Execute the script: `python -m src.dataset_generator_pipeline.main` (run from the repo root — paths in the config are resolved relative to the current working directory)

## Instructions

- The entire project works with a config.json. Check `default_config.json` to understand the structure.
  - `path` is where the script will read the `metadata.xlsx` and videos. It will look something like this:
    ![alt text](assets/path_structure.png)
  - `tip_values`: list of integers. A metadata row is processed only if its `tip` column value *starts with* `"N="` for some N in this list — e.g. `[2,3,4,5,6,7,8]` skips any row whose `tip` starts with `"1="`.
  - `non_ad_ratio`: float (default `1.0`). Number of non-ad frames to generate as a multiple of the total ad frame count — e.g. `1.0` produces one non-ad frame per ad frame, `2.0` produces twice as many.
  - `non_ad_gap_seconds`: int (default `30`). Seconds before and after each ad window that are also excluded when selecting non-ad frame candidates, to avoid borderline content.
  - `tv_channels_mapping`: maps the first character of the `can` column to a filename suffix. For example, `"1" → "DN"` means a row with `can` starting with `1` reads `{date}_DN_processed.mp4`. Add one entry per channel.
  - `videos_metadata`: one entry per video file. `start_time` (`HH:MM:SS`) is the wall-clock time the recording begins — the pipeline subtracts this from the ad's `hin`/`hfi` times to find the frame offset inside the video. `crop` (`top`/`bottom`/`left`/`right`, in pixels) trims each extracted frame; it is optional and can be omitted if no crop is needed (use `test_custom_crop_params.py` to find the right values).
- To check a particular video, you should populate `metadata.xlsx` with the data related to the video. For example, if you want to test `2024-04-06_DN.mp4`, you should filter "can" to 1 and "fec" to 06-04-24.
- The output of the script will be on `path/result`. Every image has the following structure: video name where the it was extracted + id from `metadata.xlsx`("cod" column) + counter id + .jpg.

## Expected directory layout

```
{path.dataset}/
  metadata.xlsx
  result/              # created by the pipeline, wiped on each run
    ad/
    non_ad/
    discarded/         # created by review_dataset.py, never wiped automatically
      ad/
      non_ad/
{path.videos}/
  YYYY-MM-DD_{channel}_processed.mp4
```

Output filenames follow the pattern `{video}_{cod}_{counter}.jpg`, where `cod` is the row ID from the `cod` column of `metadata.xlsx`.

## Utility scripts

### Tuning crop parameters (`src/misc/test_custom_crop_params.py`)

Helps you find the right crop values for `videos_metadata.<filename>.crop` in `config.json`. Pass a video path and crop amounts (in pixels) for each side — it opens a random frame from the video with that crop applied so you can judge the result visually. Once the frame looks right, copy the printed `"crop": {...}` snippet into your config.

```
python src/misc/test_custom_crop_params.py --video_path <path> --top <n> --bottom <n> --left <n> --right <n>
```

### Reviewing the generated dataset (`src/misc/review_dataset.py`)

A desktop GUI for manually inspecting and cleaning the images produced by the pipeline. Run it from the repo root after the pipeline has finished:

```
python src/misc/review_dataset.py
```

It lets you browse ad frames (grouped by ad) and non-ad frames, select bad images, and move them to a `result/discarded/` folder. Discarded images are not deleted and can be restored.
