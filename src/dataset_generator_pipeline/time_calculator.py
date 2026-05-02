from datetime import time, datetime, date
from typing import Optional
import os
import re
import cv2
import numpy as np
import tesserocr


def get_times(video_start_time: time, ad_start_time: time, ad_end_time: time):
    dt_video_start_time = datetime.combine(date.today(), video_start_time)
    dt_start_time = datetime.combine(date.today(), ad_start_time)
    dt_end_time = datetime.combine(date.today(), ad_end_time)

    start_time_seconds = (dt_start_time - dt_video_start_time).total_seconds()
    end_time_seconds = (dt_end_time - dt_video_start_time).total_seconds()

    return (start_time_seconds, end_time_seconds)


_TIME_RE = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})")

# We hold one in-process tesseract API for the lifetime of the program.
# pytesseract spawned `tesseract.exe` per call (~140ms each on Windows due
# to process creation + stdout pipe drain); tesserocr keeps libtesseract
# loaded and reuses the same engine state, so each OCR is just the
# recognition work itself.
_tess_api: Optional[tesserocr.PyTessBaseAPI] = None


def _resolve_tessdata_path() -> str:
    """Find the tessdata directory. tesserocr needs an explicit path on
    Windows because the UB Mannheim installer doesn't set TESSDATA_PREFIX."""
    explicit = os.environ.get("TESSDATA_PREFIX")
    if explicit and os.path.isdir(explicit):
        return explicit
    default = r"C:\Program Files\Tesseract-OCR\tessdata"
    if os.path.isdir(default):
        return default
    # Let tesserocr search its own defaults; will raise if it can't find any.
    return ""


def _get_api() -> tesserocr.PyTessBaseAPI:
    """Lazy-init a singleton API. Initialization is ~50 ms (loads the LSTM
    model into memory), which we'd otherwise pay on every OCR call."""
    global _tess_api
    if _tess_api is None:
        api = tesserocr.PyTessBaseAPI(
            path=_resolve_tessdata_path(),
            lang="eng",
            # SINGLE_LINE == --psm 7. Treat the ROI as a single text line.
            psm=tesserocr.PSM.SINGLE_LINE,
        )
        # Whitelist restricts the recognizer to characters that can appear
        # in HH:MM:SS, both faster and more accurate for the clock overlay.
        api.SetVariable("tessedit_char_whitelist", "0123456789:")
        _tess_api = api
    return _tess_api


def extract_datetime(
    frame: np.ndarray,
    roi: tuple = None,
) -> Optional[time]:
    """OCR the clock overlay from an in-memory BGR frame and return it as a time.

    Parameters:
    - frame: BGR image as returned by cv2.VideoCapture.read / cv2.imread.
    - roi: Region of interest as (x0, y0, x1, y1) ratios. Defaults to the
      lower-right corner where the timestamp overlay is burned in.

    Returns the parsed time, or None if no HH:MM:SS pattern is found.
    """
    if frame is None:
        return None

    height, width = frame.shape[:2]
    if roi is None:
        roi = (0.8, 0.9, 1.0, 1.0)

    x_start = int(width * roi[0])
    y_start = int(height * roi[1])
    x_end = int(width * roi[2])
    y_end = int(height * roi[3])

    cropped = frame[y_start:y_end, x_start:x_end]
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    # tesserocr accepts raw bytes for a grayscale image: 1 byte per pixel,
    # row stride = width. Avoids a PIL.Image roundtrip.
    h, w = gray.shape
    gray = np.ascontiguousarray(gray)

    api = _get_api()
    api.SetImageBytes(gray.tobytes(), w, h, 1, w)
    text = api.GetUTF8Text()

    match = _TIME_RE.search(text)
    if match is None:
        return None

    hours, minutes, seconds = (int(g) for g in match.groups())
    if hours >= 24 or minutes >= 60 or seconds >= 60:
        return None

    return time(hours, minutes, seconds)
