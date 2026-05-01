from datetime import time, datetime, date
from typing import Optional
import re
import cv2
import numpy as np
import pytesseract


def get_times(video_start_time: time, ad_start_time: time, ad_end_time: time):
    dt_video_start_time = datetime.combine(date.today(), video_start_time)
    dt_start_time = datetime.combine(date.today(), ad_start_time)
    dt_end_time = datetime.combine(date.today(), ad_end_time)

    start_time_seconds = (dt_start_time - dt_video_start_time).total_seconds()
    end_time_seconds = (dt_end_time - dt_video_start_time).total_seconds()

    return (start_time_seconds, end_time_seconds)


_TIME_RE = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})")
# --psm 7: treat ROI as a single text line.
# Whitelist restricts the recognizer to digits and the colon, which is both
# faster and more accurate for a burned-in HH:MM:SS clock overlay.
_TESS_CONFIG = "--psm 7 -c tessedit_char_whitelist=0123456789:"


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

    text = pytesseract.image_to_string(gray, config=_TESS_CONFIG)
    match = _TIME_RE.search(text)
    if match is None:
        return None

    hours, minutes, seconds = (int(g) for g in match.groups())
    if hours >= 24 or minutes >= 60 or seconds >= 60:
        return None

    return time(hours, minutes, seconds)
