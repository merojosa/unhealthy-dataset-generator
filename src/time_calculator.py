from datetime import time, datetime, date
from typing import Optional
import re
import cv2
import pytesseract


def get_times(video_start_time: time, ad_start_time: time, ad_end_time: time):
    dt_video_start_time = datetime.combine(date.today(), video_start_time)
    dt_start_time = datetime.combine(date.today(), ad_start_time)
    dt_end_time = datetime.combine(date.today(), ad_end_time)

    start_time_seconds = (dt_start_time - dt_video_start_time).total_seconds()
    end_time_seconds = (dt_end_time - dt_video_start_time).total_seconds()

    return (start_time_seconds, end_time_seconds)


_TIME_RE = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})")


def extract_datetime(
    image_path: str,
    roi: tuple = None
) -> Optional[time]:
    """
    OCR the clock overlay from a region of the image and return it as a time.

    Parameters:
    - image_path: Path to the image file.
    - roi: Region of interest as (x0, y0, x1, y1) ratios. Defaults to the
      lower-right corner where the timestamp overlay is burned in.

    Returns the parsed time, or None if the image can't be read or no
    HH:MM:SS pattern is found.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    height, width, _ = img.shape
    # Default ROI: lower-right corner
    if roi is None:
        roi = (0.8, 0.9, 1.0, 1.0)

    x_start = int(width * roi[0])
    y_start = int(height * roi[1])
    x_end = int(width * roi[2])
    y_end = int(height * roi[3])

    cropped = img[y_start:y_end, x_start:x_end]
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    text = pytesseract.image_to_string(gray)
    match = _TIME_RE.search(text)
    if match is None:
        return None

    hours, minutes, seconds = (int(g) for g in match.groups())
    if hours >= 24 or minutes >= 60 or seconds >= 60:
        return None

    return time(hours, minutes, seconds)
