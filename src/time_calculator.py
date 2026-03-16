from datetime import time, datetime, date, timedelta
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


def extract_datetime_from_frame(
    frame: np.ndarray,
    roi: tuple = None,
    datetime_format: str = "%d/%m/%Y %H:%M:%S"
) -> datetime | None:
    if frame is None:
        return None

    if roi is None:
        roi = (0.8, 0.9, 1.0, 1.0)

    height, width = frame.shape[:2]
    x_start = int(width * roi[0])
    y_start = int(height * roi[1])
    x_end = int(width * roi[2])
    y_end = int(height * roi[3])

    cropped = frame[y_start:y_end, x_start:x_end]
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    ocr_text = pytesseract.image_to_string(thresh, config="--psm 7").strip()

    # Try to extract date+time pattern from OCR text
    # Match patterns like "02/04/2024 10:30:45" or similar
    date_time_match = re.search(r'(\d{1,4}[/\-\.]\d{1,2}[/\-\.]\d{1,4}\s+\d{1,2}:\d{2}:\d{2})', ocr_text)
    if not date_time_match:
        return None

    try:
        return datetime.strptime(date_time_match.group(1), datetime_format)
    except ValueError:
        return None


def extract_datetime(
    image_path: str,
    roi: tuple = None,
    datetime_format: str = "%d/%m/%Y %H:%M:%S"
) -> datetime | None:
    img = cv2.imread(image_path)
    if img is None:
        return None
    return extract_datetime_from_frame(img, roi, datetime_format)


def compute_expected_datetime(
    video_date: date,
    video_start_time: time,
    offset_seconds: float
) -> datetime:
    base = datetime.combine(video_date, video_start_time)
    return base + timedelta(seconds=offset_seconds)
