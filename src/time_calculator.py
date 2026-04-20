from datetime import time, datetime, date
import cv2
import pytesseract


def get_times(video_start_time: time, ad_start_time: time, ad_end_time: time):
    dt_video_start_time = datetime.combine(date.today(), video_start_time)
    dt_start_time = datetime.combine(date.today(), ad_start_time)
    dt_end_time = datetime.combine(date.today(), ad_end_time)

    start_time_seconds = (dt_start_time - dt_video_start_time).total_seconds()
    end_time_seconds = (dt_end_time - dt_video_start_time).total_seconds()

    return (start_time_seconds, end_time_seconds)


def extract_datetime(
    image_path: str,
    roi: tuple = None
) -> bool:
    """
    Extracts a date from an image and checks if it's within a given range.

    Parameters:
    - image_path: Path to the image file.
    - roi: Region of interest as ratios (default is lower-right corner).

    Returns:
    - True if date is within range, False otherwise.
    """
    img = cv2.imread(image_path)
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

    datetime = pytesseract.image_to_string(gray).strip()
    return datetime
