import pandas as pd
from typing import Any
from datetime import datetime, time
import os
import cv2
from src.time_calculator import get_times, extract_datetime

previous_video_path = ""
video_capture = None
fps = None
total_frames = None


def process_row(row: pd.Series, config: Any):
    date = row["fec"]
    tv_channel = row["can"]
    start_time = row["hin"]
    end_time = row["hfi"]

    # Check if video exists
    tv_channel_filename = get_channel_filename(tv_channel, config)
    date_filename = get_date_filename(date)
    if tv_channel_filename is None or date_filename is None:
        print(f"Row error: incorrect tv channel or date. cod={row["cod"]}")
        return None

    filename = f"{date_filename}_{tv_channel_filename}_processed.mp4"
    file_path = f"{config.get("path").get("videos")}/{filename}"
    if not os.path.isfile(file_path):
        print(
            f"Row error: file doesn't exist. cod={row["cod"]}, file_path={file_path}")
        return None

    if not isinstance(start_time, time) and not isinstance(end_time, time):
        print(
            f"Row error: start time/end time are not datetime. cod={row["cod"]}, start_time={start_time}, end_time={end_time}"
        )
        return None

    video_start_time = None
    try:
        if (config.get("videos_metadata").get(filename) is None):
            raise RuntimeError(
                f"The file doesn't have a videos_metadata entry")

        video_start_time = datetime.strptime(
            config.get("videos_metadata").get(
                filename).get("start_time"), "%H:%M:%S"
        ).time()
    except Exception as e:
        print(
            f"video start time error. filename={filename} - Original error={e}")
        return None

    times_in_seconds = get_times(
        video_start_time,
        start_time,
        end_time,
    )

    result_path = f"{config.get("path").get("dataset")}/result/ad"
    custom_name = f'{filename.replace(".mp4", "")}_{row["cod"]}'
    extract_frames(
        video_path=file_path,
        output_dir=result_path,
        custom_name=custom_name,
        custom_crop=config.get("videos_metadata").get(filename).get("crop"),
        times_in_seconds=times_in_seconds,
        ad_start_time=start_time,
        ad_end_time=end_time,
        cod=row["cod"],
    )


def get_channel_filename(tv_channel: str, config: Any) -> str | None:
    if len(tv_channel) < 1:
        return None

    filename = config.get("tv_channels_mapping").get(tv_channel.strip()[0])

    if filename is not None:
        return filename

    return None


formats = [
    "%d/%m/%Y",  # 21/5/2024
    "%Y-%m-%d %H:%M:%S",  # 2024-09-06 00:00:00
    "%Y-%m-%d",  # 2024-09-06
    "%m/%d/%Y",  # 5/21/2024
    "%d-%m-%Y",  # 21-05-2024
    "%Y/%m/%d",  # 2024/05/21
]


def get_date_filename(date_row):
    if isinstance(date_row, datetime):
        return date_row.strftime("%Y-%m-%d")

    for fmt in formats:
        try:
            date = datetime.strptime(date_row, fmt)
            return date.strftime("%Y-%m-%d")
        except ValueError:
            continue

    return None


def extract_frames(
    video_path: str,
    output_dir: str,
    custom_name: str,
    times_in_seconds: tuple[float, float],
    custom_crop,
    ad_start_time: time,
    ad_end_time: time,
    cod: Any,
):
    global previous_video_path, video_capture, fps, total_frames
    os.makedirs(output_dir, exist_ok=True)

    if previous_video_path != video_path:
        if video_capture is not None:
            video_capture.release()

        previous_video_path = video_path
        video_capture = cv2.VideoCapture(video_path)
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(times_in_seconds[0] * fps)
    end_frame = min(
        int((times_in_seconds[1] + 1) * fps), total_frames
    )  # + 1 to include the final frame
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_count = start_frame

    counter = 0
    extracted = 0
    removed = 0
    while frame_count <= end_frame:
        # Get image
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = video_capture.read()
        if not ret:
            break

        # Crop image
        height, width = frame.shape[:2]
        if custom_crop:
            frame = frame[
                custom_crop.get("top"): height - custom_crop.get("bottom"),
                custom_crop.get("left"): width - custom_crop.get("right"),
            ]

        # OCR the burned-in clock and skip frames whose timestamp falls
        # outside [ad_start_time, ad_end_time]. Frames where OCR returns
        # None (unreadable / overlay cropped out) are kept — we can't prove
        # they're out of range.
        try:
            frame_time = extract_datetime(frame)
        except Exception as e:
            print(f"OCR error (kept frame). cod={cod}, counter={counter}, error={e}")
            frame_time = None

        extracted += 1
        if frame_time is not None and (frame_time < ad_start_time or frame_time > ad_end_time):
            removed += 1
        else:
            cv2.imwrite(f"{output_dir}/{custom_name}_{counter}.jpg", frame)

        counter += 1
        frame_count += fps

    if removed > 5:
        print(
            f"Row warning: OCR filter dropped {removed} frame(s) "
            f"(considered={extracted}, written={extracted - removed}). "
            f"Likely wrong video/metadata (start_time, hin/hfi, or channel mismatch). "
            f"cod={cod}, custom_name={custom_name}"
        )
