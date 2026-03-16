import pandas as pd
from typing import Any
from datetime import datetime, time, date
import os
import cv2
from src.time_calculator import get_times, compute_expected_datetime
from src.frame_verifier import find_verified_frame_position

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

    filename = f"{date_filename}_{tv_channel_filename}.mp4"
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

    video_metadata = config.get("videos_metadata").get(filename)
    video_date = get_date_from_filename(filename)
    datetime_roi = video_metadata.get("datetime_roi", None)
    datetime_format = video_metadata.get("datetime_format", "%d/%m/%Y %H:%M:%S")

    result_path = f"{config.get("path").get("dataset")}/result/ad"
    extract_frames(
        video_path=file_path,
        output_dir=result_path,
        custom_name=f'{filename.replace(".mp4", "")}_{row["cod"]}',
        custom_crop=video_metadata.get("crop"),
        times_in_seconds=times_in_seconds,
        video_date=video_date,
        video_start_time=video_start_time,
        datetime_roi=tuple(datetime_roi) if datetime_roi else None,
        datetime_format=datetime_format,
    )


def get_date_from_filename(filename: str) -> date | None:
    try:
        date_str = filename.split("_")[0]
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except (ValueError, IndexError):
        return None


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
    video_date: date = None,
    video_start_time: time = None,
    datetime_roi: tuple = None,
    datetime_format: str = "%d/%m/%Y %H:%M:%S",
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

    # Verify and correct frame position using OCR datetime
    if video_date is not None and video_start_time is not None:
        expected_dt = compute_expected_datetime(
            video_date, video_start_time, times_in_seconds[0]
        )
        verified_start = find_verified_frame_position(
            video_capture=video_capture,
            expected_datetime=expected_dt,
            initial_frame=start_frame,
            fps=fps,
            total_frames=total_frames,
            roi=datetime_roi,
            datetime_format=datetime_format,
        )
        if verified_start is not None and verified_start != start_frame:
            offset = verified_start - start_frame
            end_frame = min(end_frame + offset, total_frames)
            start_frame = verified_start

    video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_count = start_frame

    counter = 0
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
        # else:
            # Default crop params (for the moment, not sure if the video is the same)
            # height, width = frame.shape[:2]
            # frame = frame[8: height - 40, 13: width - 372]

        # Save image
        filename = f"{custom_name}_{counter}.jpg"
        cv2.imwrite(f"{output_dir}/{filename}", frame)

        counter += 1
        frame_count += fps
