import pandas as pd
from typing import Any
from datetime import datetime, time
import os
import cv2
from src.time_calculator import get_times

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
    file_path = f"{config.get("path")}/videos/{filename}"
    if not os.path.isfile(file_path):
        print(f"Row error: file doesn't exist. cod={row["cod"]}, file_path={file_path}")
        return None

    if not isinstance(start_time, time) and not isinstance(end_time, time):
        print(
            f"Row error: start time/end time are not datetime. cod={row["cod"]}, start_time={start_time}, end_time={end_time}"
        )
        return None

    video_start_time = None
    try:
        video_start_time = datetime.strptime(
            config.get("videos_metadata").get(filename).get("start_time"), "%H:%M:%S"
        ).time()
    except:
        print(f"Incorrect video start time. filename={filename}")
        return None

    times = get_times(
        video_start_time,
        start_time,
        end_time,
    )

    result_path = f"{config.get("path")}/result/ad"
    extract_frames(
        video_path=file_path,
        output_dir=result_path,
        start_time_seconds=times[0],
        end_time_seconds=times[1],
        custom_name=f'{filename.replace(".mp4", "")}_{row["cod"]}',
        custom_crop=config.get("videos_metadata").get(filename).get("crop"),
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
    video_path,
    output_dir,
    start_time_seconds,
    end_time_seconds,
    custom_name,
    custom_crop,
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

    start_frame = int(start_time_seconds * fps)
    end_frame = min(
        int((end_time_seconds + 1) * fps), total_frames
    )  # + 1 to include the final frame
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
                custom_crop.get("top") : height - custom_crop.get("bottom"),
                custom_crop.get("left") : width - custom_crop.get("right"),
            ]
        else:
            # Default crop params (for the moment, not sure if the video is the same)
            height, width = frame.shape[:2]
            frame = frame[8 : height - 40, 13 : width - 372]

        # Save image
        filename = f"{custom_name}_{counter}.jpg"
        cv2.imwrite(f"{output_dir}/{filename}", frame)

        counter += 1
        frame_count += fps
