import math
import pandas as pd
from typing import Any
from datetime import datetime, time
import os
import cv2
from src.time_calculator import get_times


def process_row(row: pd.Series, config: Any):
    date = row["fec"]
    tv_channel = row["can"]
    start_time = row["hin"]
    end_time = row["hfi"]

    # Check if video exists
    tv_channel_info = get_channel_info(tv_channel, config)
    date_filename = get_date_filename(date)
    if tv_channel_info is None or date_filename is None:
        print(f"Row error: incorrect tv channel or date. cid={row["cod"]}")
        return None

    file_path = f"{config.get("path")}/videos/{tv_channel_info.get("directory")}/{date_filename}_{tv_channel_info.get("filename")}.mp4"
    if not os.path.isfile(file_path):
        print(f"Row error: file doesn't exist. cid={row["cod"]}, file_path={file_path}")
        return None

    if not isinstance(start_time, time) and not isinstance(end_time, time):
        print(
            f"Row error: start time/end time are not datetime. cid={row["cod"]}, start_time={start_time}, end_time={end_time}"
        )
        return None

    result_path = f"{config.get("path")}/result/ad"
    times = get_times(start_time, end_time)
    extract_frames(file_path, result_path, times[0], times[1])


def get_channel_info(tv_channel: str, config: Any) -> str | None:
    if len(tv_channel) < 1:
        return None

    data = config.get("tv_channels_mapping").get(tv_channel.strip()[0])

    if data is not None:
        return data

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


def extract_frames(video_path, output_dir, start_time_seconds, end_time_seconds):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(start_time_seconds * fps)
    end_frame = min(
        int((end_time_seconds + 1) * fps), total_frames
    )  # + 1 to include the final frame

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_count = start_frame

    while frame_count < end_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read()
        if not ret:
            break

        filename = f"frame_{math.floor(frame_count)}.jpg"
        cv2.imwrite(f"{output_dir}/{filename}", frame)
        frame_count += fps

    cap.release()
