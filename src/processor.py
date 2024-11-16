import pandas as pd
from typing import Any
from datetime import datetime
import os


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

    filename = f"{tv_channel_info.get("directory")}/{date_filename}_{tv_channel_info.get("filename")}.mp4"
    if not os.path.isfile(f"{config.get("path")}/videos/{filename}"):
        print(f"Row error: file doesn't exist. cid={row["cod"]}, filename={filename}")
        return None

    # Extract images based on start time and end time (one frame per second?)
