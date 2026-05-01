import os
from datetime import datetime, time
from typing import Any

import cv2
import pandas as pd

from src.processor import get_channel_filename, get_date_filename
from src.time_calculator import get_times

_previous_video_path = ""
_video_capture = None
_fps = None
_total_frames = None


def collect_non_ad_candidates(
    df: pd.DataFrame, config: Any
) -> list[tuple[str, int, str, str]]:
    groups: dict[tuple[str, str], list] = {}
    for _, row in df.iterrows():
        date_str = get_date_filename(row["fec"])
        channel_str = get_channel_filename(row["can"], config)
        if date_str is None or channel_str is None:
            continue
        if not isinstance(row["hin"], time) or not isinstance(row["hfi"], time):
            continue
        key = (date_str, channel_str)
        groups.setdefault(key, []).append(row)

    gap = config.get("non_ad_gap_seconds", 30)
    candidates: list[tuple[str, int, str, str]] = []

    for (date_str, channel_str), rows in sorted(groups.items()):
        filename = f"{date_str}_{channel_str}_processed.mp4"
        file_path = f"{config['path']['videos']}/{filename}"
        if not os.path.isfile(file_path):
            continue
        if config.get("videos_metadata", {}).get(filename) is None:
            continue

        try:
            video_start_time = datetime.strptime(
                config["videos_metadata"][filename]["start_time"], "%H:%M:%S"
            ).time()
        except Exception:
            continue

        ad_seconds: set[int] = set()
        for row in rows:
            start_s, end_s = get_times(video_start_time, row["hin"], row["hfi"])
            lo = max(0, int(start_s) - gap)
            hi = int(end_s) + gap
            for s in range(lo, hi + 1):
                ad_seconds.add(s)

        cap = cv2.VideoCapture(file_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if fps <= 0:
            continue
        video_duration_s = int(total_frames / fps)

        for s in range(0, video_duration_s):
            if s not in ad_seconds:
                candidates.append((file_path, s, date_str, channel_str))

    return candidates


def select_candidates(candidates: list[tuple], target_count: int) -> list[tuple]:
    if target_count <= 0 or not candidates:
        return []
    if target_count >= len(candidates):
        return list(candidates)
    step = len(candidates) / target_count
    return [candidates[int(i * step)] for i in range(target_count)]


def extract_non_ad_frame(
    video_path: str,
    second: int,
    custom_crop,
    output_dir: str,
    filename: str,
) -> bool:
    global _previous_video_path, _video_capture, _fps, _total_frames

    if video_path != _previous_video_path:
        if _video_capture is not None:
            _video_capture.release()
        _previous_video_path = video_path
        _video_capture = cv2.VideoCapture(video_path)
        _fps = _video_capture.get(cv2.CAP_PROP_FPS)
        _total_frames = int(_video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_index = int(second * _fps)
    if frame_index >= _total_frames:
        return False

    _video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = _video_capture.read()
    if not ret:
        return False

    height, width = frame.shape[:2]
    if custom_crop:
        frame = frame[
            custom_crop.get("top"): height - custom_crop.get("bottom"),
            custom_crop.get("left"): width - custom_crop.get("right"),
        ]

    cv2.imwrite(f"{output_dir}/{filename}", frame)
    return True


def generate_non_ad_images(df: pd.DataFrame, config: Any, ad_count: int):
    target_count = round(ad_count * config.get("non_ad_ratio", 1.0))
    candidates = collect_non_ad_candidates(df, config)
    selected = select_candidates(candidates, target_count)

    output_dir = f"{config['path']['dataset']}/result/non_ad"
    os.makedirs(output_dir, exist_ok=True)

    counter = 0
    for video_path, second, date_str, channel_str in selected:
        filename_key = f"{date_str}_{channel_str}_processed.mp4"
        custom_crop = (config["videos_metadata"].get(filename_key) or {}).get("crop")
        out_filename = f"{date_str}_{channel_str}_non_ad_{counter}.jpg"
        if extract_non_ad_frame(video_path, second, custom_crop, output_dir, out_filename):
            counter += 1
