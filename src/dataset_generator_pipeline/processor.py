import pandas as pd
from typing import Any
from datetime import datetime, time
import os
import cv2
from time_calculator import get_times, extract_datetime

previous_video_path = ""
video_capture = None
fps = None
total_frames = None


def process_row(row: pd.Series, config: Any) -> int:
    """Process one metadata row. Returns the number of ad frames written
    (0 on any validation failure), so the caller can balance non-ad frames
    against the actual ad-frame total without re-scanning the output dir.
    """
    date = row["fec"]
    tv_channel = row["can"]
    start_time = row["hin"]
    end_time = row["hfi"]

    # Check if video exists
    tv_channel_filename = get_channel_filename(tv_channel, config)
    date_filename = get_date_filename(date)
    if tv_channel_filename is None or date_filename is None:
        print(f"Row error: incorrect tv channel or date. cod={row["cod"]}")
        return 0

    filename = f"{date_filename}_{tv_channel_filename}_processed.mp4"
    file_path = f"{config.get("path").get("videos")}/{filename}"
    if not os.path.isfile(file_path):
        print(
            f"Row error: file doesn't exist. cod={row["cod"]}, file_path={file_path}")
        return 0

    if not isinstance(start_time, time) and not isinstance(end_time, time):
        print(
            f"Row error: start time/end time are not datetime. cod={row["cod"]}, start_time={start_time}, end_time={end_time}"
        )
        return 0

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
        return 0

    times_in_seconds = get_times(
        video_start_time,
        start_time,
        end_time,
    )

    result_path = f"{config.get("path").get("dataset")}/result/ad"
    custom_name = f'{filename.replace(".mp4", "")}_{row["cod"]}'
    return extract_frames(
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


def _crop(frame, custom_crop):
    if not custom_crop:
        return frame
    height, width = frame.shape[:2]
    return frame[
        custom_crop.get("top"): height - custom_crop.get("bottom"),
        custom_crop.get("left"): width - custom_crop.get("right"),
    ]


def _ocr_frame_at(target_frame: int, custom_crop, cod, label):
    """Seek to target_frame, decode, crop, OCR. Returns the parsed time or None.

    Used only for the boundary-OCR check below — the per-second extraction
    loop never calls this because it advances via grab() instead of seeking.
    """
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = video_capture.read()
    if not ret:
        return None
    try:
        return extract_datetime(_crop(frame, custom_crop))
    except Exception as e:
        print(f"OCR error at {label}. cod={cod}, error={e}")
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
    """Extract one frame per second of the ad window and write JPEGs.

    Two-phase design — the first phase decides whether the second phase
    needs to OCR every frame:

    1. Boundary OCR: read the first and last sampled frames, OCR their
       burned-in clocks. If both timestamps fall inside [ad_start_time,
       ad_end_time], the seek math from videos_metadata.start_time + the
       row's hin/hfi is correct and the middle frames are guaranteed to
       be in range too — so we skip OCR on them. If either boundary is
       missing or out of range, fall back to OCR'ing every frame (the
       original behavior) so that bad metadata still gets filtered.

    2. Sequential extraction: seek once to the window start, then advance
       one second at a time with grab() (no decode of skipped frames) and
       only retrieve() the frame we keep. Much cheaper than calling
       set(POS_FRAMES) per second, which forces a keyframe seek + decode.

    The function keeps `video_capture`, `fps`, `total_frames`, and
    `previous_video_path` as module-level globals so consecutive rows from
    the same video reuse the open VideoCapture (see CLAUDE.md).
    """
    global previous_video_path, video_capture, fps, total_frames
    os.makedirs(output_dir, exist_ok=True)

    if previous_video_path != video_path:
        if video_capture is not None:
            video_capture.release()

        previous_video_path = video_path
        video_capture = cv2.VideoCapture(video_path)
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Sample one frame per second of wall-clock video. `step` is the frame
    # delta between samples; rounded so non-integer fps (e.g. 29.97) doesn't
    # cause cumulative drift.
    step = int(round(fps))
    start_frame = int(times_in_seconds[0] * fps)
    end_frame = min(
        int((times_in_seconds[1] + 1) * fps), total_frames
    )  # + 1 to include the final frame
    n_samples = (end_frame - start_frame) // step + 1
    if n_samples <= 0:
        return 0
    last_sample_frame = start_frame + (n_samples - 1) * step

    # Phase 1 — boundary OCR. See the docstring for the rationale. For a
    # single-sample window we only need one OCR call. The per-frame OCR
    # branch in phase 2 keeps the original "OCR=None means keep the frame"
    # semantic from CLAUDE.md, so a video whose overlay was cropped out
    # behaves the same as before.
    first_time = _ocr_frame_at(start_frame, custom_crop, cod, "boundary-first")
    last_time = (
        first_time
        if n_samples == 1
        else _ocr_frame_at(last_sample_frame, custom_crop, cod, "boundary-last")
    )
    boundaries_ok = (
        first_time is not None
        and last_time is not None
        and ad_start_time <= first_time <= ad_end_time
        and ad_start_time <= last_time <= ad_end_time
    )

    # Phase 2 — sequential extraction. Seek once to the window start; from
    # here on we advance only with grab(). The boundary OCR above already
    # left the capture position somewhere else, so this set() is required.
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    counter = 0
    extracted = 0
    removed = 0
    for i in range(n_samples):
        if i > 0:
            # Advance (step - 1) frames with grab() — these are decoded but
            # not retrieved into Python, which is much cheaper than read().
            # The trailing read() below grabs + retrieves the frame we keep.
            # We can't skip the grab calls entirely because video frames
            # depend on prior frames (B/P-frames), so the decoder has to
            # walk every frame in order between keyframes.
            advanced = True
            for _ in range(step - 1):
                if not video_capture.grab():
                    advanced = False
                    break
            if not advanced:
                break
        ret, frame = video_capture.read()
        if not ret:
            break

        frame = _crop(frame, custom_crop)
        extracted += 1

        if boundaries_ok:
            # Fast path: trust the seek math, no per-frame OCR.
            cv2.imwrite(f"{output_dir}/{custom_name}_{counter}.jpg", frame)
        else:
            # Fallback path: original per-frame OCR filter. Hit when the
            # boundary clock was unreadable or out of range — usually a
            # sign of wrong videos_metadata.start_time, channel mismatch,
            # or the overlay being cropped out.
            try:
                frame_time = extract_datetime(frame)
            except Exception as e:
                print(f"OCR error (kept frame). cod={cod}, counter={counter}, error={e}")
                frame_time = None
            if frame_time is not None and (frame_time < ad_start_time or frame_time > ad_end_time):
                removed += 1
            else:
                cv2.imwrite(f"{output_dir}/{custom_name}_{counter}.jpg", frame)

        counter += 1

    if removed > 5:
        print(
            f"Row warning: OCR filter dropped {removed} frame(s) "
            f"(considered={extracted}, written={extracted - removed}). "
            f"Likely wrong video/metadata (start_time, hin/hfi, or channel mismatch). "
            f"cod={cod}, custom_name={custom_name}"
        )

    return extracted - removed
