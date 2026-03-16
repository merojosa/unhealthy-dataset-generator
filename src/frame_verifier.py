import cv2
import numpy as np
from datetime import datetime
from src.time_calculator import extract_datetime_from_frame


def read_frame_at(video_capture: cv2.VideoCapture, frame_number: int) -> np.ndarray | None:
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = video_capture.read()
    if not ret:
        return None
    return frame


def find_verified_frame_position(
    video_capture: cv2.VideoCapture,
    expected_datetime: datetime,
    initial_frame: int,
    fps: float,
    total_frames: int,
    roi: tuple = None,
    datetime_format: str = "%d/%m/%Y %H:%M:%S",
    tolerance_seconds: int = 2,
    max_search_seconds: int = 300,
) -> int | None:
    # Phase 1: Quick check at calculated position
    frame = read_frame_at(video_capture, initial_frame)
    ocr_dt = extract_datetime_from_frame(frame, roi, datetime_format) if frame is not None else None

    if ocr_dt is None:
        # Try 3 neighboring frames
        for offset in [-int(fps), int(fps), int(fps * 2)]:
            neighbor = max(0, min(initial_frame + offset, total_frames - 1))
            frame = read_frame_at(video_capture, neighbor)
            ocr_dt = extract_datetime_from_frame(frame, roi, datetime_format) if frame is not None else None
            if ocr_dt is not None:
                break

        if ocr_dt is None:
            print(f"  Warning: OCR failed at frame {initial_frame}, using unverified position")
            return initial_frame

    drift = (ocr_dt - expected_datetime).total_seconds()
    if abs(drift) <= tolerance_seconds:
        return initial_frame

    # Phase 2: Correction via drift jump + binary search
    print(f"  Datetime mismatch: expected={expected_datetime}, ocr={ocr_dt}, drift={drift:.1f}s")

    # Jump by drift amount
    corrected_frame = initial_frame - int(drift * fps)
    corrected_frame = max(0, min(corrected_frame, total_frames - 1))

    frame = read_frame_at(video_capture, corrected_frame)
    ocr_dt = extract_datetime_from_frame(frame, roi, datetime_format) if frame is not None else None

    if ocr_dt is not None:
        drift = (ocr_dt - expected_datetime).total_seconds()
        if abs(drift) <= tolerance_seconds:
            print(f"  Corrected start frame: {initial_frame} -> {corrected_frame}")
            return corrected_frame

    # Binary search
    max_offset_frames = int(max_search_seconds * fps)
    low = max(0, initial_frame - max_offset_frames)
    high = min(total_frames - 1, initial_frame + max_offset_frames)
    best_frame = corrected_frame
    best_drift = abs(drift) if ocr_dt is not None else float('inf')

    for _ in range(10):
        if low > high:
            break

        mid = (low + high) // 2
        frame = read_frame_at(video_capture, mid)
        ocr_dt = extract_datetime_from_frame(frame, roi, datetime_format) if frame is not None else None

        if ocr_dt is None:
            # Can't OCR this frame, narrow search arbitrarily
            high = mid - 1
            continue

        drift = (ocr_dt - expected_datetime).total_seconds()
        if abs(drift) < best_drift:
            best_drift = abs(drift)
            best_frame = mid

        if abs(drift) <= tolerance_seconds:
            print(f"  Corrected start frame: {initial_frame} -> {mid}")
            return mid

        if drift > 0:
            high = mid - int(fps)
        else:
            low = mid + int(fps)

    if best_drift <= tolerance_seconds:
        print(f"  Corrected start frame: {initial_frame} -> {best_frame}")
        return best_frame

    print(f"  Warning: Could not converge (best drift={best_drift:.1f}s), using best match at frame {best_frame}")
    return best_frame
