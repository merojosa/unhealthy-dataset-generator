# This file is just to test by eye how a video should be cropped.
# Pass the video path and the crop params. It will return what you need to save in the config file.
# Example:
# py test_custom_crop_params.py --video_path test_dataset/videos/2024-05-15_canal_7.mp4 --top 8 --bottom 40 --left 13 --right 372

import argparse
import cv2
import random

parser = argparse.ArgumentParser(description="Get crop params")

parser.add_argument("--video_path", type=str, required=True)
parser.add_argument("--top", type=int, required=True)
parser.add_argument("--bottom", type=int, required=True)
parser.add_argument("--left", type=int, required=True)
parser.add_argument("--right", type=int, required=True)

try:
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        raise ValueError("Could not read the video file or it has no frames")

    random_frame = random.randint(0, total_frames - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError("Could not read the random frame")

    height, width = frame.shape[:2]
    frame = frame[args.top : height - args.bottom, args.left : width - args.right]

    window_name = "Cropped frame"
    cv2.imshow(window_name, frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(
        f'"crop":{{"top": {args.top}, "bottom": {args.bottom}, "left": {args.left}, "right": {args.right}}}'
    )

except Exception as e:
    print("Error:", e)
