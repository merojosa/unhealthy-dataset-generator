"""Offline data augmentation for the TRAIN split of the tip multiclass dataset.

Reads `tip_multiclass_dataset.csv` (needs the `split` column written by the generator) and
tops up each class in train to `round(target_ratio * largest train class)` frames, capped at
`max_multiplier * original train frames`. Val and test are never touched: augmented copies
of a frame stay in its split and group, so they cannot leak.

Writes `augmented/` (wiped each run) and `tip_multiclass_augmented.csv` with the columns of
the main CSV plus `source_image`. The main CSV and `images/` are not modified.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

CSV_NAME = "tip_multiclass_dataset.csv"
AUG_CSV_NAME = "tip_multiclass_augmented.csv"
IMAGES_DIR = "images"
AUG_DIR = "augmented"


def plan_augmentation(train_counts: pd.Series, target_ratio: float, max_multiplier: float) -> dict:
    """class -> number of new frames, so each class reaches min(target, n * max_multiplier)."""
    target = round(target_ratio * train_counts.max())
    return {cls: max(0, min(target, int(n * max_multiplier)) - int(n)) for cls, n in train_counts.items()}


def augment_image(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Photometric + mild geometric jitter. No flips: mirrored logos/text would change the ad."""
    h, w = image.shape[:2]

    # Random resized crop keeping the aspect ratio (80-100% of the area).
    scale = np.sqrt(rng.uniform(0.80, 1.0))
    ch, cw = int(h * scale), int(w * scale)
    y0, x0 = rng.integers(0, h - ch + 1), rng.integers(0, w - cw + 1)
    out = cv2.resize(image[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_LINEAR)

    # Small rotation and translation.
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-5, 5), 1.0)
    matrix[:, 2] += (rng.uniform(-0.03, 0.03) * w, rng.uniform(-0.03, 0.03) * h)
    out = cv2.warpAffine(out, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    # Brightness / contrast / saturation, and a tiny hue shift (brand colours carry signal).
    out = out.astype(np.float32)
    out = (out - out.mean()) * rng.uniform(0.8, 1.2) + out.mean() * rng.uniform(0.8, 1.2)
    hsv = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-2, 2)) % 180
    hsv[..., 1] *= rng.uniform(0.8, 1.2)
    out = cv2.cvtColor(np.clip(hsv, 0, [179, 255, 255]).astype(np.uint8), cv2.COLOR_HSV2BGR)

    if rng.random() < 0.3:
        k = int(rng.choice([3, 5]))
        out = cv2.GaussianBlur(out, (k, k), 0)
    if rng.random() < 0.3:
        noise = rng.normal(0, rng.uniform(3, 8), out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


def augment_dataset(root: Path, target_ratio: float, max_multiplier: float, seed: int, dry_run: bool) -> int:
    csv_path, images_dir = root / CSV_NAME, root / IMAGES_DIR
    if not csv_path.is_file() or not images_dir.is_dir():
        print(f"Error: {csv_path} or {images_dir} not found; run the multiclass generator first.")
        return 2
    df = pd.read_csv(csv_path)
    if "split" not in df.columns or df["split"].isna().any():
        print("Error: CSV has no complete 'split' column; regenerate it with the multiclass generator.")
        return 3

    train = df[df["split"] == "train"].sort_values("image_name")
    counts = train["tip"].value_counts().sort_index()
    plan = plan_augmentation(counts, target_ratio, max_multiplier)

    print(f"Train frames per class (target {round(target_ratio * counts.max())}, max x{max_multiplier}):")
    for tip, n in counts.items():
        print(f"  {tip}: {n} + {plan[tip]} augmented = {n + plan[tip]}")
    print(f"Total new frames: {sum(plan.values())}")
    if dry_run:
        print("Dry run: nothing written.")
        return 0

    out_dir = root / AUG_DIR
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    rng = np.random.default_rng(seed)
    records = []
    for tip, n_new in plan.items():
        sources = train[train["tip"] == tip].to_dict("records")
        for k in range(n_new):
            src = sources[k % len(sources)]
            image = cv2.imread(str(images_dir / src["image_name"]))
            if image is None:
                print(f"Warning: cannot read {src['image_name']}; skipped.")
                continue
            name = f"{Path(src['image_name']).stem}_aug{k // len(sources) + 1:02d}.jpg"
            quality = int(rng.integers(70, 96))
            cv2.imwrite(str(out_dir / name), augment_image(image, rng), [cv2.IMWRITE_JPEG_QUALITY, quality])
            records.append({**{c: src[c] for c in df.columns}, "image_name": name,
                            "source_image": src["image_name"]})

    pd.DataFrame(records, columns=[*df.columns, "source_image"]).to_csv(root / AUG_CSV_NAME, index=False, encoding="utf-8")
    print(f"Wrote {len(records)} augmented train frames to {out_dir} and {root / AUG_CSV_NAME}")
    return 0


def load_config() -> dict:
    try:
        with open("config.json") as f:
            return json.load(f)
    except FileNotFoundError:
        with open("default_config.json") as f:
            return json.load(f)


def main() -> None:
    config = load_config()
    default_root = config["path"]["multiclass_dataset"]
    aug = config["multiclass"]["augmentation"]

    parser = argparse.ArgumentParser(description="Augment the train split of the tip multiclass dataset.")
    parser.add_argument("dataset", nargs="?", default=default_root,
                        help=f"Multiclass dataset root (default: path.multiclass_dataset = {default_root})")
    parser.add_argument("--target-ratio", type=float, default=aug["target_ratio"],
                        help="Per-class train target as a fraction of the largest train class.")
    parser.add_argument("--max-multiplier", type=float, default=aug["max_multiplier"],
                        help="Cap: a class never exceeds this multiple of its original train frames.")
    parser.add_argument("--seed", type=int, default=aug["seed"])
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without writing files.")
    args = parser.parse_args()

    if args.target_ratio <= 0 or args.max_multiplier < 1:
        print("Error: --target-ratio must be > 0 and --max-multiplier >= 1.", file=sys.stderr)
        sys.exit(4)
    sys.exit(augment_dataset(Path(args.dataset), args.target_ratio, args.max_multiplier, args.seed, args.dry_run))


if __name__ == "__main__":
    main()
