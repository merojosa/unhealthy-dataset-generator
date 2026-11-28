"""Offline data augmentation for the TRAIN split of the marketing multilabel dataset.

Runs only when the real train share is below `min_train_share`; then adds augmented copies
of train frames until train is `target_train_share` of all frames. A copy carries every label
of its source, so sources are drawn with weight = max over their positive labels of
1 / train positives of that label (rare labels first), and no source is copied more than
`max_multiplier - 1` times. Val and test are never touched.

Writes `augmented/` (wiped each run) and `marketing_multilabel_augmented.csv` (main CSV columns
plus `source_image`). The main CSV and `images/` are not modified.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.marketing_strategies_multilabel_dataset_generator.split import positives
from src.tip_multiclass_dataset_generator.augment import augment_image

CSV_NAME = "marketing_multilabel_dataset.csv"
AUG_CSV_NAME = "marketing_multilabel_augmented.csv"
IMAGES_DIR = "images"
AUG_DIR = "augmented"


def label_columns(df: pd.DataFrame) -> list:
    cols = list(df.columns)
    return cols[cols.index("rpr"):cols.index("rie") + 1]


def augment_dataset(root: Path, min_share: float, target_share: float, max_multiplier: float,
                    seed: int, dry_run: bool) -> int:
    csv_path, images_dir = root / CSV_NAME, root / IMAGES_DIR
    if not csv_path.is_file() or not images_dir.is_dir():
        print(f"Error: {csv_path} or {images_dir} not found; run the multilabel generator first.")
        return 2
    df = pd.read_csv(csv_path, dtype=str)
    if "split" not in df.columns or df["split"].isna().any():
        print("Error: CSV has no complete 'split' column; regenerate it with the multilabel generator.")
        return 3

    out_dir, aug_csv = root / AUG_DIR, root / AUG_CSV_NAME
    train = df[df["split"] == "train"].sort_values("image_name").reset_index(drop=True)
    share = len(train) / len(df)
    print(f"Real train share: {share:.1%} ({len(train)}/{len(df)})")
    if share >= min_share:
        print(f"Train share >= {min_share:.0%}: no augmentation needed.")
        if not dry_run:
            shutil.rmtree(out_dir, ignore_errors=True)
            aug_csv.unlink(missing_ok=True)
        return 0

    n_new = int(np.ceil((target_share * len(df) - len(train)) / (1 - target_share)))
    y = positives(train, label_columns(train)).to_numpy()
    rarity = 1.0 / np.maximum(y.sum(axis=0), 1)
    weights = (y * rarity).max(axis=1)
    weights[weights == 0] = rarity.min()
    capacity = np.full(len(train), int(max_multiplier) - 1)
    if capacity.sum() < n_new:
        print(f"Warning: cap x{max_multiplier} allows only {capacity.sum()} copies; {n_new} requested.")
        n_new = int(capacity.sum())

    rng = np.random.default_rng(seed)
    copies = np.zeros(len(train), dtype=int)
    for _ in range(n_new):
        p = np.where(copies < capacity, weights, 0.0)
        copies[rng.choice(len(train), p=p / p.sum())] += 1

    y_after = y + y * copies[:, None]
    print(f"Adding {n_new} augmented frames -> train share {(len(train) + n_new) / (len(df) + n_new):.1%}")
    print(f"Rarest train label positives: {int(y.sum(axis=0)[y.sum(axis=0) > 0].min())} -> "
          f"{int(y_after.sum(axis=0)[y.sum(axis=0) > 0].min())}")
    if dry_run:
        print("Dry run: nothing written.")
        return 0

    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    records = []
    for i in np.flatnonzero(copies):
        src = train.iloc[i]
        image = cv2.imread(str(images_dir / src["image_name"]))
        if image is None:
            print(f"Warning: cannot read {src['image_name']}; skipped.")
            continue
        for k in range(copies[i]):
            name = f"{Path(src['image_name']).stem}_aug{k + 1:02d}.jpg"
            quality = int(rng.integers(70, 96))
            cv2.imwrite(str(out_dir / name), augment_image(image, rng), [cv2.IMWRITE_JPEG_QUALITY, quality])
            records.append({**src.to_dict(), "image_name": name, "source_image": src["image_name"]})

    pd.DataFrame(records, columns=[*df.columns, "source_image"]).to_csv(aug_csv, index=False)
    print(f"Wrote {len(records)} augmented train frames to {out_dir} and {aug_csv}")
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
    default_root = config["path"]["multilabel_dataset"]
    aug = config["multilabel"]["augmentation"]

    parser = argparse.ArgumentParser(description="Augment the train split of the multilabel dataset.")
    parser.add_argument("dataset", nargs="?", default=default_root,
                        help=f"Multilabel dataset root (default: path.multilabel_dataset = {default_root})")
    parser.add_argument("--min-train-share", type=float, default=aug["min_train_share"],
                        help="Augment only if the real train share is below this.")
    parser.add_argument("--target-train-share", type=float, default=aug["target_train_share"],
                        help="Train share (real + augmented) to reach when augmenting.")
    parser.add_argument("--max-multiplier", type=float, default=aug["max_multiplier"],
                        help="Cap: a source frame appears at most this many times (original included).")
    parser.add_argument("--seed", type=int, default=aug["seed"])
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without writing files.")
    args = parser.parse_args()

    if not 0 < args.target_train_share < 1 or args.max_multiplier < 1:
        print("Error: --target-train-share must be in (0, 1) and --max-multiplier >= 1.", file=sys.stderr)
        sys.exit(4)
    sys.exit(augment_dataset(Path(args.dataset), args.min_train_share, args.target_train_share,
                             args.max_multiplier, args.seed, args.dry_run))


if __name__ == "__main__":
    main()
