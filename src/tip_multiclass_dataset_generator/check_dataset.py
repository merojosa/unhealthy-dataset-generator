import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

CSV_NAME = "tip_multiclass_dataset.csv"
AUG_CSV_NAME = "tip_multiclass_augmented.csv"
AUG_DIR = "augmented"


def check_augmented(root: Path, df: pd.DataFrame) -> int:
    """Validate the optional train-only augmentation output; returns the number of problems."""
    aug_csv = root / AUG_CSV_NAME
    if not aug_csv.is_file():
        print("No augmented train set (run augment.py to create one).")
        return 0
    aug = pd.read_csv(aug_csv)
    aug_dir = root / AUG_DIR
    disk = {f for f in os.listdir(aug_dir) if f.lower().endswith('.jpg')} if aug_dir.is_dir() else set()
    names = set(aug['image_name'].astype(str))
    source_split = df.set_index('image_name')['split'] if 'split' in df.columns else pd.Series(dtype=str)

    problems = 0
    checks = (
        ("augmented rows missing on disk", sorted(names - disk)),
        ("augmented images not in the augmented CSV", sorted(disk - names)),
        ("augmented rows not in train", aug.loc[aug['split'] != 'train', 'image_name'].tolist()),
        ("augmented rows whose source is not a train frame",
         aug.loc[aug['source_image'].map(source_split) != 'train', 'image_name'].tolist()),
        ("augmented names colliding with original images", sorted(names & set(df['image_name'].astype(str)))),
    )
    for label, items in checks:
        if items:
            print(f"Error: {len(items)} {label}, e.g. {items[:5]}")
            problems += 1
    print("Augmented train frames per class (tip: count):")
    for tip, count in aug['tip'].value_counts().sort_index().items():
        print(f"  {tip}: {count}")
    return problems


def check_dataset(csv_path: Path, images_dir: Path) -> int:
    if not csv_path.is_file():
        print(f"Error: CSV not found: {csv_path}")
        return 2

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return 3

    for col in ("image_name", "tip"):
        if col not in df.columns:
            print(f"Error: '{col}' column missing from CSV")
            return 4

    if not images_dir.is_dir():
        print(f"Error: images folder not found: {images_dir}")
        return 5

    csv_images = set(df['image_name'].astype(str))
    disk_images = {f for f in os.listdir(images_dir) if f.lower().endswith('.jpg')}
    missing_on_disk = sorted(csv_images - disk_images)
    missing_in_csv = sorted(disk_images - csv_images)
    dup_count = int(df['image_name'].duplicated().sum())
    null_tips = int(df['tip'].isna().sum())

    print(f"CSV: {csv_path}")
    print(f"Rows: {len(df)}")
    print(f"Unique image_name: {len(csv_images)}")
    print(f"Images in {images_dir}: {len(disk_images)}")
    print(f"Duplicate image_name entries: {dup_count}")
    print("Class distribution (tip: count):")
    for tip, count in df['tip'].value_counts().sort_index().items():
        print(f"  {tip}: {count}")

    problems = 0
    for label, items in (("referenced in CSV but missing on disk", missing_on_disk),
                         ("on disk but not referenced in CSV", missing_in_csv)):
        if items:
            print(f"Images {label}: {len(items)}")
            for p in items[:20]:
                print(f"  - {p}")
            if len(items) > 20:
                print(f"  ... and {len(items) - 20} more")
            problems += 1

    if dup_count:
        print("Error: duplicate image_name entries found in CSV.")
        problems += 1
    if null_tips:
        print(f"Error: {null_tips} rows have an empty 'tip'.")
        problems += 1

    if 'group' not in df.columns:
        print("Error: 'group' column missing; regenerate the dataset after running group_frames.py.")
        problems += 1
    else:
        null_groups = int(df['group'].isna().sum())
        print("Groups per class (tip: frames / groups):")
        for tip, sub in df.groupby('tip'):
            print(f"  {tip}: {len(sub)} / {sub['group'].nunique()}")
        mixed = df.dropna(subset=['group']).groupby('group')['tip'].nunique()
        mixed = mixed[mixed > 1]
        if len(mixed):
            # Not an error: CLIP merged look-alike creatives of different tips; they must still share a split.
            print(f"Note: {len(mixed)} group(s) span more than one tip: {', '.join(mixed.index[:10])}")
        if null_groups:
            print(f"Error: {null_groups} rows have an empty 'group'; run group_frames.py and regenerate.")
            problems += 1

    if 'split' not in df.columns or df['split'].isna().any():
        print("Error: 'split' column missing or incomplete; regenerate the dataset.")
        problems += 1
    elif 'group' in df.columns:
        leaked = df.groupby('group')['split'].nunique()
        leaked = leaked[leaked > 1]
        if len(leaked):
            print(f"Error: {len(leaked)} group(s) appear in more than one split: {', '.join(leaked.index[:10])}")
            problems += 1
        print("Frames per class and split:")
        print(pd.crosstab(df['tip'], df['split']).reindex(columns=['train', 'val', 'test'], fill_value=0).to_string())

    problems += check_augmented(csv_path.parent, df)

    if problems == 0:
        print("OK: multiclass dataset looks consistent.")
        return 0
    print(f"Found {problems} problem(s).")
    return 1


def load_config() -> dict:
    try:
        with open("config.json") as f:
            return json.load(f)
    except FileNotFoundError:
        with open("default_config.json") as f:
            return json.load(f)


def main():
    default_dataset = load_config()["path"]["multiclass_dataset"]

    parser = argparse.ArgumentParser(description='Check tip multiclass dataset CSV and image files')
    parser.add_argument('dataset', nargs='?', default=default_dataset, help=f'Multiclass dataset root containing images/ and {CSV_NAME} (default: path.multiclass_dataset = {default_dataset})')
    args = parser.parse_args()

    dataset = Path(args.dataset)
    sys.exit(check_dataset(dataset / CSV_NAME, dataset / 'images'))


if __name__ == '__main__':
    main()
