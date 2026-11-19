import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

CSV_NAME = "tip_multiclass_dataset.csv"


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
