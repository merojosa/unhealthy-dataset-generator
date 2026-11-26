import argparse
import json
from pathlib import Path
import sys
import pandas as pd
import os


def check_dataset(csv_path: Path, images_dir: Path) -> int:
    problems = 0

    if not csv_path.is_file():
        print(f"Error: CSV not found: {csv_path}")
        return 2

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return 3

    if 'image_name' not in df.columns:
        print("Error: 'image_name' column missing from CSV")
        return 4

    csv_images = set(df['image_name'].astype(str).tolist())

    ad_dir = images_dir
    if not ad_dir.is_dir():
        print(f"Error: images folder not found: {ad_dir}")
        return 5

    disk_images = set([f for f in os.listdir(ad_dir) if f.lower().endswith('.jpg')])

    missing_on_disk = sorted(list(csv_images - disk_images))
    missing_in_csv = sorted(list(disk_images - csv_images))

    dup_count = int(df['image_name'].duplicated().sum())

    # check for rpr..rie columns presence
    lower_cols = [c.lower() for c in df.columns]
    has_rpr = 'rpr' in lower_cols
    has_rie = 'rie' in lower_cols

    print(f"CSV: {csv_path}")
    print(f"Rows: {len(df)}")
    print(f"Unique image_name: {len(csv_images)}")
    print(f"Images in {ad_dir}: {len(disk_images)}")
    print(f"Duplicate image_name entries: {dup_count}")

    if not has_rpr or not has_rie:
        print("Error: required columns 'rpr' and/or 'rie' not present in CSV header.")
        problems += 1
    else:
        print("Found 'rpr' and 'rie' columns in CSV header.")

    if missing_on_disk:
        print(f"Images referenced in CSV but missing on disk: {len(missing_on_disk)}")
        for p in missing_on_disk[:20]:
            print(f"  - {p}")
        if len(missing_on_disk) > 20:
            print(f"  ... and {len(missing_on_disk)-20} more")
        problems += 1
    else:
        print("All images referenced in CSV exist on disk.")

    if missing_in_csv:
        print(f"Images on disk but not referenced in CSV: {len(missing_in_csv)}")
        for p in missing_in_csv[:20]:
            print(f"  - {p}")
        if len(missing_in_csv) > 20:
            print(f"  ... and {len(missing_in_csv)-20} more")
        # not necessarily a problem, but report
    else:
        print("All disk images are referenced in CSV.")

    if dup_count:
        print("Error: duplicate image_name entries found in CSV.")
        problems += 1

    if 'group' not in df.columns:
        print("Error: 'group' column missing; regenerate the dataset after running group_frames.py.")
        problems += 1
    else:
        null_groups = int(df['group'].isna().sum())
        sizes = df['group'].value_counts()
        print(f"Groups: {len(sizes)} (max frames per group: {sizes.max() if len(sizes) else 0})")
        if null_groups:
            print(f"Error: {null_groups} rows have an empty 'group'; run group_frames.py and regenerate.")
            problems += 1

    if problems == 0:
        print("OK: multilabel dataset looks consistent.")
        return 0
    else:
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
    default_dataset = load_config()["path"]["multilabel_dataset"]

    parser = argparse.ArgumentParser(description='Check multilabel dataset CSV and image files')
    parser.add_argument('dataset', nargs='?', default=default_dataset, help=f'Multilabel dataset root containing images/ and marketing_multilabel_dataset.csv (default: path.multilabel_dataset = {default_dataset})')
    args = parser.parse_args()

    dataset = Path(args.dataset)
    rc = check_dataset(dataset / 'marketing_multilabel_dataset.csv', dataset / 'images')
    sys.exit(rc)


if __name__ == '__main__':
    main()
