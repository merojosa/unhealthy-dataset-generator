import os
import re
import shutil
from pathlib import Path
from typing import Optional, Union

import pandas as pd

CSV_NAME = "marketing_multilabel_dataset.csv"
IMAGES_DIR = "images"


def load_groups(groups_csv: Optional[Path]) -> dict:
    """image_name -> leakage-safe group from the binary dataset's result/groups.csv (empty if absent)."""
    if groups_csv is None or not groups_csv.is_file():
        print(f"Warning: {groups_csv} not found; the 'group' column will be empty. "
              "Run src/grouping_pipeline/group_frames.py on the binary dataset first.")
        return {}
    df = pd.read_csv(groups_csv, usecols=["image_name", "group"])
    return dict(zip(df["image_name"].astype(str), df["group"].astype(str)))


def generate_dataset(
    ad_images_dir: Union[str, Path],
    metadata_path: Union[str, Path],
    output_dir: Union[str, Path],
    groups_csv: Optional[Union[str, Path]] = None,
) -> int:
    """Build a self-contained multilabel dataset from the binary dataset's ad frames.

    - `ad_images_dir`: folder containing ad images (jpg), e.g. `{binary}/result/ad`
    - `metadata_path`: path to metadata.xlsx
    - `output_dir`: dataset root to write; gets `images/` (copies of every labelled
      ad frame, wiped on each run) and `marketing_multilabel_dataset.csv`
    - `groups_csv`: binary `result/groups.csv`; its creative group is copied to the
      `group` column so splits can keep each group in one split

    Source images are only copied, never moved. Returns 0 on success, non-zero on error.
    """
    ad_images_dir = Path(ad_images_dir)
    metadata_path = Path(metadata_path)
    output_dir = Path(output_dir)
    images_out = output_dir / IMAGES_DIR
    output_csv = output_dir / CSV_NAME

    if not metadata_path.is_file():
        print(f"Error: metadata file not found: {metadata_path}")
        return 2

    if not ad_images_dir.is_dir():
        print(f"Error: ad images folder not found: {ad_images_dir}")
        return 3

    try:
        # try the expected sheet name, fall back to the first sheet
        try:
            df_excel = pd.read_excel(metadata_path, sheet_name="Base combinada")
        except ValueError:
            df_excel = pd.read_excel(metadata_path, sheet_name=0)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return 4

    cod_pattern = re.compile(r'_(\d{5})_\d+\.jpg$')

    all_files = [f for f in os.listdir(ad_images_dir) if f.lower().endswith('.jpg')]
    print(f"Found {len(all_files)} pre-extracted ad frames in {ad_images_dir}.")

    multilabel_records = []

    # Ensure 'cod' column exists
    if 'cod' not in df_excel.columns:
        print("Error: 'cod' column not found in Excel metadata.")
        return 5

    # determine which metadata columns to copy: columns from 'rpr' to 'rie' (inclusive)
    all_meta_cols = [c for c in df_excel.columns if c != 'cod']
    lower_cols = [c.lower() for c in all_meta_cols]
    try:
        start_idx = lower_cols.index('rpr')
        end_idx = lower_cols.index('rie')
    except ValueError:
        print("Error: could not find both 'rpr' and 'rie' columns in metadata.")
        return 7

    if start_idx > end_idx:
        print("Error: 'rpr' appears after 'rie' in metadata columns.")
        return 8

    cols_to_use = all_meta_cols[start_idx : end_idx + 1]

    for filename in all_files:
        match = cod_pattern.search(filename)
        if not match:
            continue

        code = int(match.group(1))
        matched_row = df_excel[df_excel['cod'] == code]

        if matched_row.empty:
            continue

        record = {"image_name": filename}
        # include only selected metadata columns
        for col in cols_to_use:
            # take first matching row's value
            val = matched_row.iloc[0][col]
            record[col] = val

        multilabel_records.append(record)

    if not multilabel_records:
        print("Warning: no matching records found; CSV will be empty.")

    groups = load_groups(Path(groups_csv) if groups_csv is not None else None)
    for record in multilabel_records:
        record["group"] = groups.get(record["image_name"])
    ungrouped = sum(r["group"] is None for r in multilabel_records)
    if groups and ungrouped:
        print(f"Warning: {ungrouped} frames are not in {groups_csv}; re-run group_frames.py.")

    df_multilabel = pd.DataFrame(multilabel_records, columns=["image_name", *cols_to_use, "group"])

    try:
        if images_out.exists():
            shutil.rmtree(images_out)
        images_out.mkdir(parents=True)
        for record in multilabel_records:
            shutil.copy2(ad_images_dir / record["image_name"], images_out / record["image_name"])
    except Exception as e:
        print(f"Error copying images: {e}")
        return 9

    try:
        df_multilabel.to_csv(output_csv, index=False)
    except Exception as e:
        print(f"Error writing CSV: {e}")
        return 6

    print(f"Copied {len(multilabel_records)} images to {images_out}")
    print(f"Wrote {len(df_multilabel)} rows to {output_csv}")
    return 0


if __name__ == "__main__":
    # simple CLI when run directly
    import argparse

    parser = argparse.ArgumentParser(description="Generate a multilabel dataset from ad images and metadata.xlsx")
    parser.add_argument("ad_dir", help="Folder containing ad images")
    parser.add_argument("metadata", help="Path to metadata.xlsx")
    parser.add_argument("output", help="Output dataset folder (gets images/ and the CSV)")
    parser.add_argument("--groups", help="Path to the binary dataset's result/groups.csv")
    args = parser.parse_args()

    raise SystemExit(generate_dataset(args.ad_dir, args.metadata, args.output, groups_csv=args.groups))