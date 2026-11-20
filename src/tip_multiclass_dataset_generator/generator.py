import os
import re
import shutil
from pathlib import Path
from typing import Optional, Union

import pandas as pd

CSV_NAME = "tip_multiclass_dataset.csv"
IMAGES_DIR = "images"

COD_PATTERN = re.compile(r'_(\d{5})_\d+\.jpg$')
TIP_PATTERN = re.compile(r'^\s*(\d+)\s*=\s*(.*)$', re.DOTALL)
NON_AD_PATTERN = re.compile(r'^(.*)_non_ad_(\d+)\.jpg$', re.IGNORECASE)

# Synthetic class added by us for non-ad frames; 0 never collides with metadata tip codes.
NON_AD_TIP = 0
NON_AD_LABEL = "No es anuncio (non_ad)"


def _non_ad_sort_key(filename: str):
    """Time order: by video, then by numeric frame counter (lexicographic order breaks on 10 vs 9)."""
    match = NON_AD_PATTERN.match(filename)
    return (match.group(1), int(match.group(2))) if match else (filename, -1)


def sample_non_ad(non_ad_images_dir: Path, count: int) -> list:
    """Return `count` non-ad filenames evenly spread across the time-ordered sequence."""
    files = sorted((f for f in os.listdir(non_ad_images_dir) if f.lower().endswith('.jpg')), key=_non_ad_sort_key)
    count = min(count, len(files))
    if count <= 0:
        return []
    return [files[i * len(files) // count] for i in range(count)]


def parse_tip(value) -> Optional[tuple]:
    """Parse a metadata `tip` cell like "2=Productos..." into (2, "Productos...")."""
    if pd.isna(value):
        return None
    match = TIP_PATTERN.match(str(value))
    if not match:
        return None
    return int(match.group(1)), match.group(2).strip()


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
    tip_values: Optional[list] = None,
    non_ad_images_dir: Optional[Union[str, Path]] = None,
    non_ad_ratio: float = 0.0,
    groups_csv: Optional[Union[str, Path]] = None,
) -> int:
    """Build a self-contained multiclass dataset (class = metadata `tip`) from the binary dataset's frames.

    - `ad_images_dir`: folder containing ad images (jpg), e.g. `{binary}/result/ad`
    - `metadata_path`: path to metadata.xlsx
    - `output_dir`: dataset root to write; gets `images/` (copies of every labelled
      frame, wiped on each run) and `tip_multiclass_dataset.csv`
    - `tip_values`: if given, frames whose tip is not in this list are skipped
    - `non_ad_images_dir` / `non_ad_ratio`: adds class `NON_AD_TIP` with
      `round(non_ad_ratio * largest ad class)` frames sampled evenly from this folder
    - `groups_csv`: binary `result/groups.csv`; its creative/video group is copied
      to the `group` column so splits can keep each group in one split

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
        try:
            df_excel = pd.read_excel(metadata_path, sheet_name="Base combinada")
        except ValueError:
            df_excel = pd.read_excel(metadata_path, sheet_name=0)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return 4

    for col in ("cod", "tip"):
        if col not in df_excel.columns:
            print(f"Error: '{col}' column not found in Excel metadata.")
            return 5

    # cod -> (tip, label), first occurrence wins
    tip_by_cod = {}
    for cod, tip in df_excel.drop_duplicates(subset="cod")[["cod", "tip"]].itertuples(index=False):
        parsed = parse_tip(tip)
        if parsed is not None and not pd.isna(cod):
            tip_by_cod[int(cod)] = parsed

    all_files = sorted(f for f in os.listdir(ad_images_dir) if f.lower().endswith('.jpg'))
    print(f"Found {len(all_files)} pre-extracted ad frames in {ad_images_dir}.")

    allowed = set(tip_values) if tip_values is not None else None
    records = []
    skipped = 0
    for filename in all_files:
        match = COD_PATTERN.search(filename)
        parsed = tip_by_cod.get(int(match.group(1))) if match else None
        if parsed is None or (allowed is not None and parsed[0] not in allowed):
            skipped += 1
            continue
        records.append({"image_name": filename, "tip": parsed[0], "tip_label": parsed[1]})

    if skipped:
        print(f"Skipped {skipped} frames with no matching/allowed tip.")

    if non_ad_images_dir is not None and non_ad_ratio > 0:
        non_ad_images_dir = Path(non_ad_images_dir)
        if not non_ad_images_dir.is_dir():
            print(f"Error: non-ad images folder not found: {non_ad_images_dir}")
            return 10
        largest = max(pd.Series([r["tip"] for r in records]).value_counts(), default=0)
        target = round(non_ad_ratio * largest)
        sampled = sample_non_ad(non_ad_images_dir, target)
        if len(sampled) < target:
            print(f"Warning: requested {target} non-ad frames but only {len(sampled)} are available.")
        print(f"Sampled {len(sampled)} non-ad frames from {non_ad_images_dir} (ratio {non_ad_ratio} x largest class {largest}).")
        for filename in sampled:
            records.append({"image_name": filename, "tip": NON_AD_TIP, "tip_label": NON_AD_LABEL,
                            "_src": non_ad_images_dir / filename})

    if not records:
        print("Warning: no matching records found; CSV will be empty.")

    groups = load_groups(Path(groups_csv) if groups_csv is not None else None)
    for record in records:
        group = groups.get(record["image_name"])
        if group is None and record["tip"] == NON_AD_TIP:
            match = NON_AD_PATTERN.match(record["image_name"])
            group = f"non_ad_{match.group(1)}" if match else None
        record["group"] = group
    ungrouped = sum(r["group"] is None for r in records)
    if groups and ungrouped:
        print(f"Warning: {ungrouped} frames are not in {groups_csv}; re-run group_frames.py.")

    df_out = pd.DataFrame(records, columns=["image_name", "tip", "tip_label", "group"])

    try:
        if images_out.exists():
            shutil.rmtree(images_out)
        images_out.mkdir(parents=True)
        for record in records:
            src = record.get("_src", ad_images_dir / record["image_name"])
            shutil.copy2(src, images_out / record["image_name"])
    except Exception as e:
        print(f"Error copying images: {e}")
        return 9

    try:
        df_out.to_csv(output_csv, index=False, encoding="utf-8")
    except Exception as e:
        print(f"Error writing CSV: {e}")
        return 6

    print(f"Copied {len(records)} images to {images_out}")
    print(f"Wrote {len(df_out)} rows to {output_csv}")
    if records:
        print("Class distribution (tip: count):")
        for tip, count in df_out["tip"].value_counts().sort_index().items():
            print(f"  {tip}: {count}")
    return 0
