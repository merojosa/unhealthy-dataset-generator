"""Assign a leakage-safe group to every frame in {dataset}/result and cap frames per creative.

The same commercial airs many times under different metadata rows (`cod`), so splitting
by row leaks near-identical frames across train/val/test. This script groups frames by
*creative* instead:

1. Ad frames: `cod` -> metadata `nom`, normalized (accents, case, punctuation, spaces dropped).
2. Ad creatives whose mean CLIP embeddings are near-identical are merged (typos in `nom`,
   alternate cuts of the same spot). Only centroids are compared, so a single black frame
   or channel logo can't chain unrelated creatives together.
3. Non-ad frames: one group per source video.

Ad groups larger than `max_frames_per_creative` are trimmed to evenly-spaced survivors; the
rest are moved (never deleted) to result/capped/ with a manifest. Writes result/groups.csv.
"""
import argparse
import json
import re
import shutil
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rebalance_pipeline.rebalance import evenly_spaced_keep_indices  # noqa: E402

GROUPS_CSV = "groups.csv"
CAPPED_DIR = "capped"
MANIFEST_NAME = "capped_manifest.csv"

AD_PATTERN = re.compile(r'^(.*)_(\d{5})_(\d+)\.jpg$', re.IGNORECASE)
NON_AD_PATTERN = re.compile(r'^(.*)_non_ad_(\d+)\.jpg$', re.IGNORECASE)


def load_config() -> dict:
    try:
        with open("config.json") as f:
            return json.load(f)
    except FileNotFoundError:
        with open("default_config.json") as f:
            return json.load(f)


def normalize_nom(value) -> str:
    """'Suero X, nuevo sabor' -> 'sueroxnuevosabor' so spacing/accents/case typos collapse."""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r'[^0-9a-z]', '', text.lower())


def list_jpgs(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.glob("*.jpg")) if folder.is_dir() else []


def ad_sort_key(filename: str):
    """Airing order: cod (metadata rows are chronological), then frame counter."""
    match = AD_PATTERN.match(filename)
    return (int(match.group(2)), int(match.group(3))) if match else (sys.maxsize, 0)


def load_nom_by_cod(metadata_path: Path) -> dict[int, str]:
    try:
        df = pd.read_excel(metadata_path, sheet_name="Base combinada")
    except ValueError:
        df = pd.read_excel(metadata_path, sheet_name=0)
    for col in ("cod", "nom"):
        if col not in df.columns:
            raise KeyError(f"'{col}' column not found in {metadata_path}")
    df = df.dropna(subset=["cod"]).drop_duplicates(subset="cod")
    return {int(cod): nom for cod, nom in zip(df["cod"], df["nom"]) if not pd.isna(nom)}


def compute_centroids(paths: list[Path], keys: list[str], batch_size: int) -> tuple[list[str], np.ndarray]:
    """Mean normalized CLIP embedding per creative key (re-normalized so dot = cosine)."""
    from sentence_transformers import SentenceTransformer
    from duplicate_images_removal_pipeline.image_similarity import load_image
    from tqdm import tqdm

    model = SentenceTransformer("clip-ViT-B-32")
    sums: dict[str, np.ndarray] = {}
    for start in tqdm(range(0, len(paths), batch_size), desc="Embedding ad frames"):
        batch = [(load_image(p), k) for p, k in zip(paths[start:start + batch_size], keys[start:start + batch_size])]
        batch = [(img, k) for img, k in batch if img is not None]
        if not batch:
            continue
        embeddings = model.encode([img for img, _ in batch], batch_size=batch_size, convert_to_numpy=True,
                                  normalize_embeddings=True, show_progress_bar=False)
        for (img, key), emb in zip(batch, embeddings):
            sums[key] = sums.get(key, 0) + emb.astype(np.float32)
            img.close()

    ordered = sorted(sums)
    centroids = np.vstack([sums[k] for k in ordered]) if ordered else np.empty((0, 0), dtype=np.float32)
    if len(ordered):
        centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    return ordered, centroids


def merge_creatives(keys: list[str], centroids: np.ndarray, threshold: float) -> tuple[dict[str, str], list[tuple]]:
    """Union creatives whose centroid cosine >= threshold. Returns (key -> root key, merged pairs)."""
    from duplicate_images_removal_pipeline.image_similarity import UnionFind

    union_find = UnionFind(len(keys))
    merged = []
    if len(keys):
        sims = centroids @ centroids.T
        for i, j in zip(*np.triu_indices(len(keys), k=1)):
            if sims[i, j] >= threshold:
                union_find.union(int(i), int(j))
                merged.append((keys[i], keys[j], float(sims[i, j])))
    # groups() members keep sorted order, so members[0] is the lexicographically-first key.
    root = {keys[i]: keys[members[0]] for members in union_find.groups().values() for i in members}
    return root, merged


def main() -> None:
    config = load_config()
    grouping = config["grouping"]
    default_result = str(Path(config["path"]["dataset"]) / "result")

    parser = argparse.ArgumentParser(description="Assign leakage-safe creative groups to result/ frames "
                                                 "and cap frames per creative into result/capped/.")
    parser.add_argument("result", nargs="?", default=default_result,
                        help=f"result/ folder containing ad/ and non_ad/; metadata.xlsx is read from its parent "
                             f"(default: {default_result})")
    parser.add_argument("--max-frames", type=int, default=None,
                        help=f"Max ad frames per creative; 0 disables the cap "
                             f"(default: grouping.max_frames_per_creative = {grouping['max_frames_per_creative']})")
    parser.add_argument("--merge-threshold", type=float, default=None,
                        help=f"CLIP centroid cosine at which two creatives merge; >1 disables merging "
                             f"(default: grouping.merge_threshold = {grouping['merge_threshold']})")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Report groups and moves without touching files.")
    args = parser.parse_args()

    max_frames = args.max_frames if args.max_frames is not None else grouping["max_frames_per_creative"]
    threshold = args.merge_threshold if args.merge_threshold is not None else grouping["merge_threshold"]
    batch_size = args.batch_size if args.batch_size is not None else grouping["batch_size"]
    if max_frames < 0:
        sys.exit("Error: --max-frames must be >= 0.")

    result_dir = Path(args.result)
    metadata_path = result_dir.parent / "metadata.xlsx"
    if not (result_dir / "ad").is_dir():
        sys.exit(f"Error: '{result_dir / 'ad'}' not found.")
    if not metadata_path.is_file():
        sys.exit(f"Error: metadata.xlsx not found: {metadata_path}")

    nom_by_cod = load_nom_by_cod(metadata_path)
    capped_ad_dir = result_dir / CAPPED_DIR / "ad"

    # Previously capped frames still define their creative, so re-runs keep groups stable
    # and don't re-trim a creative that is already at the cap.
    ad_frames = []  # (filename, path, creative key, nom, already capped)
    for folder, capped in ((result_dir / "ad", False), (capped_ad_dir, True)):
        for name in list_jpgs(folder):
            match = AD_PATTERN.match(name)
            cod = int(match.group(2)) if match else None
            nom = nom_by_cod.get(cod)
            key = normalize_nom(nom) if nom is not None else ""
            ad_frames.append((name, folder / name, key or f"cod{cod}", nom, capped))
    print(f"Ad frames: {sum(not f[4] for f in ad_frames)} active, {sum(f[4] for f in ad_frames)} already capped.")

    if threshold <= 1.0 and ad_frames:
        keys, centroids = compute_centroids([f[1] for f in ad_frames], [f[2] for f in ad_frames], batch_size)
        root, merged = merge_creatives(keys, centroids, threshold)
    else:
        root, merged = {}, []
    root = {f[2]: root.get(f[2], f[2]) for f in ad_frames}

    creative_keys = sorted({f[2] for f in ad_frames})
    group_ids = sorted(set(root[k] for k in creative_keys))
    group_name = {g: f"ad_{i:04d}" for i, g in enumerate(group_ids, 1)}
    print(f"Creatives: {len(creative_keys)} distinct normalized `nom`, {len(group_ids)} groups after CLIP merge "
          f"(threshold {threshold}, {len(merged)} merged pair(s)).")
    for a, b, sim in sorted(merged, key=lambda m: -m[2]):
        print(f"  merged ({sim:.3f}): {a[:60]}  <->  {b[:60]}")

    rows = []
    by_group: dict[str, list] = {}
    for name, path, key, nom, capped in ad_frames:
        group = group_name[root[key]]
        rows.append({"image_name": name, "label": "ad", "group": group, "creative": nom, "capped": capped})
        if not capped:
            by_group.setdefault(group, []).append((name, path))

    moves = []
    if max_frames > 0:
        for group, frames in sorted(by_group.items()):
            if len(frames) <= max_frames:
                continue
            frames.sort(key=lambda f: ad_sort_key(f[0]))
            keep = set(int(i) for i in evenly_spaced_keep_indices(len(frames), max_frames))
            moves.extend((group, name, path) for idx, (name, path) in enumerate(frames) if idx not in keep)
    moved_names = {name for _, name, _ in moves}
    for row in rows:
        row["capped"] = row["capped"] or row["image_name"] in moved_names

    for name in list_jpgs(result_dir / "non_ad"):
        match = NON_AD_PATTERN.match(name)
        video = match.group(1) if match else Path(name).stem
        rows.append({"image_name": name, "label": "non_ad", "group": f"non_ad_{video}", "creative": None,
                     "capped": False})

    df = pd.DataFrame(rows, columns=["image_name", "label", "group", "creative", "capped"])
    active = df[~df["capped"]]
    sizes = active[active["label"] == "ad"].groupby("group").size()
    print(f"Capping ad groups at {max_frames or 'no limit'}: {len(moves)} frame(s) to move into "
          f"{result_dir / CAPPED_DIR}.")
    if len(sizes):
        print(f"Active ad frames per group after cap: min {sizes.min()}, median {int(sizes.median())}, "
              f"max {sizes.max()} ({len(active[active['label'] == 'ad'])} total).")
    print(f"Non-ad groups (one per video): {df[df['label'] == 'non_ad']['group'].nunique()}.")

    if args.dry_run:
        print("[DRY RUN] No files moved, groups.csv not written.")
        return

    for group, name, path in moves:
        destination = capped_ad_dir / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            # Overwrites a stale copy left in capped/ rather than deleting the source frame.
            shutil.move(str(path), str(destination))

    if moves:
        manifest = result_dir / CAPPED_DIR / MANIFEST_NAME
        new = pd.DataFrame([{"original_path": str(Path("ad") / name), "moved_to": str(capped_ad_dir / name),
                             "group": group} for group, name, _ in moves])
        if manifest.is_file():
            new = pd.concat([pd.read_csv(manifest), new]).drop_duplicates(subset="original_path", keep="last")
        new.to_csv(manifest, index=False)
        print(f"Manifest written to: {manifest}")

    df.to_csv(result_dir / GROUPS_CSV, index=False, encoding="utf-8")
    print(f"Wrote {len(df)} rows to {result_dir / GROUPS_CSV}")


if __name__ == "__main__":
    main()
