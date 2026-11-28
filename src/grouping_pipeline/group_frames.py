"""Assign a leakage-safe group to every frame in {dataset}/result and cap frames per creative.

The same commercial airs many times under different metadata rows (`cod`), so splitting
by row leaks near-identical frames across train/val/test. This script groups frames by
*creative* instead:

1. Ad frames: `cod` -> metadata `nom`, normalized (accents, case, punctuation, spaces dropped).
2. Ad creatives that look alike under DINOv2 are merged (typos in `nom`, alternate cuts,
   and different spots of the same brand/store that reuse shots or templates). A pair merges
   when its mean-embedding cosine >= `centroid_threshold`, or when its best frame match is
   >= `frame_threshold` AND its mean-embedding cosine >= `frame_match_min_centroid`. The
   centroid floor stops a single black frame or channel logo from chaining unrelated
   creatives together.
3. Non-ad frames: one group per channel (or per source video, `non_ad_group_by`), since
   recordings of the same channel share sets, hosts and logos.

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
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rebalance_pipeline.rebalance import evenly_spaced_keep_indices  # noqa: E402

GROUPS_CSV = "groups.csv"
CAPPED_DIR = "capped"
MANIFEST_NAME = "capped_manifest.csv"

AD_PATTERN = re.compile(r'^(.*)_(\d{5})_(\d+)\.jpg$', re.IGNORECASE)
NON_AD_PATTERN = re.compile(r'^(.*)_non_ad_(\d+)\.jpg$', re.IGNORECASE)
DATE_PREFIX = re.compile(r'^\d{4}-\d{2}-\d{2}_')
# DINOv2 input: the full 16:9 frame resized to a multiple of the 14 px patch size (no center
# crop, so logos and price tags near the edges count), normalized with ImageNet statistics.
DINO_INPUT_SIZE = (392, 224)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


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


def non_ad_group(filename: str, group_by: str) -> str:
    """'2024-04-04_canal_6_non_ad_12.jpg' -> 'non_ad_canal_6' (channel) or 'non_ad_2024-04-04_canal_6' (video)."""
    match = NON_AD_PATTERN.match(filename)
    video = match.group(1) if match else Path(filename).stem
    return f"non_ad_{DATE_PREFIX.sub('', video) if group_by == 'channel' else video}"


def embed_frames(paths: list[Path], keys: list[str], batch_size: int, model_name: str,
                 cache_dir: str) -> dict[str, np.ndarray]:
    """Normalized DINOv2 embedding of every frame, stacked per creative key."""
    import torch
    from transformers import AutoModel
    from duplicate_images_removal_pipeline.image_similarity import load_image
    from tqdm import tqdm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir).to(device).eval()
    frames: dict[str, list[np.ndarray]] = {}
    for start in tqdm(range(0, len(paths), batch_size), desc=f"Embedding ad frames ({device})"):
        batch = [(load_image(p), k) for p, k in zip(paths[start:start + batch_size], keys[start:start + batch_size])]
        batch = [(img, k) for img, k in batch if img is not None]
        if not batch:
            continue
        pixels = np.stack([(np.asarray(img.resize(DINO_INPUT_SIZE, Image.BICUBIC), dtype=np.float32) / 255
                            - IMAGENET_MEAN) / IMAGENET_STD for img, _ in batch])
        with torch.no_grad():
            out = model(pixel_values=torch.from_numpy(pixels).permute(0, 3, 1, 2).to(device))
        embeddings = out.pooler_output.float().cpu().numpy()
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        for (img, key), emb in zip(batch, embeddings):
            frames.setdefault(key, []).append(emb)
            img.close()
    return {key: np.vstack(embs) for key, embs in frames.items()}


def merge_creatives(frames: dict[str, np.ndarray], centroid_threshold: float, frame_threshold: float,
                    min_centroid: float) -> tuple[dict[str, str], list[tuple]]:
    """Union similar creatives (rule in the module docstring). Returns (key -> root key, merged pairs)."""
    from duplicate_images_removal_pipeline.image_similarity import UnionFind

    keys = sorted(frames)
    centroids = np.vstack([frames[k].mean(axis=0) for k in keys]) if keys else np.empty((0, 0), np.float32)
    if len(keys):
        centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    union_find = UnionFind(len(keys))
    merged = []
    for i, j in zip(*np.triu_indices(len(keys), k=1)):
        centroid_sim = float(centroids[i] @ centroids[j])
        sims = frames[keys[i]] @ frames[keys[j]].T
        # Best match of every frame of the smaller creative inside the larger one.
        best = sims.max(axis=1) if sims.shape[0] <= sims.shape[1] else sims.max(axis=0)
        best_frame, matching = float(best.max()), int((best >= frame_threshold).sum())
        if centroid_sim >= centroid_threshold or (best_frame >= frame_threshold and centroid_sim >= min_centroid):
            union_find.union(int(i), int(j))
            merged.append((keys[i], keys[j], centroid_sim, best_frame, matching))
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
    parser.add_argument("--centroid-threshold", type=float, default=None,
                        help=f"DINOv2 mean-embedding cosine at which two creatives merge "
                             f"(default: grouping.centroid_threshold = {grouping['centroid_threshold']})")
    parser.add_argument("--frame-threshold", type=float, default=None,
                        help=f"DINOv2 best frame-match cosine at which two creatives merge (if their centroid "
                             f"cosine is also >= grouping.frame_match_min_centroid); set both thresholds >1 to "
                             f"disable merging (default: grouping.frame_threshold = {grouping['frame_threshold']})")
    parser.add_argument("--non-ad-group-by", choices=("channel", "video"), default=None,
                        help=f"Non-ad group unit (default: grouping.non_ad_group_by = {grouping['non_ad_group_by']})")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Report groups and moves without touching files.")
    args = parser.parse_args()

    max_frames = args.max_frames if args.max_frames is not None else grouping["max_frames_per_creative"]
    centroid_threshold = args.centroid_threshold if args.centroid_threshold is not None \
        else grouping["centroid_threshold"]
    frame_threshold = args.frame_threshold if args.frame_threshold is not None else grouping["frame_threshold"]
    non_ad_group_by = args.non_ad_group_by or grouping["non_ad_group_by"]
    batch_size = args.batch_size if args.batch_size is not None else grouping["batch_size"]
    if max_frames < 0:
        sys.exit("Error: --max-frames must be >= 0.")
    if non_ad_group_by not in ("channel", "video"):
        sys.exit("Error: grouping.non_ad_group_by must be 'channel' or 'video'.")

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

    if (centroid_threshold <= 1.0 or frame_threshold <= 1.0) and ad_frames:
        frames = embed_frames([f[1] for f in ad_frames], [f[2] for f in ad_frames], batch_size,
                              grouping["model"], grouping["model_cache_dir"])
        root, merged = merge_creatives(frames, centroid_threshold, frame_threshold,
                                       grouping["frame_match_min_centroid"])
    else:
        root, merged = {}, []
    root = {f[2]: root.get(f[2], f[2]) for f in ad_frames}

    creative_keys = sorted({f[2] for f in ad_frames})
    group_ids = sorted(set(root[k] for k in creative_keys))
    group_name = {g: f"ad_{i:04d}" for i, g in enumerate(group_ids, 1)}
    print(f"Creatives: {len(creative_keys)} distinct normalized `nom`, {len(group_ids)} groups after DINOv2 merge "
          f"(centroid >= {centroid_threshold} or best frame >= {frame_threshold}; {len(merged)} merged pair(s)).")
    for a, b, centroid_sim, best_frame, matching in sorted(merged, key=lambda m: -m[3]):
        print(f"  merged (centroid {centroid_sim:.3f}, best frame {best_frame:.3f}, {matching} matching): "
              f"{a[:50]}  <->  {b[:50]}")

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
        rows.append({"image_name": name, "label": "non_ad", "group": non_ad_group(name, non_ad_group_by),
                     "creative": None, "capped": False})

    df = pd.DataFrame(rows, columns=["image_name", "label", "group", "creative", "capped"])
    active = df[~df["capped"]]
    sizes = active[active["label"] == "ad"].groupby("group").size()
    print(f"Capping ad groups at {max_frames or 'no limit'}: {len(moves)} frame(s) to move into "
          f"{result_dir / CAPPED_DIR}.")
    if len(sizes):
        print(f"Active ad frames per group after cap: min {sizes.min()}, median {int(sizes.median())}, "
              f"max {sizes.max()} ({len(active[active['label'] == 'ad'])} total).")
    print(f"Non-ad groups (one per {non_ad_group_by}): {df[df['label'] == 'non_ad']['group'].nunique()}.")

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
