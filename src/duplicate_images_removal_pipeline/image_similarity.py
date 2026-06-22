from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import argparse
import json
import shutil


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Subfolders under result/ that hold frames already pulled out of the dataset.
# They must never be scanned (we don't want to compare against, or move, frames
# that were already discarded or marked duplicate in a previous run).
EXCLUDED_DIRS = {"discarded", "duplicates"}

# Same as main.py: look for config.json in the CWD and, if it isn't there,
# fall back to default_config.json (assumed to always be present).
# Returns the whole config dict.
def load_config() -> dict:
    try:
        with open("config.json") as f:
            return json.load(f)
    except FileNotFoundError:
        with open("default_config.json") as f:
            return json.load(f)

# Convenience accessor for the "duplicate_images_removal" block.
def load_similarity_config() -> dict:
    return load_config()["duplicate_images_removal"]

# Recurse into the folder and return a lexicographically sorted list of
# image paths whose extension is a valid one.
def load_image_paths(folder: str) -> list[Path]:
    folder_path = Path(folder)
    # In case the path is wrong.
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    # Return every image path in that folder, lexicographically sorted.
    # Anything under a "discarded" or "duplicates" directory is skipped:
    # those frames were already pulled out of the active dataset, so they must
    # not be compared against (or moved like) the kept ones.
    return sorted(
        path for path in folder_path.rglob("*")
        if path.suffix.lower() in IMAGE_EXTENSIONS
        and not EXCLUDED_DIRS.intersection(path.parts)
    )

# The goal here is to handle every image in RGB format.
def load_image(path: Path) -> Image.Image | None:
    try:
        # What img.convert("RGB") does is assign every pixel an RGB value
        # (three sequential bytes) and return the "vector" of RGBs
        # (the image's equivalent of a vector).
        with Image.open(path) as img:
            return img.convert("RGB")
    except Exception as error:
        print(f"Could not read {path}: {error}")
        return None

# The idea behind perceptual hashes is that two "similar" images produce
# similar hashes.
# With a difference hash, you compare the brightness changes between adjacent
# pixels (two pixels are adjacent iff one is to the left of the other).
# perceptual_dhash is used because it's a sweet spot between effectiveness
# and speed.
def perceptual_dhash(image: Image.Image, hash_size: int = 8) -> int:

    try:
        resample_filter = Image.Resampling.LANCZOS
    except AttributeError:
        resample_filter = Image.LANCZOS
    # NOTE: perceptual_dhash always works on grayscale images.
    gray = image.convert("L").resize(
        (hash_size + 1, hash_size),
        resample_filter,
    )

    pixels = np.asarray(gray, dtype=np.int16)

    # Compare each pixel with the one to its right.
    differences = pixels[:, 1:] > pixels[:, :-1]

    hash_value = 0

    for bit in differences.flatten():
        hash_value = (hash_value << 1) | int(bit)

    return hash_value

# This just tells me how many bits differ between two hashes.
# Note that (obviously) this can be obtained with a XOR.
def hamming_distance(hash_1: int, hash_2: int) -> int:
    return (hash_1 ^ hash_2).bit_count()

# For each image this:
# 1. Loads it
# 2. Computes the perceptual hash
# 3. Computes the CLIP embedding
# and along the way discards images that, for whatever reason, couldn't be opened.
# Returns a tuple of (valid_paths, embeddings, hashes).
def compute_embeddings_and_hashes_from_paths(
    model,
    image_paths: list[Path],
    batch_size: int = 32,
    hash_size: int = 8,
) -> tuple[list[Path], np.ndarray, list[int]]:

    valid_paths = []
    all_embeddings = []
    all_hashes = []
    # Images are processed in batches of size 32.
    for start in tqdm(range(0, len(image_paths), batch_size), desc="Computing embeddings and hashes"):
        # Grab the paths for the current batch.
        batch_paths = image_paths[start:start + batch_size]

        images = []
        paths_that_loaded = []
        hashes_that_loaded = []

        for path in batch_paths:
            image = load_image(path)

            if image is not None:
                image_hash = perceptual_dhash(image, hash_size=hash_size)

                images.append(image)
                paths_that_loaded.append(path)
                hashes_that_loaded.append(image_hash)

        if len(images) == 0:
            continue
        # Normalize the vectors so that the dot product of any two is the
        # cosine similarity.
        embeddings = model.encode(
            images,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        all_embeddings.append(embeddings.astype(np.float32))
        valid_paths.extend(paths_that_loaded)
        all_hashes.extend(hashes_that_loaded)

        for image in images:
            image.close()

    if len(all_embeddings) == 0:
        return [], np.empty((0, 0), dtype=np.float32), []

    return valid_paths, np.vstack(all_embeddings), all_hashes


# Union-Find / disjoint set. Used to fold the pairwise "these two are
# duplicates" relations into connected groups: if A~B and B~C we want
# {A, B, C} in one group even when A and C weren't compared as equal.
class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # Path compression so repeated finds stay near O(1).
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            return
        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1

    # Return {representative_index: [member indices]} for every group, in a
    # deterministic order. Members keep their original (sorted) order.
    def groups(self) -> dict[int, list[int]]:
        result: dict[int, list[int]] = {}
        for i in range(len(self.parent)):
            result.setdefault(self.find(i), []).append(i)
        return result


# Compares each image against all the ones that come after it and unions any
# pair whose *combined* score (CLIP cosine + perceptual-hash similarity, mixed
# by clip_weight) reaches combined_threshold. The chunked matmul keeps the
# CLIP similarity computation vectorized while the union-find accumulates the
# duplicate groups without ever materializing the full O(n^2) pair list.
def find_duplicate_groups(
    embeddings: np.ndarray,
    hashes: list[int],
    combined_threshold: float,
    clip_weight: float,
    chunk_size: int = 512,
    hash_size: int = 8,
) -> dict[int, list[int]]:

    n = len(hashes)
    hash_bits = hash_size * hash_size
    union_find = UnionFind(n)

    for start in tqdm(range(0, n, chunk_size), desc="Comparing images"):
        end = min(start + chunk_size, n)

        # Since the embeddings are normalized, the dot product is the cosine
        # similarity.
        sim_block = embeddings[start:end] @ embeddings.T

        for local_i in range(end - start):
            i = start + local_i

            # Only check j > i so we don't repeat pairs or compare an image
            # with itself.
            for j in range(i + 1, n):
                clip_similarity = float(sim_block[local_i, j])

                distance = hamming_distance(hashes[i], hashes[j])
                hash_similarity = 1.0 - (distance / hash_bits)

                combined_score = (
                    clip_weight * clip_similarity
                    + (1.0 - clip_weight) * hash_similarity
                )

                if combined_score >= combined_threshold:
                    union_find.union(i, j)

    return union_find.groups()


# Given the duplicate groups, keep the lexicographically-first image of each
# group (the dataset's natural ordering) and move every other member into
# duplicates_dir, preserving its path relative to the scanned folder so a frame
# at result/ad/x.jpg lands at result/duplicates/ad/x.jpg.
#
# Returns the list of (moved_from, moved_to, kept) tuples (relative paths).
def move_duplicates(
    image_paths: list[Path],
    groups: dict[int, list[int]],
    folder: Path,
    duplicates_dir: Path,
    dry_run: bool,
) -> list[tuple[Path, Path, Path]]:

    moves: list[tuple[Path, Path, Path]] = []

    for members in groups.values():
        if len(members) < 2:
            continue
        # Keep the smallest path; move the rest. members already follow the
        # sorted order of image_paths, so members[0] is the keeper.
        keeper = image_paths[members[0]]

        for idx in members[1:]:
            source = image_paths[idx]
            relative = source.relative_to(folder)
            destination = duplicates_dir / relative

            if not dry_run:
                destination.parent.mkdir(parents=True, exist_ok=True)
                # If a previous run already moved this exact frame, just drop
                # the stray copy instead of crashing on an existing target.
                # missing_ok guards the case where the source is also gone
                # (e.g. a prior run already moved it and left nothing behind).
                if destination.exists():
                    source.unlink(missing_ok=True)
                elif source.exists():
                    shutil.move(str(source), str(destination))
                else:
                    # Nothing to move: the frame is already at its destination
                    # or vanished between scanning and moving. Skip it.
                    continue

            moves.append((relative, destination, keeper))

    return moves


# Small manifest so the moves are auditable and reversible. Lives inside the
# duplicates folder (it is not the old similarity "report" — it only records
# what was physically moved and which frame was kept in its place).
def write_manifest(moves: list[tuple[Path, Path, Path]], duplicates_dir: Path) -> Path:
    manifest_path = duplicates_dir / "duplicates_manifest.csv"
    duplicates_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "moved_to": str(moved_to),
            "original_path": str(moved_from),
            "kept": str(kept),
        }
        for moved_from, moved_to, kept in moves
    ]

    pd.DataFrame(
        rows,
        columns=["moved_to", "original_path", "kept"],
    ).to_csv(manifest_path, index=False)

    return manifest_path


def main():
    parser = argparse.ArgumentParser(
        description="Move duplicate frames out of the dataset into result/duplicates/."
    )

    config = load_config()
    similarity_config = config["duplicate_images_removal"]
    # Default folder to scan: {dataset}/result, where the pipeline writes
    # ad/ and non_ad/. Override by passing a folder explicitly.
    default_folder = str(Path(config["path"]["dataset"]) / "result")

    # Defaults come from the config (config.json or default_config.json).
    # Each flag starts as None: if the user doesn't pass it we use the config
    # value; if they do, the flag wins.
    parser.add_argument(
        "folder",
        nargs="?",
        default=default_folder,
        help=f"Folder with images (default: {default_folder})",
    )
    parser.add_argument(
        "--duplicates-dir",
        default=None,
        help="Where to move duplicates (default: <folder>/duplicates).",
    )

    # Combination — this is what decides duplicates now.
    # clip-weight splits the weight between CLIP and hash.
    parser.add_argument("--combined-threshold", type=float, default=None)
    parser.add_argument("--clip-weight", type=float, default=None)
    parser.add_argument("--hash-size", type=int, default=None)

    # Performance
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)

    # Preview without touching any files.
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be moved without moving anything.",
    )

    args = parser.parse_args()

    # Resolve each parameter: the CLI flag if it was passed, otherwise the config.
    combined_threshold = args.combined_threshold if args.combined_threshold is not None else similarity_config["combined_threshold"]
    clip_weight = args.clip_weight if args.clip_weight is not None else similarity_config["clip_weight"]
    hash_size = args.hash_size if args.hash_size is not None else similarity_config["hash_size"]
    batch_size = args.batch_size if args.batch_size is not None else similarity_config["batch_size"]
    chunk_size = args.chunk_size if args.chunk_size is not None else similarity_config["chunk_size"]

    # Make sure the weight is a valid proportion.
    if not (0.0 <= clip_weight <= 1.0):
        raise ValueError("clip_weight must be between 0 and 1.")

    folder = Path(args.folder)
    duplicates_dir = Path(args.duplicates_dir) if args.duplicates_dir else folder / "duplicates"

    image_paths = load_image_paths(args.folder)
    print(f"Found {len(image_paths)} images.")
    # Use the latest CLIP version, which is specifically good at image
    # similarity.
    model = SentenceTransformer("clip-ViT-B-32")

    image_paths, embeddings, hashes = compute_embeddings_and_hashes_from_paths(
        model,
        image_paths,
        batch_size=batch_size,
        hash_size=hash_size,
    )

    print(f"Processed {len(image_paths)} images.")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Computed {len(hashes)} perceptual hashes.")

    if len(image_paths) == 0:
        print("Could not load any image.")
        return

    groups = find_duplicate_groups(
        embeddings=embeddings,
        hashes=hashes,
        combined_threshold=combined_threshold,
        clip_weight=clip_weight,
        chunk_size=chunk_size,
        hash_size=hash_size,
    )

    moves = move_duplicates(
        image_paths=image_paths,
        groups=groups,
        folder=folder,
        duplicates_dir=duplicates_dir,
        dry_run=args.dry_run,
    )

    duplicate_groups = sum(1 for members in groups.values() if len(members) > 1)
    kept = len(image_paths) - len(moves)

    if args.dry_run:
        print(f"\n[DRY RUN] Would move {len(moves)} duplicate(s) from "
              f"{duplicate_groups} group(s) into {duplicates_dir}.")
        for moved_from, _, kept_path in moves[:20]:
            print(f"  {moved_from}  ->  duplicate of  {kept_path}")
        if len(moves) > 20:
            print(f"  ... and {len(moves) - 20} more.")
        print(f"{kept} unique image(s) would remain.")
        return

    manifest_path = write_manifest(moves, duplicates_dir)

    print(f"\nMoved {len(moves)} duplicate(s) from {duplicate_groups} group(s) "
          f"into {duplicates_dir}.")
    print(f"{kept} unique image(s) remain in {folder}.")
    print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
