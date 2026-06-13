from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import argparse
import json


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Same as main.py: look for config.json in the CWD and, if it isn't there,
# fall back to default_config.json (assumed to always be present).
# Returns the "duplicate_images_removal" block.
def load_similarity_config() -> dict:
    try:
        with open("config.json") as f:
            config = json.load(f)
    except FileNotFoundError:
        with open("default_config.json") as f:
            config = json.load(f)

    return config["duplicate_images_removal"]

# Recurse into the folder and return a lexicographically sorted list of
# image paths whose extension is a valid one.
def load_image_paths(folder: str) -> list[Path]:
    folder_path = Path(folder)
    # In case the path is wrong.
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    # Return every image path in that folder, lexicographically sorted.
    return sorted(
        path for path in folder_path.rglob("*")
        if path.suffix.lower() in IMAGE_EXTENSIONS
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

# This just appends results to a csv.
# On the first write a header is added, otherwise it isn't.
def append_rows_to_csv(
    rows: list[dict],
    columns: list[str],
    output: str,
    first_write: bool,
) -> bool:

    if not rows:
        return first_write

    output_path = Path(output)

    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows, columns=columns)

    df.to_csv(
        output,
        mode="w" if first_write else "a",
        header=first_write,
        index=False,
    )

    return False

# This just creates an empty csv in case nothing matches the expected
# similarity.
def create_empty_csv(output: str, columns: list[str]) -> None:
    output_path = Path(output)

    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(columns=columns).to_csv(output, index=False)

# This compares each image against all the ones that come after it.
def find_similar_pairs_three_methods_chunked(
    image_paths: list[Path],
    embeddings: np.ndarray,
    hashes: list[int],
    clip_threshold: float,
    hash_max_distance: int,
    combined_threshold: float,
    clip_weight: float,
    clip_output: str,
    hash_output: str,
    combined_output: str,
    chunk_size: int = 512,
    hash_size: int = 8,
) -> None:

    n = len(image_paths)
    hash_bits = hash_size * hash_size
    # There will be three comparisons: CLIP, HASH, and the combined one.

    clip_columns = [
        "image_1",
        "image_2",
        "similarity",
        "equal",
    ]

    hash_columns = [
        "image_1",
        "image_2",
        "hash_distance",
        "hash_similarity",
        "equal",
    ]

    combined_columns = [
        "image_1",
        "image_2",
        "clip_similarity",
        "hash_distance",
        "hash_similarity",
        "combined_score",
        "equal",
    ]

    first_clip_write = True
    first_hash_write = True
    first_combined_write = True

    for start in tqdm(range(0, n, chunk_size), desc="Comparing images"):
        end = min(start + chunk_size, n)

        # Since the embeddings are normalized, the dot product is the cosine similarity.
        sim_block = embeddings[start:end] @ embeddings.T

        clip_rows = []
        hash_rows = []
        combined_rows = []

        for local_i in range(end - start):
            i = start + local_i

            # Only check j > i so we don't repeat pairs or compare an image with itself.
            for j in range(i + 1, n):
                clip_similarity = float(sim_block[local_i, j])

                distance = hamming_distance(hashes[i], hashes[j])
                hash_similarity = 1.0 - (distance / hash_bits)
                # The combined score.
                combined_score = (
                    clip_weight * clip_similarity
                    + (1.0 - clip_weight) * hash_similarity
                )

                if clip_similarity >= clip_threshold:
                    clip_rows.append({
                        "image_1": str(image_paths[i]),
                        "image_2": str(image_paths[j]),
                        "similarity": clip_similarity,
                        "equal": True,
                    })

                if distance <= hash_max_distance:
                    hash_rows.append({
                        "image_1": str(image_paths[i]),
                        "image_2": str(image_paths[j]),
                        "hash_distance": distance,
                        "hash_similarity": hash_similarity,
                        "equal": True,
                    })

                if combined_score >= combined_threshold:
                    combined_rows.append({
                        "image_1": str(image_paths[i]),
                        "image_2": str(image_paths[j]),
                        "clip_similarity": clip_similarity,
                        "hash_distance": distance,
                        "hash_similarity": hash_similarity,
                        "combined_score": combined_score,
                        "equal": True,
                    })

        first_clip_write = append_rows_to_csv(
            clip_rows,
            clip_columns,
            clip_output,
            first_clip_write,
        )

        first_hash_write = append_rows_to_csv(
            hash_rows,
            hash_columns,
            hash_output,
            first_hash_write,
        )

        first_combined_write = append_rows_to_csv(
            combined_rows,
            combined_columns,
            combined_output,
            first_combined_write,
        )
    # first_clip_write stays True iff nothing was appended.
    # Same for the others.
    if first_clip_write:
        create_empty_csv(clip_output, clip_columns)

    if first_hash_write:
        create_empty_csv(hash_output, hash_columns)

    if first_combined_write:
        create_empty_csv(combined_output, combined_columns)

# This just builds the correct output file name.
def output_name_from_prefix(prefix: str, suffix: str) -> str:
    path = Path(prefix)

    if path.suffix == ".csv":
        path = path.with_suffix("")

    return str(path.with_name(path.name + suffix))


def main():
    parser = argparse.ArgumentParser()

    # Defaults come from the config (config.json or default_config.json).
    # Each flag starts as None: if the user doesn't pass it we use the config
    # value; if they do, the flag wins.
    config = load_similarity_config()

    parser.add_argument("folder", help="Folder with images")
    # CLIP
    # NOTE: the threshold is the minimum similarity to consider two images equal.
    parser.add_argument("--clip-threshold", type=float, default=None)

    # Perceptual hash
    # Max allowed hamming distance (the lower, the more similar).
    parser.add_argument("--hash-max-distance", type=int, default=None)
    parser.add_argument("--hash-size", type=int, default=None)

    # Combination
    # clip-weight splits the weight between CLIP and hash.
    parser.add_argument("--combined-threshold", type=float, default=None)
    parser.add_argument("--clip-weight", type=float, default=None)

    # Outputs
    parser.add_argument("--output-prefix", default="similarity_report")
    parser.add_argument("--clip-output", default=None)
    parser.add_argument("--hash-output", default=None)
    parser.add_argument("--combined-output", default=None)

    # Performance
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)

    args = parser.parse_args()

    # Resolve each parameter: the CLI flag if it was passed, otherwise the config.
    clip_threshold = args.clip_threshold if args.clip_threshold is not None else config["clip_threshold"]
    hash_max_distance = args.hash_max_distance if args.hash_max_distance is not None else config["hash_max_distance"]
    hash_size = args.hash_size if args.hash_size is not None else config["hash_size"]
    combined_threshold = args.combined_threshold if args.combined_threshold is not None else config["combined_threshold"]
    clip_weight = args.clip_weight if args.clip_weight is not None else config["clip_weight"]
    batch_size = args.batch_size if args.batch_size is not None else config["batch_size"]
    chunk_size = args.chunk_size if args.chunk_size is not None else config["chunk_size"]

    # Make sure the weight is a valid proportion.
    if not (0.0 <= clip_weight <= 1.0):
        raise ValueError("clip_weight must be between 0 and 1.")

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

    clip_output = args.clip_output or output_name_from_prefix(
        args.output_prefix,
        "_clip.csv",
    )

    hash_output = args.hash_output or output_name_from_prefix(
        args.output_prefix,
        "_hash.csv",
    )

    combined_output = args.combined_output or output_name_from_prefix(
        args.output_prefix,
        "_combined.csv",
    )

    find_similar_pairs_three_methods_chunked(
        image_paths=image_paths,
        embeddings=embeddings,
        hashes=hashes,
        clip_threshold=clip_threshold,
        hash_max_distance=hash_max_distance,
        combined_threshold=combined_threshold,
        clip_weight=clip_weight,
        clip_output=clip_output,
        hash_output=hash_output,
        combined_output=combined_output,
        chunk_size=chunk_size,
        hash_size=hash_size,
    )

    print(f"CLIP report saved to: {clip_output}")
    print(f"Hash report saved to: {hash_output}")
    print(f"Combined report saved to: {combined_output}")


if __name__ == "__main__":
    main()
