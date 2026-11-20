import argparse
import json
import os
import sys
from pathlib import Path


def load_config() -> dict:
    try:
        with open("config.json") as f:
            return json.load(f)
    except FileNotFoundError:
        with open("default_config.json") as f:
            return json.load(f)


def check_folder(path_str: str) -> int:
    """Return 0 if the binary dataset root has `result/ad` and `metadata.xlsx`, non-zero otherwise."""
    if not os.path.isdir(path_str):
        print(f"Error: folder '{path_str}' does not exist or is not a directory.", file=sys.stderr)
        return 2

    path = Path(path_str)
    if not (path / "result" / "ad").is_dir():
        print(f"Error: folder '{path}' does not contain a 'result/ad' subfolder.", file=sys.stderr)
        return 3

    metadata_path = path / "metadata.xlsx"
    if not metadata_path.is_file():
        print(f"Error: metadata.xlsx not found at dataset root: {metadata_path}", file=sys.stderr)
        return 4

    print(f"OK: found 'result/ad' in '{path}' and metadata.xlsx at '{metadata_path}'")
    return 0


def main() -> None:
    config = load_config()
    default_input = config["path"]["dataset"]
    default_output = config["path"]["multiclass_dataset"]

    parser = argparse.ArgumentParser(
        description="Build the tip multiclass dataset from the binary dataset's ad frames."
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default=default_input,
        help=f"Binary dataset root containing metadata.xlsx and result/ad (default: path.dataset = {default_input})",
    )
    default_ratio = config["multiclass"]["non_ad_ratio"]
    parser.add_argument(
        "--non-ad-ratio",
        type=float,
        default=default_ratio,
        help=f"Non-ad class (tip 0) size as a multiple of the largest ad class; 0 disables it (default: multiclass.non_ad_ratio = {default_ratio})",
    )
    parser.add_argument(
        "--output",
        default=default_output,
        help=f"Output dataset root (default: path.multiclass_dataset = {default_output})",
    )
    args = parser.parse_args()

    if args.non_ad_ratio < 0:
        print("Error: --non-ad-ratio must be >= 0.", file=sys.stderr)
        sys.exit(6)

    if Path(args.output).resolve() == Path(args.folder).resolve():
        print("Error: output folder must differ from the binary dataset folder.", file=sys.stderr)
        sys.exit(5)

    rc = check_folder(args.folder)
    if rc != 0:
        sys.exit(rc)

    try:
        from .generator import generate_dataset
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from generator import generate_dataset

    folder = Path(args.folder)
    sys.exit(generate_dataset(
        folder / "result" / "ad",
        folder / "metadata.xlsx",
        args.output,
        tip_values=config["tip_values"],
        non_ad_images_dir=folder / "result" / "non_ad",
        non_ad_ratio=args.non_ad_ratio,
        groups_csv=folder / "result" / "groups.csv",
    ))


if __name__ == "__main__":
    main()
