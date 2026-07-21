import argparse
import os
import sys
from pathlib import Path


def check_folder(path: str, repo_root: Path) -> int:
    """Return 0 if valid, non-zero exit code on error.

    Checks that `path` exists and contains `ad` and `non_ad` subfolders.
    """
    if not os.path.isdir(path):
        print(f"Error: folder '{path}' does not exist or is not a directory.", file=sys.stderr)
        return 2

    missing = []
    for sub in ("ad", "non_ad"):
        if not os.path.isdir(os.path.join(path, sub)):
            missing.append(sub)

    if missing:
        print(f"Error: missing subfolder(s) in '{path}': {', '.join(missing)}", file=sys.stderr)
        return 3

    # check for metadata.xlsx at project root
    metadata_path = repo_root / "metadata.xlsx"
    if not metadata_path.is_file():
        print(f"Error: metadata.xlsx not found at project root: {metadata_path}", file=sys.stderr)
        return 4

    print(f"OK: found both 'ad' and 'non_ad' in '{path}' and metadata.xlsx at '{metadata_path}'")
    return 0


def main() -> None:
    # default to <repo_root>/dataset when no folder is provided
    repo_root = Path(__file__).resolve().parents[2]
    default_folder = str(repo_root / "dataset")

    parser = argparse.ArgumentParser(
        description="Check a dataset folder contains 'ad' and 'non_ad' subfolders."
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default=default_folder,
        help=f"Path to folder containing 'ad' and 'non_ad' (default: {default_folder})",
    )
    args = parser.parse_args()

    rc = check_folder(args.folder, Path(repo_root))
    if rc != 0:
        sys.exit(rc)

    # proceed to generate CSV mapping images to marketing strategies
    from .generator import generate_csv

    ad_dir = Path(args.folder) / "ad"
    metadata_path = Path(repo_root) / "metadata.xlsx"
    output_csv = Path(repo_root) / "marketing_multilabel_dataset.csv"

    gen_rc = generate_csv(ad_dir, metadata_path, output_csv)
    sys.exit(gen_rc)


if __name__ == "__main__":
    main()