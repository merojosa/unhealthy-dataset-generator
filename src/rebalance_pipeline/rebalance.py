from pathlib import Path
import numpy as np
import pandas as pd
import argparse
import json
import shutil


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Subfolders under result/ that hold frames already pulled out of the active
# dataset (by the dedup pipeline, the review GUI, or a previous rebalance run).
# They must never be counted or moved again.
EXCLUDED_DIRS = {"discarded", "duplicates", "excess"}


# Same as the other pipelines: look for config.json in the CWD and, if it isn't
# there, fall back to default_config.json (assumed to always be present).
def load_config() -> dict:
    try:
        with open("config.json") as f:
            return json.load(f)
    except FileNotFoundError:
        with open("default_config.json") as f:
            return json.load(f)


# Every image directly relevant to a class folder (result/ad or result/non_ad),
# lexicographically sorted. Anything under an excluded subfolder is skipped so a
# re-run never re-counts frames that were already pulled out.
def load_class_paths(class_dir: Path) -> list[Path]:
    if not class_dir.exists():
        return []
    return sorted(
        path for path in class_dir.rglob("*")
        if path.suffix.lower() in IMAGE_EXTENSIONS
        and not EXCLUDED_DIRS.intersection(path.parts)
    )


# Pick `keep` evenly-spaced indices out of `total` while preserving the sorted
# order. floor(k * total / keep) for k in [0, keep) yields exactly `keep`
# strictly-increasing indices whenever keep <= total, so the kept frames stay
# spread across the whole (video/time-ordered) sequence instead of clustering.
def evenly_spaced_keep_indices(total: int, keep: int) -> np.ndarray:
    if keep >= total:
        return np.arange(total)
    return (np.arange(keep) * total) // keep


# Decide which class is the majority and how many of its frames to move so the
# non_ad : ad ratio lands at target_ratio. We only ever trim the larger side
# (never duplicate the smaller one), so balancing is loss-only and reversible.
#
# Returns (class_to_trim, target_count_for_that_class) or (None, 0) if already
# balanced within rounding.
def plan_trim(ad_count: int, non_ad_count: int, target_ratio: float):
    desired_non_ad = round(ad_count * target_ratio)

    if non_ad_count > desired_non_ad:
        return "non_ad", desired_non_ad

    # non_ad is at or below target: balance by trimming ad instead so that
    # non_ad / ad == target_ratio.
    desired_ad = round(non_ad_count / target_ratio) if target_ratio > 0 else ad_count
    if ad_count > desired_ad:
        return "ad", desired_ad

    return None, 0


# Move the frames that didn't make the evenly-spaced cut into excess_dir,
# preserving each frame's path relative to result/ so a frame at
# result/non_ad/x.jpg lands at result/excess/non_ad/x.jpg.
#
# Returns the list of (original_path, moved_to) tuples (relative paths kept
# relative to result_dir for a readable manifest).
def move_excess(
    paths: list[Path],
    keep_indices: np.ndarray,
    result_dir: Path,
    excess_dir: Path,
    dry_run: bool,
) -> list[tuple[Path, Path]]:

    keep = set(int(i) for i in keep_indices)
    moves: list[tuple[Path, Path]] = []

    for idx, source in enumerate(paths):
        if idx in keep:
            continue

        relative = source.relative_to(result_dir)
        destination = excess_dir / relative

        if not dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Idempotent like the dedup pipeline: if a previous run already
            # parked this frame, just drop any stray source copy.
            if destination.exists():
                source.unlink(missing_ok=True)
            elif source.exists():
                shutil.move(str(source), str(destination))

        moves.append((relative, destination))

    return moves


# Small auditable, reversible manifest of what was moved out to rebalance.
def write_manifest(moves: list[tuple[Path, Path]], excess_dir: Path) -> Path:
    manifest_path = excess_dir / "rebalance_manifest.csv"
    excess_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [{"original_path": str(original), "moved_to": str(moved_to)}
         for original, moved_to in moves],
        columns=["original_path", "moved_to"],
    ).to_csv(manifest_path, index=False)

    return manifest_path


def main():
    parser = argparse.ArgumentParser(
        description="Balance result/ad and result/non_ad by moving the surplus "
                    "of the majority class into result/excess/."
    )

    config = load_config()
    rebalance_config = config.get("rebalance", {})
    # Default folder: {dataset}/result, where the pipeline writes ad/ and
    # non_ad/. Override by passing a folder explicitly.
    default_result = str(Path(config["path"]["dataset"]) / "result")

    parser.add_argument(
        "result",
        nargs="?",
        default=default_result,
        help=f"result/ folder containing ad/ and non_ad/ (default: {default_result}).",
    )
    parser.add_argument(
        "--excess-dir",
        default=None,
        help="Where to move surplus frames (default: <result>/excess).",
    )
    # non_ad frames per ad frame. 1.0 => a perfectly balanced 1:1 dataset.
    parser.add_argument("--target-ratio", type=float, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be moved without moving anything.",
    )

    args = parser.parse_args()

    target_ratio = (
        args.target_ratio if args.target_ratio is not None
        else rebalance_config.get("target_ratio", 1.0)
    )
    if target_ratio <= 0:
        raise ValueError("target_ratio must be greater than 0.")

    result_dir = Path(args.result)
    excess_dir = Path(args.excess_dir) if args.excess_dir else result_dir / "excess"

    ad_paths = load_class_paths(result_dir / "ad")
    non_ad_paths = load_class_paths(result_dir / "non_ad")
    ad_count, non_ad_count = len(ad_paths), len(non_ad_paths)

    print(f"ad:     {ad_count}")
    print(f"non_ad: {non_ad_count}")

    class_to_trim, target_count = plan_trim(ad_count, non_ad_count, target_ratio)

    if class_to_trim is None:
        print(f"\nAlready balanced at ratio {target_ratio} (within rounding). "
              "Nothing to do.")
        return

    paths = non_ad_paths if class_to_trim == "non_ad" else ad_paths
    keep_indices = evenly_spaced_keep_indices(len(paths), target_count)

    moves = move_excess(
        paths=paths,
        keep_indices=keep_indices,
        result_dir=result_dir,
        excess_dir=excess_dir,
        dry_run=args.dry_run,
    )

    kept = len(paths) - len(moves)
    if args.dry_run:
        print(f"\n[DRY RUN] Would move {len(moves)} '{class_to_trim}' frame(s) "
              f"into {excess_dir}, keeping {kept} (target {target_count}).")
        for original, _ in moves[:20]:
            print(f"  {original}")
        if len(moves) > 20:
            print(f"  ... and {len(moves) - 20} more.")
        return

    manifest_path = write_manifest(moves, excess_dir)

    print(f"\nMoved {len(moves)} '{class_to_trim}' frame(s) into {excess_dir}.")
    print(f"Balanced to ad:{ad_count if class_to_trim == 'non_ad' else target_count} "
          f"/ non_ad:{target_count if class_to_trim == 'non_ad' else non_ad_count} "
          f"(ratio {target_ratio}).")
    print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
