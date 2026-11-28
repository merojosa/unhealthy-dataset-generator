"""Pack the binary, tip-multiclass and multilabel datasets into uncompressed zips at the repo root.

binary-dataset.zip:            ad/, non_ad/, groups.csv          (from {path.dataset}/result)
tip-multiclass-dataset.zip:    images/, tip_multiclass_dataset.csv
                               [+ augmented/, tip_multiclass_augmented.csv if present]
multi-strategies-dataset.zip:  images/, marketing_multilabel_dataset.csv
                               [+ augmented/, marketing_multilabel_augmented.csv if present]
"""
import json
import zipfile
from pathlib import Path


def load_config() -> dict:
    try:
        with open("config.json") as f:
            return json.load(f)
    except FileNotFoundError:
        with open("default_config.json") as f:
            return json.load(f)


def pack(zip_name: str, entries: list) -> None:
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_STORED) as z:
        for src, arc in entries:
            src = Path(src)
            if src.is_dir():
                for f in sorted(src.rglob("*.jpg")):
                    z.write(f, f"{arc}/{f.relative_to(src).as_posix()}")
            elif src.is_file():
                z.write(src, arc)
    with zipfile.ZipFile(zip_name) as z:
        print(f"{zip_name}: {len(z.namelist())} entries")


def main() -> None:
    config = load_config()
    binary = Path(config["path"]["dataset"]) / "result"
    multiclass = Path(config["path"]["multiclass_dataset"])
    multilabel = Path(config["path"]["multilabel_dataset"])
    pack("binary-dataset.zip", [(binary / "ad", "ad"), (binary / "non_ad", "non_ad"),
                                (binary / "groups.csv", "groups.csv")])
    pack("tip-multiclass-dataset.zip", [
        (multiclass / "images", "images"),
        (multiclass / "tip_multiclass_dataset.csv", "tip_multiclass_dataset.csv"),
        (multiclass / "augmented", "augmented"),
        (multiclass / "tip_multiclass_augmented.csv", "tip_multiclass_augmented.csv"),
    ])
    pack("multi-strategies-dataset.zip", [
        (multilabel / "images", "images"),
        (multilabel / "marketing_multilabel_dataset.csv", "marketing_multilabel_dataset.csv"),
        (multilabel / "augmented", "augmented"),
        (multilabel / "marketing_multilabel_augmented.csv", "marketing_multilabel_augmented.csv"),
    ])


if __name__ == "__main__":
    main()
