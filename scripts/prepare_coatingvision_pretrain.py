from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
import yaml


CLASSIFICATION_TAGS = ("Surface_Crack", "Delamination", "Pinhole", "unclassified")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a conservative RGB pinhole-only YOLO subset from CoatingVision."
    )
    parser.add_argument("--source", type=Path, default=Path("dataset/CoatingVision"))
    parser.add_argument("--output", type=Path, default=Path("dataset/CoatingVision/pretrain_pinhole"))
    parser.add_argument("--val-fraction", type=float, default=0.2)
    return parser.parse_args()


def specimen_group(original_name: str) -> str:
    return re.split(r"_frame_", original_name, maxsplit=1)[0]


def read_detection_classes(label_path: Path) -> set[int]:
    classes = set()
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Malformed label {label_path}:{line_number}")
        try:
            values = [float(value) for value in fields]
            class_id = int(values[0])
        except ValueError as error:
            raise ValueError(f"Invalid numeric label {label_path}:{line_number}") from error
        if values[0] != class_id or class_id not in (0, 1):
            raise ValueError(f"Unexpected source class in {label_path}:{line_number}")
        if not all(math.isfinite(value) for value in values[1:]):
            raise ValueError(f"Non-finite box coordinate in {label_path}:{line_number}")
        if any(value < 0 or value > 1 for value in values[1:]) or values[3] <= 0 or values[4] <= 0:
            raise ValueError(f"Out-of-range box in {label_path}:{line_number}")
        classes.add(class_id)
    return classes


def choose_validation_group(groups: dict[str, list[str]], val_fraction: float) -> str:
    if not 0 < val_fraction < 1:
        raise ValueError("--val-fraction must be between 0 and 1")
    target = sum(map(len, groups.values())) * val_fraction
    return min(groups, key=lambda name: (abs(len(groups[name]) - target), name))


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    classification_path = source / "classification" / "labels.csv"
    images_dir = source / "detection" / "images"
    labels_dir = source / "detection" / "labels"
    for required in (classification_path, images_dir, labels_dir):
        if not required.exists():
            raise FileNotFoundError(f"Missing CoatingVision input: {required}")
    if output.exists():
        raise FileExistsError(f"Output already exists; move it aside before preparing again: {output}")

    with classification_path.open("r", newline="", encoding="utf-8-sig") as file:
        rows = {row["file_name"]: row for row in csv.DictReader(file)}

    groups: dict[str, list[str]] = defaultdict(list)
    for label_path in sorted(labels_dir.glob("*.txt")):
        filename = f"{label_path.stem}.jpg"
        row = rows.get(filename)
        if row is None:
            continue
        tags = {tag for tag in CLASSIFICATION_TAGS if row.get(tag) == "1"}
        if tags != {"Pinhole"}:
            continue
        if read_detection_classes(label_path) != {0}:
            continue
        original_name = row.get("original_file_name", "")
        if not original_name:
            raise ValueError(f"Missing source grouping metadata for {filename}")
        groups[specimen_group(original_name)].append(filename)

    if len(groups) < 2:
        raise ValueError("Need at least two acquisition groups for a group-held-out validation split")
    validation_group = choose_validation_group(groups, args.val_fraction)
    output.mkdir(parents=True)

    split_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    image_manifest = []
    for group, filenames in sorted(groups.items()):
        split = "val" if group == validation_group else "train"
        for filename in sorted(filenames):
            image_source = images_dir / filename
            label_source = labels_dir / f"{Path(filename).stem}.txt"
            if not image_source.is_file():
                raise FileNotFoundError(f"Missing image for detection label: {image_source}")
            image_dir = output / "images" / split
            label_dir = output / "labels" / split
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            with Image.open(image_source) as image:
                mode_counts[image.mode] += 1
                if image.mode == "RGB":
                    shutil.copy2(image_source, image_dir / filename)
                else:
                    image.convert("RGB").save(image_dir / filename, quality=95)

            normalized_labels = []
            for line in label_source.read_text(encoding="utf-8-sig").splitlines():
                if line.strip():
                    _, *box = line.split()
                    normalized_labels.append("0 " + " ".join(box))
            (label_dir / f"{Path(filename).stem}.txt").write_text(
                "\n".join(normalized_labels) + "\n", encoding="utf-8"
            )
            split_counts[split] += 1
            image_manifest.append(
                {"file": filename, "split": split, "group": group, "source": str(image_source)}
            )

    data_config = {
        "path": output.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "electrode_pinhole"},
    }
    (output / "data.yaml").write_text(
        yaml.safe_dump(data_config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    manifest = {
        "source": str(source),
        "selection": "classification tags exactly Pinhole and detection labels contain class 0 only",
        "class_mapping": "source detection class 0 -> electrode_pinhole; class 1 (unclassified) excluded",
        "grouping": "prefix of original_file_name before _frame_",
        "validation_group": validation_group,
        "counts": dict(split_counts),
        "source_image_modes": dict(mode_counts),
        "class_names": ["electrode_pinhole"],
        "images": image_manifest,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Prepared {sum(split_counts.values())} images at {output}")
    print(f"Train: {split_counts['train']}; validation: {split_counts['val']}")
    print(f"Validation acquisition group: {validation_group}")
    print(f"Source image modes: {dict(mode_counts)}")
    print(f"Data config: {output / 'data.yaml'}")


if __name__ == "__main__":
    main()
