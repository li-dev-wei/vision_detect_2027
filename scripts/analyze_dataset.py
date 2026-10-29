from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import yaml
from PIL import Image, UnidentifiedImageError


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SIZE_THRESHOLDS = (4, 8, 16, 32)


@dataclass
class SplitSummary:
    images: int = 0
    instances: int = 0
    background_images: int = 0
    missing_labels: int = 0
    malformed_labels: int = 0
    unreadable_images: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a YOLO detection dataset before training.")
    parser.add_argument("--data", type=Path, required=True, help="Path to the dataset YAML file.")
    parser.add_argument("--imgsz", type=int, default=1024, help="Training size used for resized-box statistics.")
    parser.add_argument("--check-duplicates", action="store_true", help="Hash images across dataset splits.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path.")
    return parser.parse_args()


def load_dataset_config(data_path: Path) -> tuple[Path, dict, dict[int, str]]:
    with data_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Dataset YAML must contain a mapping.")

    configured_root = Path(config.get("path", data_path.parent))
    if not configured_root.is_absolute():
        configured_root = Path.cwd() / configured_root
    dataset_root = configured_root.resolve()

    raw_names = config.get("names", {})
    if isinstance(raw_names, list):
        names = {index: str(name) for index, name in enumerate(raw_names)}
    elif isinstance(raw_names, dict):
        names = {int(index): str(name) for index, name in raw_names.items()}
    else:
        raise ValueError("The 'names' field must be a list or mapping.")
    return dataset_root, config, names


def resolve_split_sources(dataset_root: Path, split_value: object) -> list[Path]:
    if split_value is None:
        return []
    values = split_value if isinstance(split_value, list) else [split_value]
    sources = []
    for value in values:
        source = Path(str(value))
        if not source.is_absolute():
            source = dataset_root / source
        sources.append(source.resolve())
    return sources


def iter_images(sources: Iterable[Path]) -> Iterable[Path]:
    seen: set[Path] = set()
    for source in sources:
        if source.is_dir():
            candidates = source.rglob("*")
        elif source.is_file() and source.suffix.lower() == ".txt":
            candidates = (
                Path(line.strip())
                for line in source.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        elif source.is_file():
            candidates = (source,)
        else:
            continue

        for candidate in candidates:
            if not candidate.is_absolute() and source.suffix.lower() == ".txt":
                candidate = (source.parent / candidate).resolve()
            if candidate.suffix.lower() in IMAGE_SUFFIXES and candidate not in seen:
                seen.add(candidate)
                yield candidate


def image_to_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    image_indices = [index for index, part in enumerate(parts) if part.lower() == "images"]
    if image_indices:
        parts[image_indices[-1]] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def parse_label_file(
    label_path: Path,
    image_width: int,
    image_height: int,
    imgsz: int,
    class_count: int,
) -> tuple[Counter[int], Counter[str], int, int]:
    classes: Counter[int] = Counter()
    resized_sizes: Counter[str] = Counter()
    malformed = 0
    instances = 0
    if not label_path.is_file():
        return classes, resized_sizes, malformed, instances

    resize_scale = imgsz / max(image_width, image_height)
    for line in label_path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        fields = stripped.split()
        if len(fields) != 5:
            malformed += 1
            continue
        try:
            class_value = float(fields[0])
            coordinates = [float(value) for value in fields[1:]]
        except ValueError:
            malformed += 1
            continue

        class_id = int(class_value)
        if class_value != class_id or class_id < 0 or class_id >= class_count:
            malformed += 1
            continue
        if any(value < 0.0 or value > 1.0 for value in coordinates) or coordinates[2] <= 0 or coordinates[3] <= 0:
            malformed += 1
            continue

        resized_width = coordinates[2] * image_width * resize_scale
        resized_height = coordinates[3] * image_height * resize_scale
        short_side = min(resized_width, resized_height)
        for threshold in SIZE_THRESHOLDS:
            if short_side < threshold:
                resized_sizes[f"short_side_lt_{threshold}px"] += 1
        resized_sizes["total"] += 1
        classes[class_id] += 1
        instances += 1
    return classes, resized_sizes, malformed, instances


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analyze_split(
    split: str,
    image_paths: list[Path],
    imgsz: int,
    names: dict[int, str],
    hashes: dict[str, list[tuple[str, str]]],
    check_duplicates: bool,
) -> tuple[SplitSummary, Counter[int], Counter[str]]:
    summary = SplitSummary(images=len(image_paths))
    class_counts: Counter[int] = Counter()
    resized_sizes: Counter[str] = Counter()

    for image_path in image_paths:
        if check_duplicates:
            try:
                hashes[sha256_file(image_path)].append((split, str(image_path)))
            except OSError:
                summary.unreadable_images += 1
                continue

        try:
            with Image.open(image_path) as image:
                image_width, image_height = image.size
        except (OSError, UnidentifiedImageError):
            summary.unreadable_images += 1
            continue

        label_path = image_to_label_path(image_path)
        if not label_path.is_file():
            summary.missing_labels += 1

        counts, size_counts, malformed, instances = parse_label_file(
            label_path,
            image_width,
            image_height,
            imgsz,
            len(names),
        )
        class_counts.update(counts)
        resized_sizes.update(size_counts)
        summary.malformed_labels += malformed
        summary.instances += instances
        if instances == 0:
            summary.background_images += 1

    return summary, class_counts, resized_sizes


def print_report(report: dict) -> None:
    print(f"Dataset root: {report['dataset_root']}")
    print(f"Analysis image size: {report['imgsz']}")
    print()
    for split, details in report["splits"].items():
        summary = details["summary"]
        print(f"[{split}] images={summary['images']} instances={summary['instances']} "
              f"background={summary['background_images']} missing_labels={summary['missing_labels']} "
              f"malformed={summary['malformed_labels']} unreadable={summary['unreadable_images']}")
        print(f"  classes: {details['class_counts']}")
        print(f"  resized boxes: {details['resized_box_counts']}")
    if report["cross_split_duplicates"]:
        print()
        print("Cross-split duplicate groups:")
        for duplicate_group in report["cross_split_duplicates"]:
            print(f"  {duplicate_group}")


def main() -> None:
    args = parse_args()
    data_path = args.data.resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset YAML does not exist: {data_path}")

    dataset_root, config, names = load_dataset_config(data_path)
    hashes: dict[str, list[tuple[str, str]]] = defaultdict(list)
    report: dict = {
        "dataset_yaml": str(data_path),
        "dataset_root": str(dataset_root),
        "imgsz": args.imgsz,
        "class_names": names,
        "splits": {},
        "cross_split_duplicates": [],
    }

    for split in ("train", "val", "test"):
        image_paths = sorted(iter_images(resolve_split_sources(dataset_root, config.get(split))))
        summary, class_counts, resized_sizes = analyze_split(
            split,
            image_paths,
            args.imgsz,
            names,
            hashes,
            args.check_duplicates,
        )
        report["splits"][split] = {
            "summary": asdict(summary),
            "class_counts": {names.get(class_id, str(class_id)): count for class_id, count in sorted(class_counts.items())},
            "resized_box_counts": dict(sorted(resized_sizes.items())),
        }

    for duplicate_paths in hashes.values():
        involved_splits = {split for split, _ in duplicate_paths}
        if len(involved_splits) > 1:
            report["cross_split_duplicates"].append(duplicate_paths)

    print_report(report)
    if args.output:
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Report written to: {output_path}")


if __name__ == "__main__":
    main()
