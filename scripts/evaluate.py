from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a YOLO12 checkpoint on a fixed split.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--project", default="runs/battery-eval")
    parser.add_argument("--name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model.resolve()
    data_path = args.data.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint does not exist: {model_path}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset YAML does not exist: {data_path}")

    model = YOLO(str(model_path))
    run_name = args.name or f"{model_path.stem}_{args.split}_{args.imgsz}"
    metrics = model.val(
        data=str(data_path),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        iou=args.iou,
        max_det=args.max_det,
        project=args.project,
        name=run_name,
        plots=True,
    )

    serializable_metrics = {
        key: float(value) if hasattr(value, "item") else value
        for key, value in metrics.results_dict.items()
    }
    print(json.dumps(serializable_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
