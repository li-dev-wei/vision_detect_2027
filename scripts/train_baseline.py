from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a reproducible YOLO26n new-energy defect experiment.")
    parser.add_argument("--data", type=Path, required=True, help="Path to the dataset YAML file.")
    parser.add_argument("--model", default="yolo26n.pt", help="Checkpoint or model YAML path.")
    parser.add_argument(
        "--pretrained",
        default=None,
        help="Optional checkpoint whose matching weights are loaded into a YAML-defined model.",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", default="runs/battery")
    parser.add_argument("--name", default=None)
    parser.add_argument("--cache", choices=("false", "disk", "ram"), default="false")
    parser.add_argument("--mosaic", type=float, default=0.5)
    parser.add_argument("--close-mosaic", type=int, default=20)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--translate", type=float, default=0.05)
    parser.add_argument("--scale", type=float, default=0.25)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--flipud", type=float, default=0.0)
    parser.add_argument("--hsv-h", type=float, default=0.01)
    parser.add_argument("--hsv-s", type=float, default=0.2)
    parser.add_argument("--hsv-v", type=float, default=0.2)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from models.ultralytics_plugins import register_dmare

    register_dmare()
    from ultralytics import YOLO

    data_path = args.data.resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset YAML does not exist: {data_path}")

    model = YOLO(args.model)
    if args.pretrained:
        model.load(args.pretrained)

    run_name = args.name or f"{Path(args.model).stem}_{args.imgsz}_seed{args.seed}"
    cache: bool | str = False if args.cache == "false" else args.cache

    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        deterministic=True,
        project=args.project,
        name=run_name,
        cache=cache,
        mosaic=args.mosaic,
        close_mosaic=args.close_mosaic,
        mixup=args.mixup,
        degrees=0.0,
        translate=args.translate,
        scale=args.scale,
        fliplr=args.fliplr,
        flipud=args.flipud,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        resume=args.resume,
        plots=True,
        val=True,
    )


if __name__ == "__main__":
    main()
