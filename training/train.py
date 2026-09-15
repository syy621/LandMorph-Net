"""Train and validate LandMorph-Net on cropped image patches."""
import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="configs/landmorph.yaml")
    parser.add_argument("--data", default="configs/landslide-seg.yaml")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    model = YOLO(args.model)
    if args.weights:
        model.load(args.weights)
    train_args = dict(data=args.data, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
                      patience=32, lr0=0.001, cos_lr=True, flipud=0.5, box=7.5, cls=0.5,
                      mosaic=0.5, mixup=0.2, copy_paste=0.2, hsv_h=0.05, hsv_s=0.4,
                      hsv_v=0.4, scale=0.8)
    if args.device:
        train_args["device"] = args.device
    model.train(**train_args)
    model.val(data=args.data, imgsz=args.imgsz, device=args.device)


if __name__ == "__main__":
    main()
