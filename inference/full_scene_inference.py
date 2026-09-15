"""Full-scene tiled instance segmentation and Shapefile post-processing."""
import argparse
import os
from pathlib import Path
import cv2
import geopandas as gpd
import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from shapely.geometry import Polygon
from torchvision.ops import nms
from ultralytics import YOLO


def sliding_window(image_h, image_w, window_size, overlap):
    stride = max(1, int(window_size * (1 - overlap)))
    for y in range(0, image_h, stride):
        for x in range(0, image_w, stride):
            yield max(0, min(x, image_w - window_size)), max(0, min(y, image_h - window_size)), window_size, window_size


def preprocess_patch(img_data):
    img = np.transpose(img_data, (1, 2, 0))
    img = np.nan_to_num(img, nan=0)
    if img.dtype != np.uint8:
        img = np.clip(img / 256, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def predict_image(model, src, windows, band_indices, conf, imgsz, device):
    predictions = []
    for x, y, win_w, win_h in windows:
        image = preprocess_patch(src.read(band_indices, window=Window(x, y, win_w, win_h)))
        result = model.predict(image, conf=conf, imgsz=imgsz, verbose=False, device=device)[0]
        if result.boxes is None or len(result.boxes) == 0 or result.masks is None:
            continue
        for index, box in enumerate(result.boxes.data.cpu().numpy()):
            x1, y1, x2, y2, confidence, _ = box
            mask = result.masks.xy[index] if index < len(result.masks.xy) else None
            if mask is not None and len(mask) >= 3:
                mask = mask + np.array([x, y])
            predictions.append({"box": [x1 + x, y1 + y, x2 + x, y2 + y, confidence], "mask": mask})
    if not predictions:
        return []
    boxes = torch.tensor([p["box"][:4] for p in predictions])
    scores = torch.tensor([p["box"][4] for p in predictions])
    keep = nms(boxes, scores, 0.35).numpy()
    return [predictions[i] for i in keep]


def export_predictions(predictions, transform, crs, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    boxes, masks = [], []
    for item in predictions:
        x1, y1, x2, y2, confidence = item["box"]
        gx1, gy1 = transform * (x1, y1)
        gx2, gy2 = transform * (x2, y2)
        boxes.append((Polygon([(gx1, gy1), (gx2, gy1), (gx2, gy2), (gx1, gy2)]), confidence))
        mask = item["mask"]
        if mask is not None and len(mask) >= 3:
            xs, ys = transform * (mask[:, 0], mask[:, 1])
            poly = Polygon(np.column_stack((xs, ys)))
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_valid and not poly.is_empty:
                masks.append((poly, confidence))
    gpd.GeoDataFrame({"confidence": [c for _, c in boxes]}, geometry=[g for g, _ in boxes], crs=crs).to_file(output_dir / "predicted_bboxes.shp")
    if masks:
        gpd.GeoDataFrame({"confidence": [c for _, c in masks]}, geometry=[g for g, _ in masks], crs=crs).to_file(output_dir / "predicted_masks.shp")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--window-size", type=int, default=1280)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    model = YOLO(args.model)
    with rasterio.open(args.source) as src:
        windows = sliding_window(src.height, src.width, args.window_size, args.overlap)
        predictions = predict_image(model, src, windows, [1, 2, 3], args.conf, args.window_size, args.device)
        export_predictions(predictions, src.transform, src.crs, Path(args.output))


if __name__ == "__main__":
    main()
