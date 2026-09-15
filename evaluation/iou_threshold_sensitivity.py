"""Evaluate object-level metrics across a range of IoU thresholds."""
import argparse
import csv
from pathlib import Path
import geopandas as gpd
from object_level import evaluate_detection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", required=True)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    truth = gpd.read_file(args.truth)
    prediction = gpd.read_file(args.prediction)
    rows = [dict(iou_threshold=t, **evaluate_detection(truth, prediction, t)) for t in args.thresholds]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
