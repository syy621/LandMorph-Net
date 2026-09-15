"""Compute relaxed object-level Pobj, Robj, and Fobj."""
import argparse
import geopandas as gpd


def compute_iou(geom1, geom2):
    if geom1.is_empty or geom2.is_empty:
        return 0.0
    union = geom1.union(geom2).area
    return geom1.intersection(geom2).area / union if union else 0.0


def evaluate_detection(truth_gdf, pred_gdf, iou_thresh=0.2):
    if truth_gdf.crs != pred_gdf.crs:
        pred_gdf = pred_gdf.to_crs(truth_gdf.crs)
    truth = list(truth_gdf.geometry)
    pred = list(pred_gdf.geometry)
    pred_hits = [any(compute_iou(p, t) >= iou_thresh for t in truth) for p in pred]
    truth_hits = [any(compute_iou(p, t) >= iou_thresh for p in pred) for t in truth]
    tp_pred, tp_truth = sum(pred_hits), sum(truth_hits)
    fp, fn = len(pred) - tp_pred, len(truth) - tp_truth
    precision = tp_pred / len(pred) if pred else 0.0
    recall = tp_truth / len(truth) if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return dict(total_truth=len(truth), total_pred=len(pred), tp_pred=tp_pred, fp=fp,
                tp_truth=tp_truth, fn=fn, Pobj=precision, Robj=recall, Fobj=f1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", required=True)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.2)
    args = parser.parse_args()
    result = evaluate_detection(gpd.read_file(args.truth), gpd.read_file(args.prediction), args.iou_threshold)
    for key, value in result.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")


if __name__ == "__main__":
    main()
