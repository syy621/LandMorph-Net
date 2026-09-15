"""Save Ultralytics intermediate response maps for a local image."""
import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", default="outputs/response")
    args = parser.parse_args()
    model = YOLO(args.model)
    model.predict(args.source, visualize=True, project=args.output, name="features", exist_ok=True)


if __name__ == "__main__":
    main()
