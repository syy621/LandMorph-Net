# LandMorph-Net

LandMorph-Net is a frequency-spatial collaborative instance-segmentation framework for regional landslide mapping from high-resolution optical imagery. The implementation extends the local Ultralytics runtime with three paper-specific modules:

- **FDAF**: Frequency-Domain Adaptive Filtering for adaptive frequency-component selection.
- **CAA**: Coordinate Attention Alignment, implemented as `CoordAttAlign`, for directional spatial-dependency preservation.
- **SubPixelUpsample**: learnable PixelShuffle-based feature reconstruction for boundary-detail recovery.

The repository contains code and model configurations only. Google Earth imagery, training labels, private datasets, prediction outputs, and trained weights are excluded because of licensing, privacy, and file-size restrictions.

## Project layout

```text
configs/       Dataset and LandMorph-Net model YAML files
models/        Public entry points for the paper-specific modules
training/      Patch-based training and validation entry point
inference/     Full-scene tiled instance-segmentation inference
evaluation/    Object-level metrics and IoU-threshold sensitivity analysis
visualization/ Feature-response visualization entry point
ultralytics/   Local Ultralytics runtime with the LandMorph-Net modules
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

## Data and weights

Place licensed data outside the repository, for example under a local `data/` directory, and keep trained checkpoints under a local `weights/` directory. Both locations are ignored by Git. Update the dataset YAML or pass paths through the command-line options. No Google Earth imagery or trained weights are distributed here.

## Training and patch validation

```bash
python training/train.py --model configs/landmorph.yaml --data configs/landslide-seg.yaml --epochs 100 --batch 6 --imgsz 1280
```

The script retains the paper training settings and runs validation after training. Use `--weights` only when fine-tuning from a local checkpoint.

## Full-scene inference and post-processing

```bash
python inference/full_scene_inference.py --model weights/landmorph.pt --source data/test_scene.tif --output outputs/test_scene
```

The inference script performs overlapping-window prediction, confidence filtering, box NMS, mask reconstruction, and Shapefile export. It requires `rasterio`, `geopandas`, and `shapely`.

## Object-level evaluation

The relaxed matching protocol used in the paper reports object-level **Pobj**, **Robj**, and **Fobj**. It permits one-to-many and many-to-one spatial correspondences, which is useful for regional screening but does not remove the need to inspect instance merging and splitting.

```bash
python evaluation/object_level.py --truth data/reference.shp --prediction outputs/test_scene/predicted_masks.shp --iou-threshold 0.2
```

To evaluate sensitivity to the IoU matching threshold:

```bash
python evaluation/iou_threshold_sensitivity.py --truth data/reference.shp --prediction outputs/test_scene/predicted_masks.shp --thresholds 0.1 0.2 0.3 0.4 0.5 --output outputs/iou_sensitivity.csv
```

## Response visualization

```bash
python visualization/response_visualization.py --model weights/landmorph.pt --source data/example_patch.tif --output outputs/response
```

This entry point uses the local Ultralytics feature-visualization pathway (`visualize=True`) to save intermediate response maps. A separate Grad-CAM implementation was not present in the supplied project files and is therefore not claimed as part of this release.

## Reproducibility notes

The repository preserves the paper-specific module implementations and model graph. Exact numerical reproduction additionally requires the original licensed imagery, annotations, preprocessing choices, local checkpoint, and matching/post-processing settings.

## License and citation

The local Ultralytics runtime is distributed under the AGPL-3.0 license. See `CITATION.cff` for the paper/software citation metadata.
