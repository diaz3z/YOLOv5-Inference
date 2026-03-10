# YOLO Inference Toolkit (PyTorch + ONNX, Detection/Segmentation/Classification/Tracking)

This repository is a local inference toolkit built around an embedded `yolov5/` checkout.  
It provides:

- Object detection (`detection/`)
- Instance segmentation (`segmentation/`)
- Image classification (`classification/`)
- ONNX inference variants (`ONNX/`)
- Real-time tracking (ByteTrack, DeepSORT, SORT) for detections
- Webcam streaming with Flask (`camera_flask.py`)

Pretrained weights are already placed under `models/`.

## Repository Layout

```text
.
|-- camera.py
|-- camera_flask.py
|-- templates/camera.html
|-- detection/
|   |-- detection_function.py
|   |-- detection_imshow.py
|   |-- ByteTracker.py
|   |-- DeepSort.py
|   |-- Sort_tracker.py
|   `-- tracker/              # local ByteTrack implementation
|-- segmentation/
|   |-- segmentation_function.py
|   |-- segmentation_imshow.py
|   `-- seg-vid-v6.py         # legacy script (see Known Issues)
|-- classification/
|   |-- classification_function.py
|   `-- classification_imshow.py
|-- ONNX/
|   |-- detection_onnx_function.py
|   |-- segmentation_onnx_function.py
|   |-- classification_onnx_function.py
|   |-- imshow-det.py
|   |-- imshow-seg.py
|   `-- imshow-cls.py
|-- models/
|   |-- detection/yolov5s.pt
|   |-- detection/yolov5s.onnx
|   |-- classification/yolov5n-cls.pt
|   |-- classification/yolov5n-cls.onnx
|   |-- segmentation/yolov5s-seg.pt
|   `-- segmentation/yolov5s-seg.onnx
`-- yolov5/                   # bundled YOLOv5 source
```

## Environment Setup

1. Create and activate a virtual environment.
2. Install YOLOv5 dependencies:

```powershell
pip install -r yolov5/requirements.txt
```

3. Install project-specific extras:

```powershell
pip install flask deep-sort-realtime lap
```

4. Install ONNX Runtime (choose one):

```powershell
# CPU
pip install onnxruntime

# OR GPU
pip install onnxruntime-gpu
```

Notes:
- `deep-sort-realtime` is only required for DeepSORT tracking.
- `lap` is required by the local ByteTrack matching logic.
- GPU inference uses `device="cuda:0"` in sample scripts. Change to `device="cpu"` if needed.
- ONNX scripts require `onnxruntime` (or `onnxruntime-gpu` for CUDA provider).

## How to Run

All commands below assume your current directory is the repository root.

### 1) Flask webcam + detection stream

```powershell
python camera_flask.py
```

Open: `http://127.0.0.1:5000/`

What it does:
- Captures webcam frames (`camera.py`)
- Runs detection via `detection/detection_function.py`
- Streams MJPEG response at `/video_feed`

### 2) Detection + tracking preview (OpenCV windows)

```powershell
cd detection
python detection_imshow.py
```

Default tracker in script: ByteTrack.  
You can switch to DeepSORT or SORT by changing the commented lines in `detection_imshow.py`.

### 3) Segmentation preview (OpenCV window)

```powershell
cd segmentation
python segmentation_imshow.py
```

### 4) Classification image test

```powershell
cd classification
python classification_imshow.py
```

This script reads `../img/road4.jpg`, writes `../results/road3_result.jpg`, and prints top-k predictions.

### 5) ONNX detection / segmentation / classification

```powershell
cd ONNX
python imshow-det.py
python imshow-seg.py
python imshow-cls.py
```

Notes:
- `imshow-det.py` uses `../models/detection/yolov5s.onnx`
- `imshow-seg.py` uses `../models/segmentation/yolov5s-seg.onnx`
- `imshow-cls.py` uses `../models/classification/yolov5n-cls.onnx`
- All ONNX demos default to `device="cuda:0"` in script; switch to `cpu` when needed.

## Core Function APIs

### Detection

File: `detection/detection_function.py`

- `load_detection_model(...)` returns model bundle:
  - `model`, `device`, `imgsz`, `names`
- `detect_objects(...)` returns:
  - `detections`: list of dicts with `class`, `class_name`, `confidence`, `bbox`, normalized bbox
  - `annotated_image`
  - `original_image`

### Segmentation

File: `segmentation/segmentation_function.py`

- `load_segmentation_model(...)` returns model bundle:
  - `model`, `device`, `imgsz`, `names`
- `segment_image(...)` returns:
  - `processed_image`
  - `detections` including `bbox`, class info, confidence, and mask array
  - `detection_count`, `class_counts`, `image_shape`, `inference_time`

### Classification

File: `classification/classification_function.py`

- `load_classification_model(...)` returns model bundle:
  - `model`, `device`, `imgsz`, `names`
- `classify_image(...)` returns:
  - `predictions` (top-k class probabilities)
  - `top1`
  - `annotated_image`
  - `original_image`

### ONNX APIs

Files under `ONNX/`:

- `detection_onnx_function.py`
  - `load_detection_onnx_model(...)`
  - `detect_objects_onnx(...)`
- `segmentation_onnx_function.py`
  - `load_segmentation_onnx_model(...)`
  - `segment_image_onnx(...)`
- `classification_onnx_function.py`
  - `load_classification_onnx_model(...)`
  - `classify_image_onnx(...)`

Return structures are aligned with the corresponding PyTorch APIs (detections, annotated/processed image, predictions, class counts).

## Tracking Modules

Under `detection/`:

- `ByteTracker.py`: wrapper around local `detection/tracker/byte_tracker.py`
- `DeepSort.py`: wrapper around `deep_sort_realtime`
- `Sort_tracker.py`: lightweight local SORT implementation

Each wrapper accepts detections from `detect_objects(...)` and returns tracked objects with:
- `track_id`
- `class_name`
- `bbox`
- `confidence`

## Known Issues

1. `segmentation/seg-vid-v6.py` imports `yolo_segmentation` from `segmentation_function`, but current `segmentation_function.py` exposes `segment_image` and does not define `yolo_segmentation`.
2. Example scripts inside `detection/`, `segmentation/`, `classification/`, and `ONNX/` use relative imports/paths that are easiest to run from their own folder (`cd <module_dir>` first).
3. Several scripts default to `device="cuda:0"`. Change to CPU if CUDA is unavailable.

## Outputs

- Image output example: `results/road3_result.jpg`
- Live outputs: OpenCV windows or Flask stream depending on script

## Suggested Next Improvements

- Add a root-level `requirements.txt` that includes both YOLOv5 and project extras.
- Convert example scripts to package-safe imports so they can run directly from root without `cd`.
- Standardize weight/config paths via a single config file or CLI arguments.
