import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import pathlib
import sys

# Add YOLOv5 root to path - ADJUST THIS PATH TO YOUR YOLOV5 DIRECTORY
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]                  # project/
YOLOV5_ROOT = ROOT / "yolov5"           # project/yolov5

if str(YOLOV5_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLOV5_ROOT))
 

from utils.torch_utils import select_device
from utils.dataloaders import letterbox
from models.common import DetectMultiBackend

# Fix for path issues (if necessary)
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


def load_classification_model(weights, device="", imgsz=224, dnn=False, data=None, fp16=False):
    """
    Load YOLOv5 classification model once.

    Args:
        weights (str): Path to model weights
        device (str): Device string such as 'cpu', '0', 'cuda:0'
        imgsz (int): Inference image size
        dnn (bool): Use OpenCV DNN for ONNX inference
        data: Optional dataset config
        fp16 (bool): Use half precision if supported

    Returns:
        dict: {
            'model': model,
            'device': device,
            'imgsz': imgsz,
            'names': names
        }
    """
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=fp16)

    # warmup once
    model.warmup(imgsz=(1, 3, imgsz, imgsz))

    names = model.names
    return {
        "model": model,
        "device": device,
        "imgsz": imgsz,
        "names": names,
    }


def classify_image(
    image,
    model,
    device,
    imgsz=224,
    topk=1,
    draw_label=True,
    class_names=None,
    text_color=(0, 255, 0),
    text_scale=1.0,
    text_thickness=2,
):
    """
    Perform YOLOv5 classification on a single image.

    Args:
        image (numpy.ndarray): Input image in BGR format
        model: Loaded YOLOv5 model instance
        device: Torch device
        imgsz (int): Inference image size
        topk (int): Number of top predictions to return
        draw_label (bool): Draw top1 label on image
        class_names (dict): Optional mapping {original_name: display_name}
        text_color (tuple): BGR text color
        text_scale (float): Font scale
        text_thickness (int): Font thickness

    Returns:
        dict: {
            'predictions': [...],
            'top1': {...},
            'annotated_image': numpy.ndarray,
            'original_image': numpy.ndarray
        }
    """
    original_image = image.copy()
    annotated_image = image.copy()

    # Preprocess
    img = letterbox(image, new_shape=imgsz, auto=False)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if model.fp16 else img.float()
    img /= 255.0
    if img.ndim == 3:
        img = img.unsqueeze(0)

    # Inference
    with torch.no_grad():
        pred = model(img)

    # Some backends may return tuple or list
    if isinstance(pred, (list, tuple)):
        pred = pred[0]

    if pred.ndim == 1:
        pred = pred.unsqueeze(0)

    probs = F.softmax(pred, dim=1)
    top_probs, top_idxs = torch.topk(probs, k=min(topk, probs.shape[1]), dim=1)

    predictions = []
    names = model.names

    for prob, idx in zip(top_probs[0], top_idxs[0]):
        cls_id = int(idx.item())
        conf = float(prob.item())
        orig_name = names[cls_id]
        disp_name = class_names.get(orig_name, orig_name) if class_names else orig_name

        predictions.append({
            "class": cls_id,
            "class_name": disp_name,
            "original_class_name": orig_name,
            "confidence": conf,
        })

    top1 = predictions[0] if predictions else None

    if draw_label and top1 is not None:
        label = f"{top1['class_name']} {top1['confidence']:.2f}"
        cv2.putText(
            annotated_image,
            label,
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            text_color,
            text_thickness,
            cv2.LINE_AA
        )

    return {
        "predictions": predictions,
        "top1": top1,
        "annotated_image": annotated_image,
        "original_image": original_image,
    }