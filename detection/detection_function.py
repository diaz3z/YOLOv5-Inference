import cv2
import numpy as np
import torch
from pathlib import Path
import sys
import pathlib

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]                  # project/
YOLOV5_ROOT = ROOT / "yolov5"           # project/yolov5

if str(YOLOV5_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLOV5_ROOT))

from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from utils.dataloaders import letterbox
from models.common import DetectMultiBackend
from ultralytics.utils.plotting import Annotator, colors


# Fix for path issues (if necessary)
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


def load_detection_model(weights, device="", imgsz=640, dnn=False, data=None, fp16=False):
    """
    Load YOLOv5 detection model once.

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

    stride = model.stride
    imgsz = check_img_size(imgsz, s=stride)

    # warmup once
    pt = model.pt
    model.warmup(imgsz=(1 if pt or model.triton else 1, 3, imgsz, imgsz))

    return {
        "model": model,
        "device": device,
        "imgsz": imgsz,
        "names": model.names,
    }


def detect_objects(
    image,
    model,
    device,
    imgsz=640,
    conf_thres=0.25,
    iou_thres=0.45,
    max_det=1000,
    classes=None,
    agnostic_nms=False,
    augment=False,
    draw_boxes=True,
    show_conf=False,
    class_names=None,
    box_color=None
):
    """
    Perform YOLOv5 object detection on a single image.

    Args:
        image (numpy.ndarray): Input image as numpy array in BGR format
        model: Loaded YOLOv5 model instance
        device: Torch device
        imgsz (int): Inference image size
        conf_thres (float): Confidence threshold
        iou_thres (float): IoU threshold for NMS
        max_det (int): Maximum detections per image
        classes (list): Filter by class indices
        agnostic_nms (bool): Class agnostic NMS
        augment (bool): Augmented inference
        draw_boxes (bool): Draw bounding boxes on image
        show_conf (bool): Show confidence scores on bounding boxes
        class_names (dict): Mapping {orig_name: new_name}
        box_color (None | tuple | callable): If None uses ultralytics colors

    Returns:
        dict: {
            'detections': [...],
            'annotated_image': numpy.ndarray,
            'original_image': numpy.ndarray
        }
    """
    original_image = image.copy()
    annotated_image = image.copy()

    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # Preprocess
    img = letterbox(image, imgsz, stride=stride, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW and BGR to RGB
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if model.fp16 else img.float()
    img /= 255.0
    if img.ndim == 3:
        img = img[None]

    # Inference
    with torch.no_grad():
        pred = model(img, augment=augment, visualize=False)

    pred = non_max_suppression(
        pred,
        conf_thres,
        iou_thres,
        classes,
        agnostic_nms,
        max_det=max_det
    )

    detections = []

    for det in pred:
        if not len(det):
            continue

        det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], original_image.shape).round()

        if draw_boxes:
            annotator = Annotator(annotated_image, line_width=3, example=str(names))

        for *xyxy, conf, cls in reversed(det):
            cls = int(cls)
            conf = float(conf)
            xyxy = [int(x) for x in xyxy]

            h, w = original_image.shape[:2]
            xyxy[0] = max(0, min(xyxy[0], w - 1))
            xyxy[1] = max(0, min(xyxy[1], h - 1))
            xyxy[2] = max(0, min(xyxy[2], w - 1))
            xyxy[3] = max(0, min(xyxy[3], h - 1))

            if xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
                continue

            orig_name = names[cls]
            disp_name = class_names.get(orig_name, orig_name) if class_names else orig_name

            if box_color is None:
                col = colors(cls, True)
            elif callable(box_color):
                col = box_color(cls, True)
            else:
                col = box_color

            if draw_boxes:
                label = f"{disp_name} {conf:.2f}" if show_conf else disp_name
                annotator.box_label(xyxy, label, color=col)

            detections.append({
                "class": cls,
                "class_name": disp_name,
                "original_class_name": orig_name,
                "confidence": conf,
                "bbox": xyxy,
                "bbox_normalized": [
                    xyxy[0] / original_image.shape[1],
                    xyxy[1] / original_image.shape[0],
                    xyxy[2] / original_image.shape[1],
                    xyxy[3] / original_image.shape[0],
                ]
            })

        if draw_boxes:
            annotated_image = annotator.result()

    return {
        "detections": detections,
        "annotated_image": annotated_image,
        "original_image": original_image,
    }