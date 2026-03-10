import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# Resolve paths from this file, not from current terminal location
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
YOLOV5_ROOT = ROOT / "yolov5"

if str(YOLOV5_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLOV5_ROOT))

from ultralytics.utils.plotting import Annotator, colors
from models.common import DetectMultiBackend
from utils.dataloaders import letterbox
from utils.general import Profile, check_img_size, non_max_suppression, scale_boxes
from utils.segment.general import process_mask, process_mask_native
from utils.torch_utils import select_device


def load_segmentation_model(weights, device="", imgsz=(640, 640), dnn=False, data=None, fp16=False):
    """
    Load YOLOv5 segmentation model once.

    Args:
        weights (str): Path to model weights
        device (str): Device string such as 'cpu', '0', 'cuda:0'
        imgsz (tuple | int): Inference image size
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

    pt = model.pt
    if isinstance(imgsz, int):
        warmup_shape = (1 if pt or model.triton else 1, 3, imgsz, imgsz)
    else:
        warmup_shape = (1 if pt or model.triton else 1, 3, imgsz[0], imgsz[1])

    model.warmup(imgsz=warmup_shape)

    raw_names = model.names
    names = raw_names if isinstance(raw_names, dict) else {i: n for i, n in enumerate(raw_names)}

    return {
        "model": model,
        "device": device,
        "imgsz": imgsz,
        "names": names,
    }


def segment_image(
    image,
    model,
    device,
    imgsz=(640, 640),
    show_masks=True,
    show_boxes=True,
    show_labels=True,
    show_confidence=True,
    conf_threshold=0.25,
    iou_threshold=0.45,
    max_det=1000,
    classes=None,
    agnostic_nms=False,
    augment=False,
    retina_masks=False,
    class_names=None,
):
    """
    Run YOLOv5 segmentation on a single image.

    Args:
        image (numpy.ndarray): Input BGR image
        model: Loaded YOLOv5 segmentation model
        device: Torch device
        imgsz (tuple | int): Inference image size
        show_masks (bool): Draw segmentation masks
        show_boxes (bool): Draw bounding boxes
        show_labels (bool): Draw class labels
        show_confidence (bool): Draw confidence with class label
        conf_threshold (float): Confidence threshold
        iou_threshold (float): IoU threshold for NMS
        max_det (int): Maximum detections
        classes (list | None): Optional class filter
        agnostic_nms (bool): Class agnostic NMS
        augment (bool): Augmented inference
        retina_masks (bool): Use native mask processing
        class_names (dict | None): Optional mapping {orig_name: display_name}

    Returns:
        dict: {
            'processed_image': numpy.ndarray,
            'original_image': numpy.ndarray,
            'detections': list,
            'detection_count': int,
            'class_counts': dict,
            'image_shape': tuple,
            'inference_time': dict
        }
    """
    if image is None:
        raise ValueError("Input image is None")
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a numpy array")
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be a 3-channel BGR image")

    stride, raw_names, pt = model.stride, model.names, model.pt
    names = raw_names if isinstance(raw_names, dict) else {i: n for i, n in enumerate(raw_names)}
    imgsz_checked = check_img_size(imgsz, s=stride)

    original_image = image.copy()
    im0 = image.copy()

    im = letterbox(image, imgsz_checked, stride=stride, auto=pt)[0]
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)

    dt = (Profile(device=device), Profile(device=device), Profile(device=device))

    detection_results = []
    class_counts = {}

    annotator = Annotator(
        im0,
        line_width=round(max(im0.shape[:2]) / 300),
        example=str(list(names.values())),
    )

    with dt[0]:
        im_t = torch.from_numpy(im).to(device)
        im_t = im_t.half() if model.fp16 else im_t.float()
        im_t /= 255.0
        if im_t.ndim == 3:
            im_t = im_t.unsqueeze(0)

    with dt[1]:
        pred, proto = model(im_t, augment=augment, visualize=False)[:2]

    with dt[2]:
        pred = non_max_suppression(
            pred,
            conf_threshold,
            iou_threshold,
            classes,
            agnostic_nms,
            max_det=max_det,
            nm=32
        )

    for i, det in enumerate(pred):
        if det is None or not det.numel():
            continue

        if retina_masks:
            det[:, :4] = scale_boxes(im_t.shape[2:], det[:, :4], im0.shape).round()
            masks = process_mask_native(proto[i], det[:, 6:], det[:, :4], im0.shape[:2])
        else:
            masks = process_mask(proto[i], det[:, 6:], det[:, :4], im_t.shape[2:], upsample=True)
            det[:, :4] = scale_boxes(im_t.shape[2:], det[:, :4], im0.shape).round()

        if show_masks and masks is not None and len(masks):
            mask_colors = [colors(int(cls_id), True) for cls_id in det[:, 5].tolist()]
            annotator.masks(
                masks,
                colors=mask_colors,
                im_gpu=(
                    torch.as_tensor(im0, dtype=torch.float16)
                    .to(device)
                    .permute(2, 0, 1)
                    .flip(0)
                    .contiguous() / 255.0
                ) if retina_masks else im_t[i],
            )

        for j, (*xyxy, conf, cls) in enumerate(det[:, :6]):
            cls = int(cls)
            orig_name = names.get(cls, str(cls))
            disp_name = class_names.get(orig_name, orig_name) if class_names else orig_name

            class_counts[disp_name] = class_counts.get(disp_name, 0) + 1

            mask_np = masks[j].cpu().numpy() if masks is not None else None

            bbox = [float(x) for x in xyxy]

            detection_results.append({
                "class": cls,
                "class_name": disp_name,
                "original_class_name": orig_name,
                "confidence": float(conf),
                "bbox": bbox,
                "bbox_normalized": [
                    bbox[0] / original_image.shape[1],
                    bbox[1] / original_image.shape[0],
                    bbox[2] / original_image.shape[1],
                    bbox[3] / original_image.shape[0],
                ],
                "mask": mask_np,
                "mask_id": j,
            })

            if show_boxes:
                if not show_labels:
                    label = None
                elif show_confidence:
                    label = f"{disp_name} {conf:.2f}"
                else:
                    label = disp_name

                annotator.box_label(xyxy, label, color=colors(cls, True))

    processed_image = annotator.result()

    return {
        "processed_image": processed_image,
        "original_image": original_image,
        "detections": detection_results,
        "detection_count": len(detection_results),
        "class_counts": class_counts,
        "image_shape": original_image.shape,
        "inference_time": {
            "preprocess": dt[0].dt,
            "inference": dt[1].dt,
            "postprocess": dt[2].dt,
        }
    }