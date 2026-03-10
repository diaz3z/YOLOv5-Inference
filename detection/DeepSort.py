import numpy as np
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    return inter / (area_a + area_b - inter + 1e-6)


def load_deepsort_tracker(
    max_age=60,
    n_init=3,
    max_iou_distance=0.7,
    max_cosine_distance=0.2,
    embedder="mobilenet",
    nn_budget=None,
    bgr=True,
    nms_max_overlap=1.0,
):
    return DeepSort(
        max_age=max_age,
        n_init=n_init,
        max_iou_distance=max_iou_distance,
        max_cosine_distance=max_cosine_distance,
        nn_budget=nn_budget,
        embedder=embedder,
        half=torch.cuda.is_available(),
        bgr=bgr,
        nms_max_overlap=nms_max_overlap,
    )


def update_deepsort_tracker(frame, detections, tracker, allowed_classes=None, class_names=None):
    """
    Same calling style as SORT version.
    class_names is accepted for API consistency.
    """
    ds_dets = []
    det_boxes = []
    det_scores = []
    det_classes = []

    for det in detections:
        if not isinstance(det, dict):
            continue

        bbox = det.get("bbox")
        if bbox is None:
            continue

        cls_name = det.get("class_name")

        if cls_name is None:
            cls_id = det.get("cls")
            if cls_id is None:
                cls_id = det.get("class")

            if cls_id is None or class_names is None:
                continue

            try:
                cls_id = int(cls_id)
                if isinstance(class_names, dict):
                    cls_name = str(class_names[cls_id])
                else:
                    if cls_id < 0 or cls_id >= len(class_names):
                        continue
                    cls_name = str(class_names[cls_id])
            except Exception:
                continue

        if allowed_classes is not None and cls_name not in allowed_classes:
            continue

        x1, y1, x2, y2 = bbox[:4]
        w = float(x2 - x1)
        h = float(y2 - y1)
        if w <= 0 or h <= 0:
            continue

        conf = det.get("confidence", det.get("conf", det.get("score", 1.0)))
        conf = float(conf)

        ds_dets.append(([float(x1), float(y1), w, h], conf, str(cls_name)))
        det_boxes.append([float(x1), float(y1), float(x2), float(y2)])
        det_scores.append(conf)
        det_classes.append(str(cls_name))

    tracks = tracker.update_tracks(ds_dets, frame=frame)

    tracked_objects = []

    for tr in tracks:
        if not tr.is_confirmed() or tr.time_since_update > 0:
            continue

        x1, y1, x2, y2 = tr.to_ltrb()
        track_box = [float(x1), float(y1), float(x2), float(y2)]

        det_cls = None
        if hasattr(tr, "det_class"):
            det_cls = tr.det_class
        elif hasattr(tr, "get_det_class"):
            det_cls = tr.get_det_class()

        det_conf = getattr(tr, "det_conf", None)

        if det_cls is None or det_conf is None:
            if not det_boxes:
                continue

            ious = [iou(track_box, db) for db in det_boxes]
            best = int(np.argmax(ious))

            if ious[best] <= 0.1:
                continue

            det_cls = det_classes[best]
            det_conf = det_scores[best]

        tracked_objects.append({
            "track_id": int(tr.track_id),
            "class_name": str(det_cls),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "confidence": float(det_conf),
        })

    return tracked_objects