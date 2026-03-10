import numpy as np
from tracker.byte_tracker import BYTETracker


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


class BTArgs:
    def __init__(
        self,
        track_thresh=0.80,
        track_buffer=120,
        match_thresh=0.85,
        mot20=False,
    ):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.mot20 = mot20


def load_bytetracker(
    track_thresh=0.80,
    track_buffer=120,
    match_thresh=0.85,
    mot20=False,
    frame_rate=30,
):
    args = BTArgs(
        track_thresh=track_thresh,
        track_buffer=track_buffer,
        match_thresh=match_thresh,
        mot20=mot20,
    )
    return BYTETracker(args, frame_rate=frame_rate)


def update_bytetracker(frame, detections, tracker, allowed_classes=None, class_names=None):
    """
    Same calling style as DeepSORT and SORT helpers.

    Args
        frame
            current frame, used for frame size
        detections
            output from detect_objects()
        tracker
            BYTETracker instance
        allowed_classes
            optional class filter
        class_names
            optional fallback mapping if class_name is absent

    Returns
        list of tracked objects
    """
    frame_h, frame_w = frame.shape[:2]

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
        score = det.get("conf")
        if score is None:
            score = det.get("confidence")
        if score is None:
            score = det.get("score")
        if score is None:
            score = 1.0

        det_boxes.append([float(x1), float(y1), float(x2), float(y2)])
        det_scores.append(float(score))
        det_classes.append(cls_name)

    if det_boxes:
        dets = np.concatenate(
            [
                np.asarray(det_boxes, dtype=np.float32),
                np.asarray(det_scores, dtype=np.float32)[:, None],
            ],
            axis=1
        )
    else:
        dets = np.zeros((0, 5), dtype=np.float32)

    online_targets = tracker.update(
        dets,
        (frame_h, frame_w),
        (frame_h, frame_w)
    )

    tracked_objects = []

    for t in online_targets:
        tlwh = t.tlwh
        track_id = int(t.track_id)

        x1 = float(tlwh[0])
        y1 = float(tlwh[1])
        x2 = float(tlwh[0] + tlwh[2])
        y2 = float(tlwh[1] + tlwh[3])
        track_box = [x1, y1, x2, y2]

        if not det_boxes:
            continue

        ious = [iou(track_box, db) for db in det_boxes]
        best = int(np.argmax(ious))

        if ious[best] <= 0.1:
            continue

        cls_name = det_classes[best]
        confidence = float(det_scores[best])

        tracked_objects.append({
            "track_id": track_id,
            "class_name": cls_name,
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "confidence": confidence,
        })

    return tracked_objects