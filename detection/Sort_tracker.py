import numpy as np


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


class SortTrack:
    __slots__ = ("track_id", "bbox", "hits", "age", "time_since_update")

    def __init__(self, track_id, bbox_xyxy):
        self.track_id = track_id
        self.bbox = np.array(bbox_xyxy, dtype=np.float32)
        self.hits = 1
        self.age = 1
        self.time_since_update = 0


class SORT:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.iou_threshold = float(iou_threshold)
        self._next_id = 1
        self.tracks = []

    def update(self, dets_xyxy_score):
        for trk in self.tracks:
            trk.age += 1
            trk.time_since_update += 1

        dets = dets_xyxy_score if dets_xyxy_score is not None else np.zeros((0, 5), dtype=np.float32)
        if dets.shape[0] == 0:
            self._prune()
            return self._export_tracks()

        det_boxes = dets[:, :4].astype(np.float32)

        if len(self.tracks) > 0:
            iou_mat = np.zeros((len(self.tracks), det_boxes.shape[0]), dtype=np.float32)
            for ti, trk in enumerate(self.tracks):
                for di, db in enumerate(det_boxes):
                    iou_mat[ti, di] = iou(trk.bbox, db)
        else:
            iou_mat = np.zeros((0, det_boxes.shape[0]), dtype=np.float32)

        matched_det = set()
        matched_trk = set()

        while iou_mat.size:
            ti, di = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
            best = iou_mat[ti, di]

            if best < self.iou_threshold:
                break

            if ti in matched_trk or di in matched_det:
                iou_mat[ti, di] = -1.0
                continue

            matched_trk.add(ti)
            matched_det.add(di)

            trk = self.tracks[ti]
            trk.bbox = det_boxes[di]
            trk.hits += 1
            trk.time_since_update = 0

            iou_mat[ti, :] = -1.0
            iou_mat[:, di] = -1.0

        for di in range(det_boxes.shape[0]):
            if di not in matched_det:
                self.tracks.append(SortTrack(self._next_id, det_boxes[di]))
                self._next_id += 1

        self._prune()
        return self._export_tracks()

    def _prune(self):
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

    def _export_tracks(self):
        exported = []
        for t in self.tracks:
            if t.hits >= self.min_hits or t.time_since_update == 0:
                x1, y1, x2, y2 = t.bbox.tolist()
                tlwh = np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)

                class Obj:
                    pass

                o = Obj()
                o.track_id = t.track_id
                o.tlwh = tlwh
                exported.append(o)

        return exported


def load_sort_tracker(max_age=60, min_hits=3, iou_threshold=0.3):
    return SORT(
        max_age=max_age,
        min_hits=min_hits,
        iou_threshold=iou_threshold
    )


def update_sort_tracker(frame, detections, tracker, allowed_classes=None, class_names=None):
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
        score = det.get("conf") or det.get("confidence") or det.get("score") or 1.0

        det_boxes.append([float(x1), float(y1), float(x2), float(y2)])
        det_scores.append(float(score))
        det_classes.append(cls_name)

    if det_boxes:
        dets = np.concatenate(
            [
                np.asarray(det_boxes, dtype=np.float32),
                np.asarray(det_scores, dtype=np.float32)[:, None]
            ],
            axis=1
        )
    else:
        dets = np.zeros((0, 5), dtype=np.float32)

    online_targets = tracker.update(dets)

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