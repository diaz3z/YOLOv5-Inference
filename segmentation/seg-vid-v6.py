import cv2
import numpy as np
from collections import defaultdict

from segmentation.segmentation_function import yolo_segmentation
from tracker.byte_tracker import BYTETracker


# ─── Ground Truth for Verification ───────────────────────────────────────────

GROUND_TRUTH = {
    "card-doc":  5,
    "picture":   1,
    "paper-doc": 5,
    "item":      1,
}


# ─── Helpers ────────────────────────────────────────────────────────────────

def clamp_bbox(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return x1, y1, x2, y2


def blur_bbox(image, bbox, ksize=31, sigma=0):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = clamp_bbox(*bbox, w, h)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return image
    if ksize % 2 == 0:
        ksize += 1
    roi = image[y1:y2, x1:x2]
    image[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (ksize, ksize), sigma)
    return image


def get_class_name(det):
    if not isinstance(det, dict):
        return "unknown"
    for key in ("label", "class", "name"):
        if key in det and det[key] is not None:
            return str(det[key])
    if "cls" in det and det["cls"] is not None:
        return str(det["cls"])
    return "unknown"


def get_conf(det):
    if not isinstance(det, dict):
        return 1.0
    for key in ("conf", "confidence", "score"):
        if key in det and det[key] is not None:
            return float(det[key])
    return 1.0


def point_in_rect(pt, rect):
    x, y = pt
    rx1, ry1, rx2, ry2 = rect
    return (rx1 <= x <= rx2) and (ry1 <= y <= ry2)


def bbox_center(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    return inter / (area_a + area_b - inter + 1e-6)


# ─── Drawing Helpers ─────────────────────────────────────────────────────────

def draw_rounded_rect(img, x1, y1, x2, y2, radius, color, alpha=0.55):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    cv2.ellipse(overlay, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, -1)
    cv2.ellipse(overlay, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, -1)
    cv2.ellipse(overlay, (x1 + radius, y2 - radius), (radius, radius),  90, 0, 90, color, -1)
    cv2.ellipse(overlay, (x2 - radius, y2 - radius), (radius, radius),   0, 0, 90, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def put_text(img, text, pos, scale=0.55, color=(255, 255, 255), thickness=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


# ─── Draw Bounding box ─────────────────────────────────────────────────────────────────

def draw_track_box(img, box, track_id, cls_name="unknown", color=(0, 255, 0)):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = clamp_bbox(*box, w, h)

    # rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    # label background
    label = f"{cls_name}"
    # label = f"ID {track_id}  {cls_name}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y_text = max(0, y1 - th - 8)
    cv2.rectangle(img, (x1, y_text), (x1 + tw + 10, y_text + th + 8), color, -1)

    # label text
    cv2.putText(img, label, (x1 + 5, y_text + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)



# ─── Colors ─────────────────────────────────────────────────────────────────

BG_COLOR    = (15, 15, 15)
TEXT_WHITE  = (255, 255, 255)
TEXT_YELLOW = (0, 220, 255)
COLOR_ZONE  = (100, 220, 160)
HEADER_COL  = (180, 180, 180)


# ─── UI Panels ──────────────────────────────────────────────────────────────

SIDEBAR_X   = 10
SIDEBAR_Y   = 10
SIDEBAR_W   = 230
ROW_H       = 32
PADDING     = 12
RADIUS      = 10


def draw_verification(img, class_counts, frame_h):
    all_classes = sorted(GROUND_TRUTH.keys())
    n_rows = len(all_classes) + 1

    panel_h = PADDING + n_rows * ROW_H + PADDING + 8
    x1, y1 = SIDEBAR_X, SIDEBAR_Y
    x2, y2 = SIDEBAR_X + SIDEBAR_W, SIDEBAR_Y + panel_h

    draw_rounded_rect(img, x1, y1, x2, y2, RADIUS, BG_COLOR, alpha=0.60)

    hy = y1 + PADDING + 16
    put_text(img, "LIVE COUNT", (x1 + PADDING, hy), scale=0.52, color=HEADER_COL, thickness=1)
    hy += ROW_H

    for cls in all_classes:
        expected = GROUND_TRUTH[cls]
        actual   = class_counts.get(cls, 0)

        put_text(img, cls.upper(), (x1 + PADDING, hy), scale=0.55, color=TEXT_WHITE, thickness=1)
        put_text(img, f"{actual} / {expected}", (x1 + 140, hy), scale=0.58, color=TEXT_YELLOW, thickness=1)
        hy += ROW_H


def draw_zone(img, zone, triggered):
    zx1, zy1, zx2, zy2 = zone

    if triggered:
        overlay = img.copy()
        cv2.rectangle(overlay, (zx1, zy1), (zx2, zy2), COLOR_ZONE, -1)
        cv2.addWeighted(overlay, 0.08, img, 0.92, 0, img)

    tick = 24
    lw   = 2
    col  = COLOR_ZONE if triggered else (180, 180, 180)

    for (cx, cy, dx, dy) in [
        (zx1, zy1,  1,  1),
        (zx2, zy1, -1,  1),
        (zx1, zy2,  1, -1),
        (zx2, zy2, -1, -1),
    ]:
        cv2.line(img, (cx, cy), (cx + dx * tick, cy), col, lw, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * tick), col, lw, cv2.LINE_AA)


# ─── ByteTrack Args ──────────────────────────────────────────────────────────

class BTArgs:
    track_thresh = 0.80
    track_buffer = 120
    match_thresh = 0.85
    mot20 = False


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    input_video_path  = "./input/original/vid-5.mp4"
    output_video_path = "./output/V4/Output-4-with-ByteTracking.mp4"
    device = "cuda:0"

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_video_path}")

    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0

    window_name = "Segmented Frame with Tracking"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    tracker = BYTETracker(BTArgs(), frame_rate=fps)

    zone                 = (576, 12, 1462, 1072)
    class_counts         = defaultdict(int)

    counted_ids          = set()
    was_in_zone          = {}
    track_id_to_class    = {}

    frame_idx            = 0
    zone_flash_remaining = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        zone_triggered = False

        results = yolo_segmentation(
            image=frame,
            device=device,
            show_confidence=False,
            show_boxes=False,
            conf_threshold=0.75,
            show_masks=False,
            show_labels=True,
        )

        processed = results.get("processed_image", None)
        if processed is None:
            processed = frame.copy()

        if (processed.shape[1], processed.shape[0]) != (frame_width, frame_height):
            processed = cv2.resize(processed, (frame_width, frame_height))

        raw_detections = results.get("detections", [])

        det_boxes = []
        det_scores = []
        det_classes = []

        for det in raw_detections:
            if isinstance(det, dict) and "bbox" in det:
                bbox = det["bbox"]
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = bbox[:4]
                    det_boxes.append([float(x1), float(y1), float(x2), float(y2)])
                    det_scores.append(get_conf(det))
                    det_classes.append(get_class_name(det))

        # blur all detected regions (same behavior as your SORT version)
        for b in det_boxes:
            processed = blur_bbox(processed, b, ksize=51, sigma=20)

        # ByteTrack input format: Nx5 [x1,y1,x2,y2,score]
        if len(det_boxes) > 0:
            dets = np.concatenate(
                [np.array(det_boxes, dtype=np.float32), np.array(det_scores, dtype=np.float32)[:, None]],
                axis=1
            )
        else:
            dets = np.zeros((0, 5), dtype=np.float32)

        online_targets = tracker.update(
            dets,
            (frame_height, frame_width),
            (frame_height, frame_width)
        )

        for t in online_targets:
            tlwh = t.tlwh
            track_id = int(t.track_id)

            x1 = float(tlwh[0])
            y1 = float(tlwh[1])
            x2 = float(tlwh[0] + tlwh[2])
            y2 = float(tlwh[1] + tlwh[3])

            track_box = [x1, y1, x2, y2]

            cls_name = track_id_to_class.get(track_id, "unknown")

            # optional choose a color per class
            color = (0, 255, 0)

            draw_track_box(processed, track_box, track_id, cls_name, color=color)

            # Assign class to track using best IoU with current detections
            if len(det_boxes) > 0:
                ious = [iou(track_box, db) for db in det_boxes]
                best = int(np.argmax(ious))
                if ious[best] > 0.1:
                    track_id_to_class[track_id] = det_classes[best]

            cx, cy = bbox_center(track_box)
            in_zone = point_in_rect((cx, cy), zone)

            prev_in_zone = was_in_zone.get(track_id, False)
            was_in_zone[track_id] = in_zone

            # Count only on the edge transition outside -> inside
            if (not prev_in_zone) and in_zone and (track_id not in counted_ids):
                cls_name = track_id_to_class.get(track_id, "unknown")
                class_counts[cls_name] += 1
                counted_ids.add(track_id)
                zone_triggered = True
                zone_flash_remaining = int(fps * 0.4)

        if not zone_triggered and zone_flash_remaining > 0:
            zone_flash_remaining -= 1

        # ── Draw UI ──────────────────────────────────────────────────────────
        draw_zone(processed, zone, zone_flash_remaining > 0)
        draw_verification(processed, class_counts, frame_height)

        out.write(processed)
        cv2.imshow(window_name, processed)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Finished. Output saved to: {output_video_path}")