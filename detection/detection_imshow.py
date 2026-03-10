import cv2
from pathlib import Path

from detection_function import load_detection_model, detect_objects
from DeepSort import load_deepsort_tracker, update_deepsort_tracker
from Sort_tracker import load_sort_tracker, update_sort_tracker
from ByteTracker import load_bytetracker, update_bytetracker

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

weights_path = "../models/detection/yolov5s.pt"

detector = load_detection_model(
    weights=str(weights_path),
    device="cuda:0",   # or "cpu"
    imgsz=640
)

tracker = load_bytetracker(frame_rate=30)
# tracker = load_deepsort_tracker()
# tracker = load_sort_tracker()

cap = cv2.VideoCapture(0)  # Change to video file path if needed


while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = detect_objects(
        image=frame,
        model=detector["model"],
        device=detector["device"],
        imgsz=detector["imgsz"],
        conf_thres=0.25,
        iou_thres=0.45,
        draw_boxes=True,
        show_conf=False
    )

    tracked_objects = update_bytetracker(
    # tracked_objects = update_deepsort_tracker(
    # tracked_objects = update_sort_tracker(
        frame=frame,
        detections=result["detections"],
        tracker=tracker,
        allowed_classes=None,
        class_names=detector["names"]
    )
    annotated = frame.copy()
    for obj in tracked_objects:
        x1, y1, x2, y2 = obj["bbox"]
        track_id = obj["track_id"]
        class_name = obj["class_name"]
        confidence = obj["confidence"]

        label = f"{class_name} ID:{track_id} {confidence:.2f}"
        color = (0, 255, 0)  # Green for tracking
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("frame", result["annotated_image"])
    cv2.imshow("tracked", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()