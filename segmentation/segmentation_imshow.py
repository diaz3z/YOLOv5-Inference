import cv2
from pathlib import Path

from segmentation_function import load_segmentation_model, segment_image

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

# weights_path = ROOT / "models" / "segmentation" / "yolov5s-seg.pt"
weights_path = "../models/segmentation/yolov5s-seg.pt"

segmenter = load_segmentation_model(
    weights=str(weights_path),
    device="cuda:0",
    imgsz=(640, 640),
    fp16=False
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = segment_image(
        image=frame,
        model=segmenter["model"],
        device=segmenter["device"],
        imgsz=segmenter["imgsz"],
        show_masks=True,
        show_boxes=True,
        show_labels=True,
        show_confidence=False,
        conf_threshold=0.25,
        iou_threshold=0.45,
    )

    cv2.imshow("segmentation", result["processed_image"])

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()