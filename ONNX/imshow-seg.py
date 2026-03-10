import cv2
from pathlib import Path

from segmentation_onnx_function import load_segmentation_onnx_model, segment_image_onnx


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

weights_path = "../models/segmentation/yolov5s-seg.onnx"

det_model = load_segmentation_onnx_model(
    weights=str(weights_path),
    device="cuda:0"   # or "cpu"
)


cap = cv2.VideoCapture(0)  # Change to video file path if needed


while True:
    ret, frame = cap.read()
    if not ret:
        break
    result = segment_image_onnx(
        image=frame,
        session=det_model["session"],
        input_name=det_model["input_name"],
        imgsz=det_model["imgsz"],
        class_names=det_model["names"]
    )


    cv2.imshow("frame", result["processed_image"])

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()