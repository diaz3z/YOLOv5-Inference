import cv2
from pathlib import Path

from detection_onnx_function import load_detection_onnx_model, detect_objects_onnx


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

weights_path = "../models/detection/yolov5s.onnx"

det_model = load_detection_onnx_model(
    weights=str(weights_path),
    device="cuda:0"   # or "cpu"
)


cap = cv2.VideoCapture(0)  # Change to video file path if needed


while True:
    ret, frame = cap.read()
    if not ret:
        break
    result = detect_objects_onnx(
        image=frame,
        session=det_model["session"],
        input_name=det_model["input_name"],
        imgsz=det_model["imgsz"],
        class_names=det_model["names"]
    )


    cv2.imshow("frame", result["annotated_image"])

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()