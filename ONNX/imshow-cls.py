from classification_onnx_function import load_classification_onnx_model, classify_image_onnx
import cv2

# Load once
cls_bundle = load_classification_onnx_model(
    weights="../models/classification/yolov5n-cls.onnx",
    device="cuda:0"
)

# Inference many times
frame = cv2.imread("../img/road4.jpg")

result = classify_image_onnx(
    image=frame,
    session=cls_bundle["session"],
    input_name=cls_bundle["input_name"],
    imgsz=cls_bundle["imgsz"],
    topk=3,
    draw_label=True,
)

cv2.imwrite("../results/road3_result.jpg", result["annotated_image"])
cv2.imshow("frame", result["annotated_image"])
cv2.waitKey(0)
cv2.destroyAllWindows()

print(result["predictions"])