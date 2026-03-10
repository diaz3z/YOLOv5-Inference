from classification_function import load_classification_model, classify_image
import cv2

# Load once
cls_bundle = load_classification_model(
    weights="../models/classification/yolov5n-cls.pt",
    device="cuda:0",
    imgsz=224,
    fp16=False,   # set True if you want and your backend supports it
)

# Inference many times
frame = cv2.imread("../img/road4.jpg")

result = classify_image(
    image=frame,
    model=cls_bundle["model"],
    device=cls_bundle["device"],
    imgsz=cls_bundle["imgsz"],
    topk=3,
    draw_label=True,
)

cv2.imwrite("../results/road3_result.jpg", result["annotated_image"])
cv2.imshow("frame", result["annotated_image"])
cv2.waitKey(0)
cv2.destroyAllWindows()

print(result["predictions"])