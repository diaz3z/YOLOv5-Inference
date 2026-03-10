import cv2
import numpy as np
import onnxruntime as ort


def load_classification_onnx_model(weights, device="cpu", class_names=None):
    """
    Load ONNX classification model once.

    Args:
        weights (str): Path to ONNX model
        device (str): "cpu" or "cuda"
        class_names (list | dict | None): Optional class names

    Returns:
        dict
    """
    providers = ["CPUExecutionProvider"]
    if str(device).lower() in ["cuda", "cuda:0", "gpu", "0"]:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    session = ort.InferenceSession(weights, providers=providers)
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape

    if isinstance(input_shape[2], int) and isinstance(input_shape[3], int):
        imgsz = (input_shape[2], input_shape[3])
    else:
        imgsz = (224, 224)

    return {
        "session": session,
        "input_name": input_name,
        "imgsz": imgsz,
        "names": class_names,
    }


def softmax(x, axis=1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def classify_image_onnx(
    image,
    session,
    input_name,
    imgsz=(224, 224),
    topk=1,
    draw_label=True,
    class_names=None,
    text_color=(0, 255, 0),
    text_scale=1.0,
    text_thickness=2,
):
    original_image = image.copy()
    annotated_image = image.copy()

    img = cv2.resize(image, (imgsz[1], imgsz[0]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    outputs = session.run(None, {input_name: img})
    pred = outputs[0]

    if pred.ndim == 1:
        pred = np.expand_dims(pred, axis=0)

    probs = softmax(pred, axis=1)
    top_idxs = np.argsort(-probs[0])[:topk]

    predictions = []
    for idx in top_idxs:
        conf = float(probs[0][idx])

        if class_names is None:
            class_name = str(idx)
        elif isinstance(class_names, dict):
            class_name = str(class_names[idx])
        else:
            class_name = str(class_names[idx])

        predictions.append({
            "class": int(idx),
            "class_name": class_name,
            "confidence": conf,
        })

    top1 = predictions[0] if predictions else None

    if draw_label and top1 is not None:
        label = f"{top1['class_name']} {top1['confidence']:.2f}"
        cv2.putText(
            annotated_image,
            label,
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            text_color,
            text_thickness,
            cv2.LINE_AA
        )

    return {
        "predictions": predictions,
        "top1": top1,
        "annotated_image": annotated_image,
        "original_image": original_image,
    }