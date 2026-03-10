import cv2
import numpy as np
import onnxruntime as ort


def load_detection_onnx_model(weights, device="cpu", class_names=None):
    providers = ["CPUExecutionProvider"]
    if str(device).lower() in ["cuda", "cuda:0", "gpu", "0"]:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    session = ort.InferenceSession(weights, providers=providers)
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape

    if isinstance(input_shape[2], int) and isinstance(input_shape[3], int):
        imgsz = (input_shape[2], input_shape[3])
    else:
        imgsz = (640, 640)

    return {
        "session": session,
        "input_name": input_name,
        "imgsz": imgsz,
        "names": class_names,
    }


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    shape = im.shape[:2]

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return im, r, (dw, dh)


def xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def nms(boxes, scores, iou_thres):
    idxs = np.argsort(scores)[::-1]
    keep = []

    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)

        if len(idxs) == 1:
            break

        xx1 = np.maximum(boxes[i, 0], boxes[idxs[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[idxs[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[idxs[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[idxs[1:], 3])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        area1 = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area2 = (boxes[idxs[1:], 2] - boxes[idxs[1:], 0]) * (boxes[idxs[1:], 3] - boxes[idxs[1:], 1])
        union = area1 + area2 - inter + 1e-6
        iou = inter / union

        idxs = idxs[1:][iou < iou_thres]

    return keep


def detect_objects_onnx(
    image,
    session,
    input_name,
    imgsz=(640, 640),
    conf_thres=0.25,
    iou_thres=0.45,
    draw_boxes=True,
    show_conf=False,
    class_names=None,
):
    original_image = image.copy()
    annotated_image = image.copy()

    img, ratio, (dw, dh) = letterbox(image, imgsz)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    outputs = session.run(None, {input_name: img})
    pred = outputs[0]

    if pred.ndim == 3:
        pred = pred[0]

    boxes = pred[:, :4]
    obj_conf = pred[:, 4]
    cls_conf = pred[:, 5:]
    cls_ids = np.argmax(cls_conf, axis=1)
    cls_scores = cls_conf[np.arange(len(cls_conf)), cls_ids]
    scores = obj_conf * cls_scores

    mask = scores > conf_thres
    boxes = boxes[mask]
    scores = scores[mask]
    cls_ids = cls_ids[mask]

    if len(boxes) == 0:
        return {
            "detections": [],
            "annotated_image": annotated_image,
            "original_image": original_image,
        }

    boxes = xywh2xyxy(boxes)

    boxes[:, [0, 2]] -= dw
    boxes[:, [1, 3]] -= dh
    boxes /= ratio

    h, w = original_image.shape[:2]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h - 1)

    final_detections = []

    for cls in np.unique(cls_ids):
        cls_mask = cls_ids == cls
        cls_boxes = boxes[cls_mask]
        cls_scores_local = scores[cls_mask]
        keep = nms(cls_boxes, cls_scores_local, iou_thres)

        for k in keep:
            box = cls_boxes[k]
            conf = float(cls_scores_local[k])
            cls_id = int(cls)

            if class_names is None:
                class_name = str(cls_id)
            elif isinstance(class_names, dict):
                class_name = str(class_names[cls_id])
            else:
                class_name = str(class_names[cls_id])

            x1, y1, x2, y2 = map(int, box.tolist())

            final_detections.append({
                "class": cls_id,
                "class_name": class_name,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
                "bbox_normalized": [x1 / w, y1 / h, x2 / w, y2 / h],
            })

            if draw_boxes:
                label = f"{class_name} {conf:.2f}" if show_conf else class_name
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated_image,
                    label,
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

    return {
        "detections": final_detections,
        "annotated_image": annotated_image,
        "original_image": original_image,
    }