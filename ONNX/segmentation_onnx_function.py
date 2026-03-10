import cv2
import numpy as np
import onnxruntime as ort


def load_segmentation_onnx_model(weights, device="cpu", class_names=None):
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


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def process_mask_onnx(proto, masks_in, bboxes, image_shape):
    c, mh, mw = proto.shape
    masks = sigmoid(masks_in @ proto.reshape(c, -1)).reshape(-1, mh, mw)

    processed = []
    ih, iw = image_shape[:2]

    for i, mask in enumerate(masks):
        mask = cv2.resize(mask, (iw, ih))
        x1, y1, x2, y2 = map(int, bboxes[i])
        crop = np.zeros_like(mask, dtype=np.uint8)
        crop[y1:y2, x1:x2] = (mask[y1:y2, x1:x2] > 0.5).astype(np.uint8)
        processed.append(crop)

    return np.array(processed)


def segment_image_onnx(
    image,
    session,
    input_name,
    imgsz=(640, 640),
    show_masks=True,
    show_boxes=True,
    show_labels=True,
    show_confidence=True,
    conf_threshold=0.25,
    iou_threshold=0.45,
    class_names=None,
):
    original_image = image.copy()
    processed_image = image.copy()

    img, ratio, (dw, dh) = letterbox(image, imgsz)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_input = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    img_input = np.expand_dims(img_input, axis=0)

    outputs = session.run(None, {input_name: img_input})

    pred = outputs[0]
    proto = outputs[1]

    if pred.ndim == 3:
        pred = pred[0]
    if proto.ndim == 4:
        proto = proto[0]

    nm = 32
    boxes = pred[:, :4]
    obj_conf = pred[:, 4]
    cls_conf = pred[:, 5:-nm]
    mask_coeff = pred[:, -nm:]

    cls_ids = np.argmax(cls_conf, axis=1)
    cls_scores = cls_conf[np.arange(len(cls_conf)), cls_ids]
    scores = obj_conf * cls_scores

    keep_mask = scores > conf_threshold
    boxes = boxes[keep_mask]
    scores = scores[keep_mask]
    cls_ids = cls_ids[keep_mask]
    mask_coeff = mask_coeff[keep_mask]

    if len(boxes) == 0:
        return {
            "processed_image": processed_image,
            "original_image": original_image,
            "detections": [],
            "detection_count": 0,
            "class_counts": {},
        }

    boxes = xywh2xyxy(boxes)
    boxes[:, [0, 2]] -= dw
    boxes[:, [1, 3]] -= dh
    boxes /= ratio

    h, w = original_image.shape[:2]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h - 1)

    final_indices = []
    for cls in np.unique(cls_ids):
        cls_mask = cls_ids == cls
        cls_boxes = boxes[cls_mask]
        cls_scores_local = scores[cls_mask]
        cls_indices = np.where(cls_mask)[0]
        keep = nms(cls_boxes, cls_scores_local, iou_threshold)
        final_indices.extend(cls_indices[keep])

    final_indices = np.array(final_indices, dtype=int)
    boxes = boxes[final_indices]
    scores = scores[final_indices]
    cls_ids = cls_ids[final_indices]
    mask_coeff = mask_coeff[final_indices]

    masks = process_mask_onnx(proto, mask_coeff, boxes, original_image.shape)

    detections = []
    class_counts = {}

    overlay = processed_image.copy()

    for i, box in enumerate(boxes):
        cls_id = int(cls_ids[i])
        conf = float(scores[i])

        if class_names is None:
            class_name = str(cls_id)
        elif isinstance(class_names, dict):
            class_name = str(class_names[cls_id])
        else:
            class_name = str(class_names[cls_id])

        x1, y1, x2, y2 = map(int, box.tolist())

        class_counts[class_name] = class_counts.get(class_name, 0) + 1

        detections.append({
            "class": cls_id,
            "class_name": class_name,
            "confidence": conf,
            "bbox": [x1, y1, x2, y2],
            "bbox_normalized": [x1 / w, y1 / h, x2 / w, y2 / h],
            "mask": masks[i],
            "mask_id": i,
        })

        if show_masks:
            color = np.array([(cls_id * 37) % 255, (cls_id * 17) % 255, (cls_id * 29) % 255], dtype=np.uint8)
            overlay[masks[i] > 0] = overlay[masks[i] > 0] * 0.5 + color * 0.5

        if show_boxes:
            label = None
            if show_labels:
                label = f"{class_name} {conf:.2f}" if show_confidence else class_name

            cv2.rectangle(processed_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if label is not None:
                cv2.putText(
                    processed_image,
                    label,
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

    if show_masks:
        processed_image = cv2.addWeighted(overlay, 0.6, processed_image, 0.4, 0)

    return {
        "processed_image": processed_image,
        "original_image": original_image,
        "detections": detections,
        "detection_count": len(detections),
        "class_counts": class_counts,
    }