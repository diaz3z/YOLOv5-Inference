from flask import Flask, render_template, Response
from pathlib import Path
import cv2

from camera import Camera
from detection.detection_function import load_detection_model, detect_objects

app = Flask(__name__)

FILE = Path(__file__).resolve()
ROOT = FILE.parent
weights_path = "./models/detection/yolov5s.pt"
# weights_path = (ROOT / "../models/detection/yolov5s.pt").resolve()

detector = load_detection_model(
    weights=str(weights_path),
    device="cuda:0",
    imgsz=640
)

camera = Camera()

@app.route('/')
def index():
    return render_template('camera.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame is None:
            continue

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

        ret, jpeg = cv2.imencode('.jpg', result["annotated_image"])
        if not ret:
            continue

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'
        )

@app.route('/video_feed')
def video_feed():
    return Response(
        gen(camera),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, use_reloader=False)