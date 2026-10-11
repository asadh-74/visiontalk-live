"""YOLO object detection endpoint (fast, local, free) - the 'classic CV' half of the hybrid."""
import io, time, base64
from fastapi import APIRouter
from pydantic import BaseModel
from PIL import Image

router = APIRouter()
_model = None

def model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO("yolov8n.pt")   # auto-downloads (~6 MB) on first run
    return _model

class DetIn(BaseModel):
    image_b64: str
    conf: float = 0.4

@router.post("/api/detect")
def detect(b: DetIn):
    t = time.time()
    img = Image.open(io.BytesIO(base64.b64decode(b.image_b64))).convert("RGB")
    w, h = img.size
    r = model().predict(img, conf=b.conf, imgsz=320, verbose=False)[0]
    objs = [{"label": r.names[int(c)], "conf": round(float(s), 2),
             "box": [round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)]}
            for (x1, y1, x2, y2), c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]
    return {"objects": objs, "ms": round((time.time() - t) * 1000)}
