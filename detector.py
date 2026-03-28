import cv2
from ultralytics import YOLO

# Load once at module level — not inside a function
model = YOLO("yolov8n.pt")  # auto-downloads on first run

# COCO-detectable classes relevant to satellite imagery
SUPPORTED_CLASSES = {"car", "truck", "bus", "person", "airplane", "boat", "motorcycle"}

def detect_objects(image_path: str) -> list:
    """Run YOLO inference. Call this ONCE per image upload."""
    results = model(image_path)[0]
    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        if label not in SUPPORTED_CLASSES:
            continue  # Skip unsupported COCO classes
        x, y, w, h = box.xywh[0].tolist()
        conf = float(box.conf[0])
        detections.append({
            "class": label,
            "x": round(x, 2),
            "y": round(y, 2),
            "w": round(w, 2),
            "h": round(h, 2),
            "confidence": round(conf, 2)
        })
    return detections


def is_low_quality_image(image_path: str) -> bool:
    """Return True if the image is blurry or unreadable (Laplacian variance < 50)."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return True
    variance = cv2.Laplacian(img, cv2.CV_64F).var()
    return variance < 50.0
