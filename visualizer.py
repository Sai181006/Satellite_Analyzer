import cv2
import numpy as np

CLASS_COLORS = {
    "car":        (0,   255,  0),
    "truck":      (0,   200, 255),
    "bus":        (255, 150,  0),
    "person":     (255,   0, 150),
    "airplane":   (100, 100, 255),
    "boat":       (255, 255,   0),
    "motorcycle": (0,   255, 200),
    "default":    (180, 180, 180),
}

def draw_boxes(image_bgr: np.ndarray, all_detections: list,
               highlighted: list, hotspot: dict = None) -> np.ndarray:
    """Draw bounding boxes; highlighted ones are thicker and brighter."""
    img = image_bgr.copy()
    highlighted_set = {id(d) for d in highlighted}

    for d in all_detections:
        x, y, w, h = int(d["x"]), int(d["y"]), int(d["w"]), int(d["h"])
        x1, y1 = x - w // 2, y - h // 2
        x2, y2 = x + w // 2, y + h // 2
        color = CLASS_COLORS.get(d["class"], CLASS_COLORS["default"])
        is_highlighted = id(d) in highlighted_set
        thickness = 3 if is_highlighted else 1

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        if is_highlighted:
            # Semi-transparent fill for highlighted detections
            overlay = img.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)

        label = f'{d["class"]} {d["confidence"]:.2f}'
        cv2.putText(img, label, (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    if hotspot is not None:
        hx1, hy1, hx2, hy2 = hotspot["x1"], hotspot["y1"], hotspot["x2"], hotspot["y2"]
        cv2.rectangle(img, (hx1, hy1), (hx2, hy2), (0, 0, 255), 3)
        cv2.putText(img, "Highest Activity Region", (hx1, max(hy1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

    return img


def generate_heatmap(image_bgr: np.ndarray, detections: list,
                     grid_rows: int = 24, grid_cols: int = 24) -> np.ndarray:
    """Overlay a density heatmap on the image."""
    total = len(detections)
    if total == 0:
        return image_bgr.copy()
    h, w = image_bgr.shape[:2]
    grid = np.zeros((grid_rows, grid_cols), dtype=np.float32)
    cell_h = h / grid_rows
    cell_w = w / grid_cols

    for d in detections:
        row = min(int(d["y"] / cell_h), grid_rows - 1)
        col = min(int(d["x"] / cell_w), grid_cols - 1)
        grid[row][col] += 1 / total

    if grid.max() == 0:
        return image_bgr.copy()  # Nothing to show

    heatmap = cv2.resize(grid, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(image_bgr, 0.55, heatmap_colored, 0.45, 0)
