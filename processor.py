import math
from typing import Optional

def filter_detections(detections: list, parsed_query: dict, img_w: int = 640, img_h: int = 640) -> list:
    """Filter detections based on parsed query JSON."""
    obj_class = parsed_query.get("object", "car").lower()
    condition = parsed_query.get("condition", "all")
    relation = parsed_query.get("relation", "none")
    target_class = parsed_query.get("target")

    # Step 1: Filter by object class
    matched = [d for d in detections if d["class"].lower() == obj_class]

    if not matched:
        return []

    # Step 2: Apply condition
    radius = min(img_w, img_h) * 0.05
    if condition == "high_density":
        matched = _get_dense_cluster(matched, top_fraction=0.5, radius=radius)
    elif condition == "low_density":
        matched = _get_dense_cluster(matched, top_fraction=0.5, invert=True, radius=radius)

    # Step 3: Apply spatial relation
    if relation == "near" and target_class:
        anchors = [d for d in detections if d["class"].lower() == target_class.lower()]
        if anchors:
            matched = _filter_by_proximity(matched, anchors, top_n=10)

    return matched


def _get_dense_cluster(detections: list, top_fraction: float = 0.5,
                        invert: bool = False, radius: float = 100.0) -> list:
    """Score each detection by local density (nearby detections count)."""

    def density_score(d):
        return sum(
            1 for other in detections
            if other is not d and
            math.dist((d["x"], d["y"]), (other["x"], other["y"])) < radius
        )

    scored = sorted(detections, key=density_score, reverse=not invert)
    cutoff = max(1, int(len(scored) * top_fraction))
    return scored[:cutoff]


def _filter_by_proximity(targets: list, anchors: list, top_n: int = 10) -> list:
    """Return targets closest to the centroid of anchor objects."""
    ax = sum(a["x"] for a in anchors) / len(anchors)
    ay = sum(a["y"] for a in anchors) / len(anchors)
    return sorted(
        targets,
        key=lambda d: math.dist((d["x"], d["y"]), (ax, ay))
    )[:top_n]


def calculate_density_stats(detections: list, img_w: int, img_h: int) -> dict:
    """Return density stats per class."""
    from collections import Counter
    counts = Counter(d["class"] for d in detections)
    area = img_w * img_h
    return {
        cls: {"count": cnt, "per_1000px": round(cnt / area * 1000, 4)}
        for cls, cnt in counts.items()
    }


def calculate_confidence_score(matched: list, total: list) -> tuple:
    """Return (ratio, label) indicating how well the query matched detections."""
    if len(total) == 0:
        return (0.0, "Low")
    ratio = len(matched) / len(total)
    if ratio > 0.6:
        return (ratio, "High")
    elif ratio > 0.3:
        return (ratio, "Medium")
    else:
        return (ratio, "Low")


def get_highest_density_region(detections: list, img_w: int, img_h: int,
                                grid_rows: int = 24, grid_cols: int = 24) -> Optional[dict]:
    """Return pixel bounding box of the grid cell with the most detections."""
    if not detections:
        return None
    cell_h = img_h / grid_rows
    cell_w = img_w / grid_cols
    cell_counts: dict = {}
    for d in detections:
        row = min(int(d["y"] / cell_h), grid_rows - 1)
        col = min(int(d["x"] / cell_w), grid_cols - 1)
        cell_counts[(row, col)] = cell_counts.get((row, col), 0) + 1
    best_cell = max(cell_counts, key=lambda k: cell_counts[k])
    row, col = best_cell
    return {
        "x1": int(col * cell_w),
        "y1": int(row * cell_h),
        "x2": int((col + 1) * cell_w),
        "y2": int((row + 1) * cell_h),
        "count": cell_counts[best_cell],
    }
