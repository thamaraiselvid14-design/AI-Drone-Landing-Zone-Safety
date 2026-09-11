"""Scene Analysis Module

Provides computer vision scene understanding using pretrained YOLOv8n object detection
for hazard identification and OpenCV edge-density heuristics for open landing candidate regions.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import cv2
import os

import streamlit as st

# COCO hazard classes to detect
HAZARD_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}


def is_valid_frame(frame: Any) -> bool:
    """Validate if input is a valid non-empty NumPy array frame."""
    return isinstance(frame, np.ndarray) and frame.size > 0


@st.cache_resource
def load_yolo_model():
    """Lazy load and cache the YOLOv8n model instance on CPU safely."""
    try:
        try:
            import torch
            torch.set_num_threads(2)
        except Exception:
            pass
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        return model
    except Exception:
        return None


def is_yolo_loaded() -> bool:
    """Check if YOLOv8n model is loaded and ready."""
    return load_yolo_model() is not None


def resize_frame_if_needed(frame: np.ndarray, max_dim: int = 640) -> np.ndarray:
    """Resize image preserving aspect ratio if max dimension exceeds max_dim pixels."""
    if not is_valid_frame(frame):
        return frame
    h, w = frame.shape[:2]
    if max(h, w) > max_dim:
        scale = float(max_dim) / float(max(h, w))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return frame


def boxes_overlap(box1: List[int], box2: List[int], padding: int = 15, overlap_threshold: float = 0.15) -> bool:
    """Check if candidate box1 overlaps with hazard box2 (with safety padding margin).

    Args:
        box1: [x1, y1, x2, y2] candidate box.
        box2: [x1, y1, x2, y2] hazard box.
        padding: Safety margin expansion in pixels around hazard.
        overlap_threshold: Overlap ratio threshold.

    Returns:
        True if overlapping beyond threshold, False otherwise.
    """
    x1_a, y1_a, x2_a, y2_a = box1
    x1_b, y1_b, x2_b, y2_b = box2

    # Apply safety margin padding to hazard box
    hx1 = max(0, x1_b - padding)
    hy1 = max(0, y1_b - padding)
    hx2 = x2_b + padding
    hy2 = y2_b + padding

    # Intersection coordinates
    ix1 = max(x1_a, hx1)
    iy1 = max(y1_a, hy1)
    ix2 = min(x2_a, hx2)
    iy2 = min(y2_a, hy2)

    if ix1 < ix2 and iy1 < iy2:
        overlap_area = (ix2 - ix1) * (iy2 - iy1)
        box1_area = max(1, (x2_a - x1_a) * (y2_a - y1_a))
        if (overlap_area / box1_area) > overlap_threshold:
            return True
    return False


def find_candidate_landing_zones(
    frame: np.ndarray,
    hazards: List[Dict[str, Any]],
    max_zones: int = 4
) -> List[Dict[str, Any]]:
    """OpenCV heuristic for discovering open, low-texture candidate landing regions.

    Args:
        frame: BGR NumPy frame array.
        hazards: List of detected hazard dictionaries containing 'bbox'.
        max_zones: Maximum candidate zones to return (2 to 4).

    Returns:
        List of candidate zone dictionaries with 'label' and 'bbox'.
    """
    if not is_valid_frame(frame):
        return []

    h, w = frame.shape[:2]

    # Preprocess frame for open space evaluation
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)

    # Grid candidate generation
    zone_w = int(w * 0.28)
    zone_h = int(h * 0.28)
    step_x = max(20, int(w * 0.12))
    step_y = max(20, int(h * 0.12))

    hazard_boxes = [hz["bbox"] for hz in hazards if "bbox" in hz]

    candidates = []

    for y in range(20, h - zone_h - 20, step_y):
        for x in range(20, w - zone_w - 20, step_x):
            cand_box = [x, y, x + zone_w, y + zone_h]

            # Check overlap against all detected hazards
            has_hazard = False
            for h_box in hazard_boxes:
                if boxes_overlap(cand_box, h_box, padding=20):
                    has_hazard = True
                    break

            if has_hazard:
                continue

            # Calculate edge density & smoothness score in window
            sub_edges = edges[y:y + zone_h, x:x + zone_w]
            edge_density = np.mean(sub_edges)

            sub_gray = gray[y:y + zone_h, x:x + zone_w]
            std_dev = np.std(sub_gray)

            # Lower edge density + moderate std_dev indicates open uniform terrain
            openness_score = 100.0 - (edge_density * 1.5 + std_dev * 0.5)

            candidates.append({
                "bbox": cand_box,
                "score": float(openness_score)
            })

    # Sort candidates by openness score descending
    candidates.sort(key=lambda c: c["score"], reverse=True)

    # Filter candidates with Non-Maximum Suppression (distance check)
    selected_zones = []
    zone_labels = ["Zone A", "Zone B", "Zone C", "Zone D"]

    for cand in candidates:
        if len(selected_zones) >= max_zones:
            break

        c_box = cand["bbox"]
        c_cx = (c_box[0] + c_box[2]) / 2.0
        c_cy = (c_box[1] + c_box[3]) / 2.0

        too_close = False
        for sel in selected_zones:
            s_box = sel["bbox"]
            s_cx = (s_box[0] + s_box[2]) / 2.0
            s_cy = (s_box[1] + s_box[3]) / 2.0
            dist = np.sqrt((c_cx - s_cx) ** 2 + (c_cy - s_cy) ** 2)

            if dist < (zone_w * 0.75):
                too_close = True
                break

        if not too_close:
            label = zone_labels[len(selected_zones)]
            selected_zones.append({
                "label": label,
                "bbox": c_box,
                "score": round(cand["score"], 2)
            })

    return selected_zones


def draw_scene_analysis(
    frame: np.ndarray,
    hazards: List[Dict[str, Any]],
    candidate_zones: List[Dict[str, Any]]
) -> np.ndarray:
    """Draw red hazard boxes and white candidate landing boxes on a copy of the frame.

    Args:
        frame: Original input BGR frame.
        hazards: List of hazard dictionaries.
        candidate_zones: List of candidate zone dictionaries.

    Returns:
        Annotated BGR frame copy.
    """
    if not is_valid_frame(frame):
        return frame

    annotated = frame.copy()

    # 1. Draw HAZARDS in Red (0, 0, 255)
    for hz in hazards:
        bbox = hz.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        class_name = hz.get("class_name", "hazard")
        conf = hz.get("confidence", 0.0)

        # Red bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Tag label text
        label = f"HAZARD: {class_name} {conf:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        tag_y1 = max(0, y1 - text_h - 8)
        tag_y2 = max(text_h + 4, y1)

        cv2.rectangle(annotated, (x1, tag_y1), (x1 + text_w + 6, tag_y2), (0, 0, 255), -1)
        cv2.putText(annotated, label, (x1 + 3, tag_y2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # 2. Draw CANDIDATE ZONES in White (255, 255, 255)
    for zone in candidate_zones:
        bbox = zone.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        label = zone.get("label", "Zone")

        # White bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 255), 2)

        # Label tag
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        tag_y1 = max(0, y1 - text_h - 8)
        tag_y2 = max(text_h + 6, y1)

        cv2.rectangle(annotated, (x1, tag_y1), (x1 + text_w + 10, tag_y2), (255, 255, 255), -1)
        cv2.putText(annotated, label, (x1 + 4, tag_y2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    return annotated


def analyze_scene(frame: np.ndarray, conf_threshold: float = 0.30) -> Dict[str, Any]:
    """Execute YOLOv8n object detection and OpenCV open-space region proposal.

    Args:
        frame: BGR NumPy frame array.
        conf_threshold: Minimum confidence threshold for YOLO detections.

    Returns:
        Dictionary containing detections, hazards, ignored_objects, candidate_zones, and annotated_frame.
    """
    empty_result = {
        "detections": [],
        "hazards": [],
        "ignored_objects": [],
        "candidate_zones": [],
        "annotated_frame": frame if is_valid_frame(frame) else np.array([]),
        "error": None
    }

    if not is_valid_frame(frame):
        empty_result["error"] = "Invalid frame provided."
        empty_result["detection_mode"] = "Prototype Fallback"
        return empty_result

    # Memory optimization: Cap maximum image dimension to 640px for inference
    eval_frame = resize_frame_if_needed(frame, max_dim=640)

    detections = []
    hazards = []
    ignored_objects = []
    detection_mode = "Prototype Fallback"

    model = load_yolo_model()

    if model is not None:
        try:
            results = model(eval_frame, conf=conf_threshold, verbose=False, device="cpu")
            if results and len(results) > 0:
                boxes = results[0].boxes
                names = model.names

                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    class_name = str(names.get(cls_id, f"cls_{cls_id}"))
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = [int(v) for v in xyxy]
                    bbox = [x1, y1, x2, y2]

                    category = "HAZARD" if class_name.lower() in HAZARD_CLASSES else "IGNORE"

                    item = {
                        "class_name": class_name,
                        "confidence": round(conf, 2),
                        "bbox": bbox,
                        "category": category
                    }

                    detections.append(item)
                    if category == "HAZARD":
                        hazards.append(item)
                    else:
                        ignored_objects.append(item)
            detection_mode = "YOLOv8n"
        except Exception as e:
            empty_result["error"] = f"YOLO inference error: {str(e)}"
            detection_mode = "Prototype Fallback"
    else:
        empty_result["error"] = "YOLOv8n model unavailable."
        detection_mode = "Prototype Fallback"

    # Discover open landing candidate regions using OpenCV edge density heuristics
    candidate_zones = find_candidate_landing_zones(eval_frame, hazards, max_zones=4)

    # Annotate frame copy
    annotated_frame = draw_scene_analysis(eval_frame, hazards, candidate_zones)

    return {
        "detections": detections,
        "hazards": hazards,
        "ignored_objects": ignored_objects,
        "candidate_zones": candidate_zones,
        "annotated_frame": annotated_frame,
        "detection_mode": detection_mode,
        "error": empty_result["error"]
    }
