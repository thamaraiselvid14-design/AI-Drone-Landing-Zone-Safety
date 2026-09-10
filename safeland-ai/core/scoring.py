"""Scoring & Zone Ranking Module

Calculates multi-factor safety scores and ranks candidate landing zones based on:
1. Spatial Clearance Result (40%)
2. Distance from Nearest Hazard (25%)
3. Nearby Hazard Count (20%)
4. Zone Area / Landing Margin (15%)
"""

import math
from typing import Dict, List, Any, Tuple, Optional


def _calculate_box_distance(box1: List[float], box2: List[float]) -> float:
    """Calculate minimum pixel distance between two bounding boxes [x1, y1, x2, y2]."""
    x1_a, y1_a, x2_a, y2_a = box1
    x1_b, y1_b, x2_b, y2_b = box2

    dx = max(0.0, max(x1_a - x2_b, x1_b - x2_a))
    dy = max(0.0, max(y1_a - y2_b, y1_b - y2_a))

    return math.sqrt(dx * dx + dy * dy)


def score_zone(
    zone: Dict[str, Any],
    drone_profile: Dict[str, Any],
    hazards: List[Dict[str, Any]],
    clearance_result: Optional[Dict[str, Any]] = None,
    frame_shape: Optional[Tuple[int, ...]] = None
) -> Dict[str, Any]:
    """Score a single candidate landing zone based on clearance, hazard distance, hazard count, and area margin.

    Scoring Weights (100% total):
    1. Clearance Result Factor: 40% (40 points max)
    2. Distance to Nearest Hazard Factor: 25% (25 points max)
    3. Nearby Hazard Count Factor: 20% (20 points max)
    4. Zone Area / Clearance Margin Factor: 15% (15 points max)

    Args:
        zone: Candidate zone dictionary (label, bbox, score).
        drone_profile: Active drone specification dict.
        hazards: List of detected hazard dictionaries (bbox, class_name, confidence).
        clearance_result: Optional pre-computed clearance result dict.
        frame_shape: Optional frame shape tuple (height, width, ...).

    Returns:
        Dictionary containing integer score (0-100), human-readable reasons, status, and color code.
    """
    if clearance_result is None and isinstance(zone, dict):
        clearance_result = zone.get("clearance")

    # Fallback default values
    clr_passed = False
    req_w = 0.0
    req_l = 0.0
    est_w = 0.0
    est_l = 0.0

    if isinstance(clearance_result, dict):
        clr_passed = bool(clearance_result.get("passed", False))
        req_w = float(clearance_result.get("required_width_m", 0.0))
        req_l = float(clearance_result.get("required_length_m", 0.0))
        est_w = float(clearance_result.get("estimated_zone_width_m", clearance_result.get("estimated_width_m", 0.0)))
        est_l = float(clearance_result.get("estimated_zone_height_m", clearance_result.get("estimated_length_m", 0.0)))

    # Frame scale dimensions
    frame_w = 1000.0
    frame_h = 800.0
    if frame_shape and len(frame_shape) >= 2 and frame_shape[0] > 0 and frame_shape[1] > 0:
        frame_h = float(frame_shape[0])
        frame_w = float(frame_shape[1])

    max_dim = max(frame_w, frame_h)

    # Candidate bbox parsing
    cand_bbox = zone.get("bbox") if isinstance(zone, dict) else None
    if cand_bbox and isinstance(cand_bbox, (list, tuple)) and len(cand_bbox) == 4:
        cand_box = [float(cand_bbox[0]), float(cand_bbox[1]), float(cand_bbox[2]), float(cand_bbox[3])]
    else:
        cand_box = [0.0, 0.0, 100.0, 100.0]

    # --- 1. CLEARANCE FACTOR (40% Weight) ---
    clearance_score = 40.0 if clr_passed else 0.0

    # --- 2. HAZARD DISTANCE FACTOR (25% Weight) & 3. NEARBY HAZARD COUNT (20% Weight) ---
    valid_hazards = []
    if hazards and isinstance(hazards, list):
        for h in hazards:
            if isinstance(h, dict) and "bbox" in h:
                hb = h.get("bbox")
                if isinstance(hb, (list, tuple)) and len(hb) == 4:
                    valid_hazards.append({
                        "bbox": [float(hb[0]), float(hb[1]), float(hb[2]), float(hb[3])],
                        "class_name": str(h.get("class_name", "hazard")).lower(),
                        "confidence": float(h.get("confidence", 0.0))
                    })

    reasons = []

    if clr_passed:
        reasons.append("✓ Sufficient landing clearance")
    else:
        reasons.append("✕ Insufficient landing clearance")

    min_hazard_dist = max_dim
    nearby_hazards = []
    overlapping_hazards = []

    near_threshold = max_dim * 0.25  # Proximity threshold radius in pixels

    for h in valid_hazards:
        dist = _calculate_box_distance(cand_box, h["bbox"])
        if dist < min_hazard_dist:
            min_hazard_dist = dist

        if dist == 0.0:
            overlapping_hazards.append(h)
        if dist <= near_threshold:
            nearby_hazards.append(h)

    if not valid_hazards:
        dist_score = 25.0
        count_score = 20.0
        reasons.append("✓ No nearby dynamic hazards")
    else:
        # Distance factor score (0 to 25 points)
        dist_ratio = min(1.0, min_hazard_dist / (max_dim * 0.35))
        dist_score = 25.0 * dist_ratio

        # Count factor score (0 to 20 points)
        num_nearby = len(nearby_hazards)
        if num_nearby == 0:
            count_score = 20.0
            reasons.append("✓ No nearby dynamic hazards")
        elif num_nearby == 1:
            count_score = 10.0
            hz_class = nearby_hazards[0]["class_name"]
            if hz_class in ["car", "bus", "truck"]:
                reasons.append(f"⚠ {hz_class.capitalize()} detected near landing boundary")
            elif hz_class == "person":
                reasons.append("⚠ Person detected near landing zone")
            else:
                reasons.append(f"⚠ {hz_class.capitalize()} detected nearby")
        elif num_nearby == 2:
            count_score = 4.0
            classes = [h["class_name"].capitalize() for h in nearby_hazards]
            reasons.append(f"⚠ Multiple hazards nearby ({', '.join(classes)})")
        else:
            count_score = 0.0
            reasons.append(f"⚠ High hazard density ({num_nearby} hazards detected nearby)")

        if overlapping_hazards:
            hz_class = overlapping_hazards[0]["class_name"].capitalize()
            reasons.append(f"✕ Hazard ({hz_class}) overlaps candidate zone")

    # --- 4. ZONE AREA / MARGIN FACTOR (15% Weight) ---
    est_area = est_w * est_l
    req_area = req_w * req_l

    if not clr_passed or req_area <= 0.0:
        margin_score = 0.0
        if not clr_passed:
            reasons.append("✕ Zone area below clearance requirement")
    else:
        margin_ratio = est_area / req_area
        if margin_ratio >= 1.5:
            margin_score = 15.0
            reasons.append("✓ Large landing margin")
        elif margin_ratio >= 1.0:
            margin_score = 8.0 + 7.0 * ((margin_ratio - 1.0) / 0.5)
            reasons.append("⚠ Limited landing margin")
        else:
            margin_score = 8.0 * margin_ratio
            reasons.append("⚠ Limited landing margin")

    # Calculate Total Score
    total_raw = clearance_score + dist_score + count_score + margin_score
    total_score = int(round(max(0.0, min(100.0, total_raw))))

    # Safety Constraint: If clearance REJECTED, score cannot exceed 49 (UNSAFE)
    if not clr_passed:
        total_score = min(49, total_score)

    # Determine Status and Color
    if total_score >= 80:
        status = "SAFE"
        color = "green"
    elif total_score >= 50:
        status = "CAUTION"
        color = "amber"
    else:
        status = "UNSAFE"
        color = "red"

    # Deduplicate reasons while preserving order
    unique_reasons = []
    for r in reasons:
        if r not in unique_reasons:
            unique_reasons.append(r)

    return {
        "score": total_score,
        "reasons": unique_reasons[:4],
        "status": status,
        "color": color,
        "score_breakdown": {
            "clearance_factor": round(clearance_score, 1),
            "distance_factor": round(dist_score, 1),
            "count_factor": round(count_score, 1),
            "margin_factor": round(margin_score, 1)
        }
    }


def rank_zones(
    candidate_zones: List[Dict[str, Any]],
    drone_profile: Dict[str, Any],
    hazards: List[Dict[str, Any]],
    clearance_results: Optional[List[Dict[str, Any]]] = None,
    frame_shape: Optional[Tuple[int, ...]] = None
) -> List[Dict[str, Any]]:
    """Score and rank all proposed candidate landing zones from safest to riskiest.

    Args:
        candidate_zones: List of candidate zone dictionaries.
        drone_profile: Selected drone profile dict.
        hazards: List of detected hazard dictionaries.
        clearance_results: Optional list of clearance result dictionaries.
        frame_shape: Active frame shape tuple (height, width, ...).

    Returns:
        List of candidate zone result dicts sorted descending by score with 1-based rank.
    """
    if not candidate_zones or not isinstance(candidate_zones, list):
        return []

    # Map clearance results by label or index
    clr_map = {}
    if clearance_results and isinstance(clearance_results, list):
        for idx, clr in enumerate(clearance_results):
            if isinstance(clr, dict):
                label = clr.get("label", f"Zone {idx + 1}")
                clr_map[label] = clr.get("clearance", clr)

    scored_zones = []
    zone_labels = ["Zone A", "Zone B", "Zone C", "Zone D"]

    for idx, zone in enumerate(candidate_zones):
        if not isinstance(zone, dict):
            continue

        label = zone.get("label") or (zone_labels[idx] if idx < len(zone_labels) else f"Zone {idx + 1}")
        clr_res = clr_map.get(label) or zone.get("clearance")

        eval_score = score_zone(
            zone=zone,
            drone_profile=drone_profile,
            hazards=hazards,
            clearance_result=clr_res,
            frame_shape=frame_shape
        )

        zone_entry = dict(zone)
        zone_entry["label"] = label
        zone_entry["score"] = eval_score["score"]
        zone_entry["status"] = eval_score["status"]
        zone_entry["color"] = eval_score["color"]
        zone_entry["reasons"] = eval_score["reasons"]
        zone_entry["score_breakdown"] = eval_score.get("score_breakdown", {})
        if "clearance" not in zone_entry and clr_res:
            zone_entry["clearance"] = clr_res

        scored_zones.append(zone_entry)

    # Sort descending by score
    scored_zones.sort(key=lambda z: z["score"], reverse=True)

    # Assign 1-based rank
    for rank_idx, zone_entry in enumerate(scored_zones, start=1):
        zone_entry["rank"] = rank_idx

    return scored_zones


def calculate_safety_score(zone_features: Dict[str, Any], environmental_factors: Dict[str, Any]) -> float:
    """Legacy helper function for backward compatibility."""
    return 100.0


def categorize_risk_level(safety_score: float) -> Tuple[str, str]:
    """Legacy helper function for backward compatibility."""
    if safety_score >= 80.0:
        return "SAFE", "#10B981"
    elif safety_score >= 50.0:
        return "CAUTION", "#F59E0B"
    else:
        return "UNSAFE", "#EF4444"

