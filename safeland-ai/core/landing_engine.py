"""Landing Engine Module

Generates candidate landing zones, evaluates spatial clearances, and ranks
potential sites based on multi-factor decision logic.
"""

from typing import List, Dict, Any, Tuple, Optional

# Module-level safe landing threshold constant
SAFE_THRESHOLD = 80


def check_clearance(
    candidate_zone: Dict[str, Any],
    drone_profile: Dict[str, Any],
    frame_shape: Tuple[int, ...],
    estimated_ground_width_m: float = 10.0
) -> Dict[str, Any]:
    """Evaluate whether a candidate landing zone provides sufficient physical clearance for a specified drone profile.

    This is a simplified image-scale assumption for demonstration only.

    Args:
        candidate_zone: Dictionary containing candidate zone data and bounding box.
        drone_profile: Dictionary containing drone specifications.
        frame_shape: Tuple representing frame shape (height, width, ...).
        estimated_ground_width_m: Configured estimated horizontal ground width in meters.

    Returns:
        Dictionary with detailed clearance evaluation results including pass/fail status,
        estimated dimensions, required dimensions, status string, and explanation reason.
    """
    # Interpretation of required_clearance_m:
    # required_clearance_m is interpreted as the total minimum landing-space requirement
    # configured for that drone profile.
    # To guarantee both physical frame and safety boundary requirements are met:
    # required_width_m = max(width_m, required_clearance_m)
    # required_length_m = max(length_m, required_clearance_m)

    result = {
        "passed": False,
        "status": "REJECTED",
        "estimated_width_m": 0.0,
        "estimated_length_m": 0.0,
        "estimated_zone_width_m": 0.0,
        "estimated_zone_height_m": 0.0,
        "required_width_m": 0.0,
        "required_length_m": 0.0,
        "reason": ""
    }

    # 1. Drone profile validation
    if not isinstance(drone_profile, dict) or not drone_profile:
        result["reason"] = "Select a drone profile before clearance assessment."
        return result

    try:
        drone_width = float(drone_profile.get("width_m", drone_profile.get("width", 0.0)))
        drone_length = float(drone_profile.get("length_m", drone_profile.get("length", 0.0)))
        req_clearance = float(
            drone_profile.get(
                "required_clearance_m",
                drone_profile.get("required_clearance", drone_profile.get("clearance_m", 0.0))
            )
        )
    except (ValueError, TypeError):
        result["reason"] = "Malformed drone profile: invalid numerical specifications."
        return result

    if drone_width <= 0 or drone_length <= 0 or req_clearance <= 0:
        result["reason"] = "Malformed drone profile: dimensions must be positive non-zero numbers."
        return result

    required_width_m = round(max(drone_width, req_clearance), 2)
    required_length_m = round(max(drone_length, req_clearance), 2)

    result["required_width_m"] = required_width_m
    result["required_length_m"] = required_length_m

    # 2. Frame shape validation
    if not frame_shape or len(frame_shape) < 2:
        result["reason"] = "Invalid frame shape provided."
        return result

    frame_height_pixels = frame_shape[0]
    frame_width_pixels = frame_shape[1]

    if frame_width_pixels <= 0 or frame_height_pixels <= 0:
        result["reason"] = "Invalid frame dimensions: width and height must be positive non-zero numbers."
        return result

    # 3. Estimated ground width validation
    try:
        ground_width_m = float(estimated_ground_width_m)
    except (ValueError, TypeError):
        ground_width_m = 0.0

    if ground_width_m <= 0:
        result["reason"] = "Invalid estimated ground width: must be greater than zero."
        return result

    # 4. Candidate zone bounding box parsing
    if not isinstance(candidate_zone, dict):
        result["reason"] = "Invalid candidate zone specification."
        return result

    cand_pixel_w = 0.0
    cand_pixel_h = 0.0

    bbox = candidate_zone.get("bbox")
    if bbox is not None:
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                cand_pixel_w = abs(x2 - x1)
                cand_pixel_h = abs(y2 - y1)
            except (ValueError, TypeError):
                cand_pixel_w = 0.0
                cand_pixel_h = 0.0
        elif isinstance(bbox, dict):
            try:
                if "w" in bbox and "h" in bbox:
                    cand_pixel_w = float(bbox["w"])
                    cand_pixel_h = float(bbox["h"])
                elif "width" in bbox and "height" in bbox:
                    cand_pixel_w = float(bbox["width"])
                    cand_pixel_h = float(bbox["height"])
            except (ValueError, TypeError):
                cand_pixel_w = 0.0
                cand_pixel_h = 0.0
    else:
        # Check top-level keys in candidate_zone
        try:
            if "w" in candidate_zone and "h" in candidate_zone:
                cand_pixel_w = float(candidate_zone["w"])
                cand_pixel_h = float(candidate_zone["h"])
            elif "width" in candidate_zone and "height" in candidate_zone:
                cand_pixel_w = float(candidate_zone["width"])
                cand_pixel_h = float(candidate_zone["height"])
        except (ValueError, TypeError):
            cand_pixel_w = 0.0
            cand_pixel_h = 0.0

    if cand_pixel_w <= 0 or cand_pixel_h <= 0:
        result["reason"] = "Invalid candidate zone: zero or negative bounding box dimensions."
        return result

    # 5. Pixel to Estimated Meter Conversion
    # This is a simplified image-scale assumption for demonstration only.
    meters_per_pixel = ground_width_m / frame_width_pixels
    estimated_zone_w_m = round(cand_pixel_w * meters_per_pixel, 2)
    estimated_zone_h_m = round(cand_pixel_h * meters_per_pixel, 2)

    result["estimated_width_m"] = estimated_zone_w_m
    result["estimated_length_m"] = estimated_zone_h_m
    result["estimated_zone_width_m"] = estimated_zone_w_m
    result["estimated_zone_height_m"] = estimated_zone_h_m

    # 6. Dual Orientation Fitting Check
    # Standard orientation: candidate_width >= required_width AND candidate_height >= required_length
    standard_fit = (estimated_zone_w_m >= required_width_m) and (estimated_zone_h_m >= required_length_m)
    # Rotated orientation: candidate_width >= required_length AND candidate_height >= required_width
    rotated_fit = (estimated_zone_w_m >= required_length_m) and (estimated_zone_h_m >= required_width_m)

    if standard_fit or rotated_fit:
        result["passed"] = True
        result["status"] = "PASSED"
        result["reason"] = "Estimated candidate dimensions satisfy the configured drone clearance requirement."
    else:
        result["passed"] = False
        result["status"] = "REJECTED"
        result["reason"] = f"Insufficient clearance: Zone size ({estimated_zone_w_m}m × {estimated_zone_h_m}m) is smaller than required clearance ({required_width_m}m × {required_length_m}m)."

    return result


def evaluate_all_candidate_clearances(
    candidate_zones: List[Dict[str, Any]],
    drone_profile: Dict[str, Any],
    frame_shape: Tuple[int, ...],
    estimated_ground_width_m: float = 10.0
) -> List[Dict[str, Any]]:
    """Evaluate clearance for all proposed candidate landing zones.

    Args:
        candidate_zones: List of candidate landing zone dictionaries.
        drone_profile: Selected drone profile dict.
        frame_shape: Active frame shape tuple (height, width, ...).
        estimated_ground_width_m: Configured ground width scale in meters.

    Returns:
        List of candidate zone result dictionaries augmented with clearance information.
    """
    if not candidate_zones or not isinstance(candidate_zones, list):
        return []

    evaluated_results = []
    zone_labels = ["Zone A", "Zone B", "Zone C", "Zone D"]

    for idx, zone in enumerate(candidate_zones):
        if not isinstance(zone, dict):
            label = zone_labels[idx] if idx < len(zone_labels) else f"Zone {idx + 1}"
            clearance_info = {
                "passed": False,
                "status": "REJECTED",
                "estimated_width_m": 0.0,
                "estimated_length_m": 0.0,
                "estimated_zone_width_m": 0.0,
                "estimated_zone_height_m": 0.0,
                "required_width_m": 0.0,
                "required_length_m": 0.0,
                "reason": "Malformed candidate data."
            }
            evaluated_results.append({
                "label": label,
                "passed": False,
                "status": "REJECTED",
                "estimated_width_m": 0.0,
                "estimated_length_m": 0.0,
                "estimated_zone_width_m": 0.0,
                "estimated_zone_height_m": 0.0,
                "required_width_m": 0.0,
                "required_length_m": 0.0,
                "reason": "Malformed candidate data.",
                "clearance": clearance_info
            })
            continue

        label = zone.get("label") or (zone_labels[idx] if idx < len(zone_labels) else f"Zone {idx + 1}")
        
        clearance_info = check_clearance(
            candidate_zone=zone,
            drone_profile=drone_profile,
            frame_shape=frame_shape,
            estimated_ground_width_m=estimated_ground_width_m
        )

        res_dict = dict(zone)
        res_dict["label"] = label
        res_dict["passed"] = clearance_info["passed"]
        res_dict["status"] = clearance_info["status"]
        res_dict["estimated_width_m"] = clearance_info["estimated_width_m"]
        res_dict["estimated_length_m"] = clearance_info["estimated_length_m"]
        res_dict["estimated_zone_width_m"] = clearance_info["estimated_zone_width_m"]
        res_dict["estimated_zone_height_m"] = clearance_info["estimated_zone_height_m"]
        res_dict["required_width_m"] = clearance_info["required_width_m"]
        res_dict["required_length_m"] = clearance_info["required_length_m"]
        res_dict["reason"] = clearance_info["reason"]
        res_dict["clearance"] = clearance_info

        evaluated_results.append(res_dict)

    return evaluated_results


SAFE_THRESHOLD = 80


def evaluate_landing_decision(
    ranked_zones: List[Dict[str, Any]],
    safe_threshold: int = SAFE_THRESHOLD,
    emergency_mode: bool = False
) -> Dict[str, Any]:
    """Perform final landing decision logic over ranked candidate landing zones.

    A candidate zone is recommended ONLY when:
      - score >= safe_threshold (80)
      AND
      - clearance passed == True

    If no candidate satisfies both conditions:
      - decision = "NO_SAFE_ZONE"
      - If emergency_mode is True and candidate zones exist:
        provides emergency_candidate (the highest ranked fallback zone) without relabeling it as SAFE.

    Args:
        ranked_zones: List of candidate zone dicts scored and ranked by scoring engine.
        safe_threshold: Configurable score threshold required for a normal safe recommendation (default: 80).
        emergency_mode: Toggle boolean to expose least-risk fallback candidate when no zone meets safe threshold.

    Returns:
        Dict containing decision status, recommended_zone label, score, status, action, and emergency info.
    """
    if not isinstance(ranked_zones, list) or not ranked_zones:
        return {
            "decision": "NO_SAFE_ZONE",
            "recommended_zone": None,
            "highest_candidate": None,
            "highest_score": 0,
            "highest_zone": None,
            "score": 0,
            "status": "UNSAFE",
            "action": "Abort Landing / Continue Search",
            "emergency_candidate": None,
            "emergency_reason": None
        }

    # 1. Search for normal recommended safe landing zone
    normal_recommendation = None
    for zone in ranked_zones:
        if not isinstance(zone, dict):
            continue
        clr = zone.get("clearance", {})
        passed = clr.get("passed", False) if isinstance(clr, dict) else zone.get("passed", False)
        score = zone.get("score", 0)

        if passed and score >= safe_threshold:
            normal_recommendation = zone
            break

    if normal_recommendation is not None:
        return {
            "decision": "RECOMMEND",
            "recommended_zone": normal_recommendation.get("label", "Zone A"),
            "zone": normal_recommendation,
            "score": normal_recommendation.get("score", 0),
            "status": normal_recommendation.get("status", "SAFE"),
            "action": "Proceed with operator confirmation",
            "emergency_candidate": None,
            "emergency_reason": None
        }

    # 2. NO SAFE LANDING ZONE Case
    highest_candidate_zone = ranked_zones[0] if ranked_zones else None
    highest_label = highest_candidate_zone.get("label") if highest_candidate_zone else None
    highest_score = highest_candidate_zone.get("score", 0) if highest_candidate_zone else 0

    emergency_cand = None
    emergency_reason = None

    if emergency_mode and highest_candidate_zone:
        emergency_cand = highest_candidate_zone
        
        # Extract short, actual risk reason produced by scoring/clearance engine
        reasons = highest_candidate_zone.get("reasons", [])
        risk_reasons = [
            r for r in reasons
            if not r.startswith("✓") and any(k in r.lower() for k in ["vehicle", "person", "hazard", "clearance", "below", "insufficient", "margin", "risk", "caution", "unsafe", "⚠", "❌"])
        ]
        
        if risk_reasons:
            emergency_reason = risk_reasons[0].replace("⚠ ", "").replace("❌ ", "").strip()
        elif highest_candidate_zone.get("clearance", {}).get("reason"):
            emergency_reason = highest_candidate_zone["clearance"]["reason"]
        elif reasons:
            emergency_reason = reasons[-1].replace("⚠ ", "").replace("❌ ", "").strip()
        else:
            emergency_reason = "Candidate score is below the normal safety threshold."

    return {
        "decision": "NO_SAFE_ZONE",
        "recommended_zone": None,
        "highest_candidate": highest_label,
        "highest_score": highest_score,
        "highest_zone": highest_candidate_zone,
        "score": highest_score,
        "status": "UNSAFE",
        "action": "Abort Landing / Continue Search",
        "emergency_candidate": emergency_cand,
        "emergency_reason": emergency_reason
    }


def hazard_intersects_zone(hazard_bbox: List[int], zone_bbox: List[int], padding: int = 15) -> bool:
    """Determine whether a detected hazard bounding box intersects or overlaps a candidate landing zone.

    Args:
        hazard_bbox: Bounding box [x1, y1, x2, y2] of hazard.
        zone_bbox: Bounding box [x1, y1, x2, y2] of landing zone.
        padding: Pixel safety margin padding applied to hazard bounding box.

    Returns:
        True if hazard bounding box intersects or enters the candidate zone area, False otherwise.
    """
    if not hazard_bbox or not zone_bbox or len(hazard_bbox) != 4 or len(zone_bbox) != 4:
        return False

    try:
        x1_h, y1_h, x2_h, y2_h = [int(v) for v in hazard_bbox]
        x1_z, y1_z, x2_z, y2_z = [int(v) for v in zone_bbox]

        # Expand hazard box with safety padding margin
        hx1 = max(0, x1_h - padding)
        hy1 = max(0, y1_h - padding)
        hx2 = x2_h + padding
        hy2 = y2_h + padding

        # Calculate intersection rectangle coordinates
        ix1 = max(x1_z, hx1)
        iy1 = max(y1_z, hy1)
        ix2 = min(x2_z, hx2)
        iy2 = min(y2_z, hy2)

        inter_w = max(0, ix2 - ix1)
        inter_h = max(0, iy2 - iy1)

        return (inter_w * inter_h) > 0
    except Exception:
        return False



