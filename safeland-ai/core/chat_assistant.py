"""Grounded Chat Assistant Module for SafeLand AI

Provides grounded, rule-based answers to operator queries using ONLY the live analysis
results stored in st.session_state. Never invents unsupported hazards, scores, or reasons.
"""

import re
from typing import List, Dict, Any, Optional

DISCLAIMER = "SafeLand AI provides prototype landing decision support only. Final landing authority remains with the authorized operator/control system."


def hazard_summary_text(hazards: List[Dict[str, Any]]) -> str:
    """Format actual detected YOLO hazard classes."""
    if not hazards or not isinstance(hazards, list):
        return "No supported dynamic hazards are currently detected in this frame."
    
    classes = []
    for h in hazards:
        if isinstance(h, dict) and "class_name" in h:
            classes.append(str(h["class_name"]).lower())
    
    if not classes:
        return "No supported dynamic hazards are currently detected in this frame."
    
    unique_classes = sorted(list(set(classes)))
    return "Current detected hazards:\n" + "\n".join([f"- {c}" for c in unique_classes])


def find_zone_by_name(zone_name: str, ranked_zones: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Look up a specific candidate landing zone dict by name/label (e.g. 'Zone A', 'A', 'Zone B')."""
    if not ranked_zones or not isinstance(ranked_zones, list):
        return None
    
    target = zone_name.strip().lower()
    for z in ranked_zones:
        if not isinstance(z, dict):
            continue
        label = str(z.get("label", "")).strip().lower()
        if label == target or label.replace("zone ", "") == target.replace("zone ", ""):
            return z
        if f"zone {target}" == label or target == label.replace("zone", "").strip():
            return z
            
    return None


def answer_question(
    question: str,
    ranked_zones: Optional[List[Dict[str, Any]]] = None,
    hazards: Optional[List[Dict[str, Any]]] = None,
    landing_decision: Optional[Dict[str, Any]] = None,
    clearance_results: Optional[List[Dict[str, Any]]] = None,
    emergency_mode: bool = False
) -> str:
    """Answer user question using ONLY the provided live analysis state.

    Args:
        question: User query string.
        ranked_zones: List of candidate zone dicts from current analysis.
        hazards: List of detected hazard dicts.
        landing_decision: Decision result dictionary from landing_engine.
        clearance_results: Candidate clearance evaluation results list.
        emergency_mode: Current Emergency Mode toggle status.

    Returns:
        Formatted markdown response string grounded strictly in live state.
    """
    if not question or not isinstance(question, str):
        return "Please ask a question about the current landing scene."

    # 1. NO ANALYSIS CASE
    if not ranked_zones and not hazards and not landing_decision:
        return "Run AI analysis first so I can answer questions about the current landing scene."

    q_clean = question.strip().lower()
    hazards_list = hazards or []
    zones_list = ranked_zones or []

    # Extract target zone name if present (e.g., "Zone A", "Zone B", "Zone C", "Zone D")
    zone_match = re.search(r'\b(zone\s+[a-d]|zone[a-d]|[a-d])\b', q_clean)
    target_zone_name = None
    if zone_match:
        raw_match = zone_match.group(1).upper()
        if len(raw_match) == 1:
            target_zone_name = f"Zone {raw_match}"
        elif "ZONE" in raw_match and not " " in raw_match:
            target_zone_name = raw_match.replace("ZONE", "Zone ")
        else:
            target_zone_name = raw_match.title()

    target_zone = find_zone_by_name(target_zone_name, zones_list) if target_zone_name else None

    # -------------------------------------------------------------
    # INTENT 1: DETECTED HAZARDS / OBSTACLES
    # -------------------------------------------------------------
    if any(k in q_clean for k in ["hazard", "obstacle", "threat", "object", "car", "person", "truck", "bus"]):
        if not target_zone_name:
            hz_text = hazard_summary_text(hazards_list)
            return f"{hz_text}\n\n*{DISCLAIMER}*"

    # -------------------------------------------------------------
    # INTENT 2: WHICH ZONE IS SAFEST / BEST LANDING ZONE
    # -------------------------------------------------------------
    if any(k in q_clean for k in ["which zone", "safest", "best zone", "top zone", "recommended", "recommend"]):
        if landing_decision and isinstance(landing_decision, dict):
            dec_type = landing_decision.get("decision")
            rec_label = landing_decision.get("recommended_zone")
            top_score = landing_decision.get("score", 0)

            if dec_type == "RECOMMEND" and rec_label:
                return (
                    f"🏆 **{rec_label}** is currently the highest-ranked safe landing candidate with a score of **{top_score}/100 (SAFE)**.\n\n"
                    f"*{DISCLAIMER}*"
                )
            
            em_cand = landing_decision.get("emergency_candidate")
            highest_cand_label = landing_decision.get("highest_candidate", "Zone A")
            highest_cand_score = landing_decision.get("highest_score", 0)

            if emergency_mode and em_cand:
                em_label = em_cand.get("label", highest_cand_label)
                em_score = em_cand.get("score", highest_cand_score)
                em_status = em_cand.get("status", "CAUTION")
                return (
                    f"🔴 SafeLand AI currently has **NO SAFE LANDING ZONE**.\n"
                    f"⚠ **{em_label}** is the current **emergency candidate only** at **{em_score}/100 ({em_status})**. "
                    f"It does NOT meet the normal safety threshold.\n\n"
                    f"*{DISCLAIMER}*"
                )
            else:
                return (
                    f"🔴 SafeLand AI currently has **NO SAFE LANDING ZONE**.\n"
                    f"The highest-scoring candidate is **{highest_cand_label}** at **{highest_cand_score}/100**, "
                    f"which is below the normal safety threshold (80).\n"
                    f"Recommendation: Abort Landing / Continue Search.\n\n"
                    f"*{DISCLAIMER}*"
                )

    # -------------------------------------------------------------
    # INTENT 3: WHY ZONE X? (EXPLANATION)
    # -------------------------------------------------------------
    if "why" in q_clean:
        if target_zone_name and not target_zone:
            return f"**{target_zone_name}** is not present in the current analysis results."
        
        if target_zone:
            z_label = target_zone.get("label", "Zone")
            z_score = target_zone.get("score", 0)
            z_status = target_zone.get("status", "UNSAFE")
            z_reasons = target_zone.get("reasons", [])
            reasons_formatted = "\n".join([f"- {r}" for r in z_reasons]) if z_reasons else "- No specific reasons recorded."

            return (
                f"**{z_label}** currently has a score of **{z_score}/100** and is **{z_status}**.\n\n"
                f"**Reasons:**\n{reasons_formatted}\n\n"
                f"*{DISCLAIMER}*"
            )

    # -------------------------------------------------------------
    # INTENT 4: CAN I LAND AT ZONE X? / IS ZONE X SAFE?
    # -------------------------------------------------------------
    if any(k in q_clean for k in ["can i land", "can the drone land", "is zone", "safe"]):
        if target_zone_name and not target_zone:
            return f"**{target_zone_name}** is not present in the current analysis results."

        if target_zone:
            z_label = target_zone.get("label", "Zone")
            z_score = target_zone.get("score", 0)
            z_status = target_zone.get("status", "UNSAFE")
            clr = target_zone.get("clearance", {})
            clr_passed = clr.get("passed", False) if isinstance(clr, dict) else target_zone.get("passed", False)
            clr_reason = clr.get("reason", "") if isinstance(clr, dict) else ""

            if z_status == "SAFE" and clr_passed:
                return (
                    f"✅ **{z_label}** currently passes prototype safety checks with a score of **{z_score}/100 (SAFE)**. "
                    f"SafeLand AI recommends it for operator review.\n\n"
                    f"*{DISCLAIMER}*"
                )
            elif z_status == "CAUTION":
                reasons = target_zone.get("reasons", [])
                main_reason = reasons[0] if reasons else clr_reason or "Moderate risk factors detected nearby."
                return (
                    f"⚠ **{z_label}** currently scores **{z_score}/100** and is **CAUTION**. "
                    f"It does not meet the normal SafeLand safety threshold (80).\n"
                    f"**Main concern:** {main_reason}\n\n"
                    f"*{DISCLAIMER}*"
                )
            else:  # UNSAFE
                reasons = target_zone.get("reasons", [])
                main_reason = reasons[0] if reasons else clr_reason or "Insufficient clearance or high hazard risk."
                return (
                    f"🔴 **{z_label}** currently scores **{z_score}/100** and is **UNSAFE**. It is not recommended.\n"
                    f"**Main blocking reason:** {main_reason}\n\n"
                    f"*{DISCLAIMER}*"
                )

    # -------------------------------------------------------------
    # INTENT 5: ANALYZE ZONE X
    # -------------------------------------------------------------
    if "analyze" in q_clean or (target_zone_name and target_zone):
        if target_zone_name and not target_zone:
            return f"**{target_zone_name}** is not present in the current analysis results."

        if target_zone:
            z_label = target_zone.get("label", "Zone")
            z_score = target_zone.get("score", 0)
            z_status = target_zone.get("status", "UNSAFE")
            clr = target_zone.get("clearance", {})
            clr_passed = clr.get("passed", False) if isinstance(clr, dict) else target_zone.get("passed", False)
            clr_status = "PASSED" if clr_passed else "REJECTED"
            z_reasons = target_zone.get("reasons", [])
            reasons_formatted = "\n".join([f"- {r}" for r in z_reasons]) if z_reasons else "- No specific reasons recorded."

            return (
                f"### 📍 {z_label} Analysis\n"
                f"- **Score:** {z_score}/100\n"
                f"- **Status:** {z_status}\n"
                f"- **Clearance:** {clr_status}\n\n"
                f"**Main Reasons:**\n{reasons_formatted}\n\n"
                f"*{DISCLAIMER}*"
            )

    # -------------------------------------------------------------
    # GENERAL GROUNDED SUMMARY FALLBACK
    # -------------------------------------------------------------
    if zones_list:
        summary_lines = []
        for z in zones_list[:4]:
            summary_lines.append(f"- **{z.get('label', 'Zone')}**: {z.get('score', 0)}/100 ({z.get('status', 'UNSAFE')})")
        zones_summary = "\n".join(summary_lines)

        return (
            f"Here is the current landing scene summary:\n\n"
            f"{zones_summary}\n\n"
            f"{hazard_summary_text(hazards_list)}\n\n"
            f"You can ask me questions like:\n"
            f"- *Which zone is safest?*\n"
            f"- *Why Zone A?*\n"
            f"- *Can I land at Zone B?*\n"
            f"- *Detected hazards?*\n\n"
            f"*{DISCLAIMER}*"
        )

    return "Run AI analysis first so I can answer questions about the current landing scene."
