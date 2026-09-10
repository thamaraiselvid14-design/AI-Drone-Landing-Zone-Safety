import sys
from pathlib import Path

__file_path__ = globals().get("__file__", ".")
PROJECT_ROOT = Path(__file_path__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import os
import numpy as np
import cv2

import core.drone_profiles as drone_profiles
import core.scene_analysis as scene_analysis
import core.landing_engine as landing_engine
import core.scoring as scoring
import core.history as history
import core.input_handler as input_handler
import core.chat_assistant as chat_assistant
import core.voice_assistant as voice_assistant
from typing import List, Dict, Any, Optional
from datetime import datetime


def draw_analysis_dashboard_frame(
    frame: np.ndarray,
    hazards: List[Dict[str, Any]],
    ranked_zones: List[Dict[str, Any]],
    recommended_zone_label: Optional[str] = None
) -> np.ndarray:
    """Draw hazard boxes in red and candidate landing zones color-coded by safety status with ASCII score labels."""
    if not isinstance(frame, np.ndarray) or frame.size == 0:
        return frame

    annotated = frame.copy()

    # 1. Draw HAZARDS in Red (0, 0, 255)
    for hz in hazards:
        bbox = hz.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        class_name = hz.get("class_name", "hazard")
        conf = hz.get("confidence", 0.0)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"HAZARD: {class_name} {conf:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        tag_y1 = max(0, y1 - text_h - 8)
        tag_y2 = max(text_h + 4, y1)

        cv2.rectangle(annotated, (x1, tag_y1), (x1 + text_w + 6, tag_y2), (0, 0, 255), -1)
        cv2.putText(annotated, label, (x1 + 3, tag_y2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # 2. Draw RANKED CANDIDATE ZONES
    for zone in ranked_zones:
        bbox = zone.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        z_label = zone.get("label", "Zone")
        score = zone.get("score", 0)
        status = zone.get("status", "UNSAFE")
        color_name = zone.get("color", "red")

        if color_name == "green" or status == "SAFE":
            bgr_color = (0, 255, 0)
        elif color_name == "amber" or status == "CAUTION":
            bgr_color = (0, 165, 255)
        else:
            bgr_color = (0, 0, 255)

        is_recommended = (recommended_zone_label is not None and z_label == recommended_zone_label)
        thickness = 3 if is_recommended else 2

        cv2.rectangle(annotated, (x1, y1), (x2, y2), bgr_color, thickness)

        if is_recommended:
            tag_text = f"{z_label} - {score} - RECOMMENDED"
        else:
            tag_text = f"{z_label} - {score}"

        (text_w, text_h), baseline = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        tag_y1 = max(0, y1 - text_h - 8)
        tag_y2 = max(text_h + 6, y1)

        cv2.rectangle(annotated, (x1, tag_y1), (x1 + text_w + 10, tag_y2), bgr_color, -1)
        text_color = (0, 0, 0) if (color_name in ["green", "amber"] or status in ["SAFE", "CAUTION"]) else (255, 255, 255)
        cv2.putText(annotated, tag_text, (x1 + 4, tag_y2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2, cv2.LINE_AA)

    return annotated


def draw_selected_area_frame(
    frame: np.ndarray,
    hazards: List[Dict[str, Any]],
    selected_bbox: List[int],
    score_info: Dict[str, Any]
) -> np.ndarray:
    """Draw hazard boxes in red and selected area box color-coded by score status."""
    if not isinstance(frame, np.ndarray) or frame.size == 0:
        return frame

    annotated = frame.copy()

    # 1. Draw HAZARDS in Red (0, 0, 255)
    for hz in hazards:
        bbox = hz.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        class_name = hz.get("class_name", "hazard")
        conf = hz.get("confidence", 0.0)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"HAZARD: {class_name} {conf:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        tag_y1 = max(0, y1 - text_h - 8)
        tag_y2 = max(text_h + 4, y1)

        cv2.rectangle(annotated, (x1, tag_y1), (x1 + text_w + 6, tag_y2), (0, 0, 255), -1)
        cv2.putText(annotated, label, (x1 + 3, tag_y2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # 2. Draw SELECTED AREA
    if selected_bbox and len(selected_bbox) == 4:
        x1, y1, x2, y2 = [int(v) for v in selected_bbox]
        score = score_info.get("score", 0)
        status = score_info.get("status", "UNSAFE")
        color_name = score_info.get("color", "red")

        if color_name == "green" or status == "SAFE":
            bgr_color = (0, 255, 0)
        elif color_name == "amber" or status == "CAUTION":
            bgr_color = (0, 165, 255)
        else:
            bgr_color = (0, 0, 255)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), bgr_color, 3)

        tag_text = f"Selected Area - Score: {score} ({status})"
        (text_w, text_h), baseline = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        tag_y1 = max(0, y1 - text_h - 8)
        tag_y2 = max(text_h + 6, y1)

        cv2.rectangle(annotated, (x1, tag_y1), (x1 + text_w + 10, tag_y2), bgr_color, -1)
        text_color = (0, 0, 0) if (color_name in ["green", "amber"] or status in ["SAFE", "CAUTION"]) else (255, 255, 255)
        cv2.putText(annotated, tag_text, (x1 + 4, tag_y2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)

    return annotated

# Page configuration
st.set_page_config(
    page_title="SafeLand AI",
    page_icon="🚁",
    layout="wide"
)

# Custom CSS for dark command-center aesthetic
custom_css = """
<style>
    /* Dark Command Center Theme Variables */
    :root {
        --bg-main: #080A0F;
        --bg-card: #111520;
        --bg-card-hover: #161B29;
        --border-color: #1E2638;
        --text-primary: #F8FAFC;
        --text-secondary: #94A3B8;
        --text-muted: #64748B;
        --status-safe: #10B981;
        --status-safe-bg: rgba(16, 185, 129, 0.12);
        --status-risky: #F59E0B;
        --status-unsafe: #EF4444;
        --accent-blue: #3B82F6;
    }

    /* Main background & base typography */
    .stApp {
        background-color: var(--bg-main) !important;
        color: var(--text-primary) !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    /* Card Container */
    .content-card {
        background-color: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 32px;
        margin-bottom: 24px;
    }

    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: var(--text-primary);
        margin: 0 0 6px 0;
        letter-spacing: -0.02em;
    }

    .main-subtitle {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--accent-blue);
        margin: 0 0 16px 0;
    }

    .main-description {
        font-size: 0.95rem;
        color: var(--text-secondary);
        max-width: 720px;
        line-height: 1.6;
        margin-bottom: 20px;
    }

    /* Status Badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        background-color: var(--status-safe-bg);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: var(--status-safe);
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 18px;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        background-color: var(--status-safe);
        border-radius: 50%;
        box-shadow: 0 0 8px var(--status-safe);
    }

    .init-message {
        font-size: 0.88rem;
        color: var(--text-secondary);
        background-color: rgba(255, 255, 255, 0.03);
        border-left: 3px solid var(--accent-blue);
        padding: 10px 16px;
        border-radius: 0 8px 8px 0;
    }

    /* Drone Specification Card */
    .drone-spec-card {
        background: linear-gradient(135deg, #111520 0%, #161C2C 100%);
        border: 1px solid #232D42;
        border-radius: 14px;
        padding: 24px;
        margin-top: 16px;
    }

    .drone-spec-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 1px solid var(--border-color);
    }

    .drone-name {
        font-size: 1.4rem;
        font-weight: 700;
        color: var(--text-primary);
    }

    .default-badge {
        background-color: rgba(59, 130, 246, 0.15);
        color: var(--accent-blue);
        border: 1px solid rgba(59, 130, 246, 0.4);
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
    }

    .spec-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 16px;
    }

    .spec-item {
        background-color: rgba(0, 0, 0, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 12px;
    }

    .spec-label {
        font-size: 0.75rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }

    .spec-value {
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--text-primary);
    }

    /* Input Card Container */
    .input-intake-card {
        background-color: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 14px;
        padding: 20px;
        height: 100%;
    }

    .input-card-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 6px;
    }

    .input-card-desc {
        font-size: 0.85rem;
        color: var(--text-secondary);
        margin-bottom: 14px;
    }

    /* Top Navigation Button Styling & Contrast Fixes */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"],
    button[data-testid="stBaseButton-primary"] {
        background-color: #FF2B2B !important;
        color: #FFFFFF !important;
        border: 1px solid #FF2B2B !important;
    }

    .stButton > button[kind="primary"] p,
    .stButton > button[data-testid="stBaseButton-primary"] p,
    button[data-testid="stBaseButton-primary"] p,
    .stButton > button[kind="primary"] span,
    .stButton > button[data-testid="stBaseButton-primary"] span,
    button[data-testid="stBaseButton-primary"] span {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    .stButton > button[kind="secondary"],
    .stButton > button[data-testid="stBaseButton-secondary"],
    button[data-testid="stBaseButton-secondary"] {
        background-color: #F1F5F9 !important;
        color: #0F172A !important;
        border: 1px solid #CBD5E1 !important;
    }

    .stButton > button[kind="secondary"] p,
    .stButton > button[data-testid="stBaseButton-secondary"] p,
    button[data-testid="stBaseButton-secondary"] p,
    .stButton > button[kind="secondary"] span,
    .stButton > button[data-testid="stBaseButton-secondary"] span,
    button[data-testid="stBaseButton-secondary"] span {
        color: #0F172A !important;
        font-weight: 600 !important;
    }

    .stButton > button:disabled,
    button[disabled],
    .stButton > button[disabled] {
        background-color: #E2E8F0 !important;
        color: #0F172A !important;
        border: 1px solid #CBD5E1 !important;
        opacity: 0.9 !important;
        cursor: not-allowed !important;
    }

    .stButton > button:disabled p,
    button[disabled] p,
    .stButton > button[disabled] p,
    .stButton > button:disabled span,
    button[disabled] span,
    .stButton > button[disabled] span {
        color: #0F172A !important;
        font-weight: 600 !important;
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# Initialize Session State Variables Safely
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Dashboard"

if "active_page" not in st.session_state:
    st.session_state["active_page"] = st.session_state["current_page"]

if "drone_selection_mode" not in st.session_state:
    st.session_state["drone_selection_mode"] = "Default"

if "current_frame" not in st.session_state:
    st.session_state["current_frame"] = None

if "input_source" not in st.session_state:
    st.session_state["input_source"] = None

if "active_input_type" not in st.session_state:
    st.session_state["active_input_type"] = st.session_state.get("input_source")

if "last_camera_id" not in st.session_state:
    st.session_state["last_camera_id"] = None

if "last_video_frame_idx" not in st.session_state:
    st.session_state["last_video_frame_idx"] = None

if "analysis_requested" not in st.session_state:
    st.session_state["analysis_requested"] = False

if "pending_uploaded_image" not in st.session_state:
    st.session_state["pending_uploaded_image"] = None

if "emergency_mode" not in st.session_state:
    st.session_state["emergency_mode"] = False

if "previous_ranked_zones" not in st.session_state:
    st.session_state["previous_ranked_zones"] = None

if "previous_recommended_zone" not in st.session_state:
    st.session_state["previous_recommended_zone"] = None

if "previous_top_score" not in st.session_state:
    st.session_state["previous_top_score"] = None

if "previous_decision" not in st.session_state:
    st.session_state["previous_decision"] = None

if "previous_hazards" not in st.session_state:
    st.session_state["previous_hazards"] = None

if "re_eval_alerts" not in st.session_state:
    st.session_state["re_eval_alerts"] = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "current_mission_id" not in st.session_state:
    st.session_state["current_mission_id"] = None

if "current_mission_record" not in st.session_state:
    st.session_state["current_mission_record"] = None


def reset_previous_analysis_state():
    """Reset previous analysis state for new input media intakes."""
    st.session_state["previous_ranked_zones"] = None
    st.session_state["previous_recommended_zone"] = None
    st.session_state["previous_top_score"] = None
    st.session_state["previous_decision"] = None
    st.session_state["previous_hazards"] = None
    st.session_state["re_eval_alerts"] = []
    st.session_state["current_mission_id"] = None
    st.session_state["current_mission_record"] = None


def compute_re_evaluation_alerts(
    curr_hazards: List[Dict[str, Any]],
    curr_ranked_zones: List[Dict[str, Any]],
    curr_decision: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Compare current frame analysis with previous analysis in session state and generate re-evaluation alert events."""
    alerts = []
    
    prev_decision = st.session_state.get("previous_decision")
    prev_rec = st.session_state.get("previous_recommended_zone")
    
    if not prev_decision or not isinstance(prev_decision, dict):
        return alerts

    prev_dec_type = prev_decision.get("decision")
    prev_rec_label = prev_rec.get("label") if prev_rec else None
    prev_rec_score = prev_rec.get("score", 0) if prev_rec else 0
    prev_rec_bbox = prev_rec.get("bbox") if prev_rec else None

    curr_dec_type = curr_decision.get("decision")
    curr_rec = curr_decision.get("zone") if curr_dec_type == "RECOMMEND" else None
    curr_rec_label = curr_decision.get("recommended_zone")
    curr_rec_score = curr_decision.get("score", 0) if curr_rec else 0

    # 1. NEW HAZARD INTERSECTION CHECK against previous recommended zone
    if prev_rec_bbox and len(prev_rec_bbox) == 4:
        new_hazards_in_zone = []
        for hz in curr_hazards:
            hz_box = hz.get("bbox")
            hz_class = hz.get("class_name", "hazard")
            if landing_engine.hazard_intersects_zone(hz_box, prev_rec_bbox):
                new_hazards_in_zone.append(hz_class)
        
        if new_hazards_in_zone:
            hz_str = ", ".join(sorted(list(set(new_hazards_in_zone))))
            alerts.append({
                "type": "NEW_OBSTACLE",
                "title": "🚨 NEW OBSTACLE DETECTED",
                "message": f"New obstacle ({hz_str}) detected near previous recommended landing zone ({prev_rec_label}).",
                "color": "amber"
            })

    # 2. DECISION / RECOMMENDATION CHANGE DETECTION
    # CASE A: Same recommended zone, score updated
    if prev_dec_type == "RECOMMEND" and curr_dec_type == "RECOMMEND" and prev_rec_label == curr_rec_label:
        if prev_rec_score != curr_rec_score:
            alerts.append({
                "type": "SCORE_UPDATE",
                "title": f"Zone {prev_rec_label} Safety Score Updated",
                "message": f"Zone {prev_rec_label} safety score updated: {prev_rec_score} → {curr_rec_score}",
                "color": "info"
            })

    # CASE B: Recommendation changed (e.g. Zone A -> Zone B)
    elif prev_dec_type == "RECOMMEND" and curr_dec_type == "RECOMMEND" and prev_rec_label != curr_rec_label:
        prev_z_in_curr = next((z for z in curr_ranked_zones if z.get("label") == prev_rec_label), None)
        curr_score_of_prev = prev_z_in_curr.get("score", 0) if prev_z_in_curr else 0

        alerts.append({
            "type": "RECOMMENDATION_CHANGED",
            "title": "🚨 LANDING RECOMMENDATION CHANGED",
            "message": f"Previous recommended landing zone {prev_rec_label} score dropped: {prev_rec_score} → {curr_score_of_prev}.",
            "new_recommendation": curr_rec_label,
            "prev_zone": prev_rec_label,
            "prev_score": prev_rec_score,
            "curr_prev_score": curr_score_of_prev,
            "color": "red"
        })

    # CASE C: Previous = Zone A, Current = NO SAFE ZONE
    elif prev_dec_type == "RECOMMEND" and curr_dec_type == "NO_SAFE_ZONE":
        prev_z_in_curr = next((z for z in curr_ranked_zones if z.get("label") == prev_rec_label), None)
        curr_score_of_prev = prev_z_in_curr.get("score", 0) if prev_z_in_curr else 0

        alerts.append({
            "type": "NO_LONGER_SAFE",
            "title": "🚨 PREVIOUS LANDING ZONE NO LONGER SAFE",
            "message": f"Previous recommended zone {prev_rec_label} score dropped: {prev_rec_score} → {curr_score_of_prev}. Safety threshold not satisfied.",
            "prev_zone": prev_rec_label,
            "prev_score": prev_rec_score,
            "curr_prev_score": curr_score_of_prev,
            "color": "red"
        })

    # CASE D: Previous = NO SAFE ZONE, Current = Zone B (Safe candidate now available)
    elif prev_dec_type == "NO_SAFE_ZONE" and curr_dec_type == "RECOMMEND":
        alerts.append({
            "type": "NEW_SAFE_AVAILABLE",
            "title": "🟢 SAFE LANDING CANDIDATE NOW AVAILABLE",
            "message": f"Valid safe landing zone detected: {curr_rec_label} (Score: {curr_rec_score}/100)",
            "new_recommendation": curr_rec_label,
            "color": "green"
        })

    return alerts

# Sidebar Demo Settings
st.sidebar.markdown("### ⚙️ Operational Settings")
estimated_ground_width_m = st.sidebar.slider(
    "Estimated Ground Width (m)",
    min_value=5.0,
    max_value=50.0,
    value=10.0,
    step=0.5,
    help="Demo scale assumption representing full frame width in meters."
)
st.sidebar.caption("⚠️ Estimated scale for demo purposes — real deployment requires camera calibration, altitude data or depth sensing.")

emergency_mode = st.sidebar.toggle(
    "🚨 Emergency Mode",
    key="emergency_mode",
    help="When enabled, presents the least-risk fallback candidate if no zone meets normal safe thresholds."
)

if emergency_mode:
    st.sidebar.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #F59E0B; margin-top: -6px; margin-bottom: 12px;'>Emergency Mode: ACTIVE</div>", unsafe_allow_html=True)
else:
    st.sidebar.markdown("<div style='font-size: 0.85rem; font-weight: 600; color: #94A3B8; margin-top: -6px; margin-bottom: 12px;'>Emergency Mode: OFF</div>", unsafe_allow_html=True)

# Top Navigation Bar
nav_col1, nav_col2, nav_col3, nav_col4, nav_col5, nav_col6 = st.columns([2, 1, 1.2, 1, 1, 1])

with nav_col1:
    st.markdown("### 🚁 SAFELAND AI")

with nav_col2:
    if st.button("Dashboard", use_container_width=True, type="primary" if st.session_state.get("current_page") == "Dashboard" else "secondary"):
        st.session_state["current_page"] = "Dashboard"
        st.session_state["active_page"] = "Dashboard"
        st.rerun()

with nav_col3:
    if st.button("Drone Profiles", use_container_width=True, type="primary" if st.session_state.get("current_page") == "Drone Profiles" else "secondary"):
        st.session_state["current_page"] = "Drone Profiles"
        st.session_state["active_page"] = "Drone Profiles"
        st.rerun()

with nav_col4:
    if st.button("Missions", use_container_width=True, type="primary" if st.session_state.get("current_page") == "Missions" else "secondary"):
        st.session_state["current_page"] = "Missions"
        st.session_state["active_page"] = "Missions"
        st.rerun()

with nav_col5:
    if st.button("History", use_container_width=True, type="primary" if st.session_state.get("current_page") == "History" else "secondary"):
        st.session_state["current_page"] = "History"
        st.session_state["active_page"] = "History"
        st.rerun()

with nav_col6:
    if st.button("Settings", use_container_width=True, type="primary" if st.session_state.get("current_page") == "Settings" else "secondary"):
        st.session_state["current_page"] = "Settings"
        st.session_state["active_page"] = "Settings"
        st.rerun()

st.markdown("---")

# ==========================================
# PAGE 1: DASHBOARD / HOME SCREEN
# ==========================================
if st.session_state["current_page"] == "Dashboard":

    st.markdown("""
    <div class="content-card">
        <div class="status-badge">
            <span class="status-dot"></span>
            <span>SYSTEM READY</span>
        </div>
        <h1 class="main-title">Hello SafeLand AI</h1>
        <h3 class="main-subtitle">Adaptive &amp; Explainable Drone Landing Intelligence</h3>
        <p class="main-description">
            AI-powered decision support for identifying and evaluating safer drone landing zones.
        </p>
        <div class="init-message">
            Phase 1 — Project foundation initialized successfully.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 1. Active Drone Selection Section
    st.markdown("### 🚁 Active Drone Specification")

    drones = drone_profiles.load_drone_profiles()
    if not isinstance(drones, list):
        drones = []
    drones = [d for d in drones if isinstance(d, dict) and "name" in d and isinstance(d["name"], str)]
    drone_names = [d["name"] for d in drones]
    default_drone_name = drone_profiles.get_default_drone_name()

    ctrl_col1, ctrl_col2 = st.columns([1, 2])

    with ctrl_col1:
        selection_mode = st.radio(
            "Drone Selection Mode",
            ["⭐ Use Default Drone", "🔄 Select Another"],
            key="home_drone_mode_radio"
        )

    selected_drone_name = default_drone_name or (drone_names[0] if drone_names else None)

    with ctrl_col2:
        if selection_mode == "🔄 Select Another" or default_drone_name is None:
            default_index = drone_names.index(default_drone_name) if (default_drone_name and default_drone_name in drone_names) else 0
            selected_drone_name = st.selectbox(
                "Select Operational Drone",
                options=drone_names,
                index=default_index,
                key="home_drone_selectbox"
            )
            if default_drone_name is None and selection_mode == "⭐ Use Default Drone":
                st.warning("No default drone configured. Please select an operational drone or set a default in Drone Profiles.")
        else:
            st.info(f"Using Default Saved Profile: **{default_drone_name}**")

    # Fetch selected drone details
    selected_drone = drone_profiles.get_drone_by_name(selected_drone_name)

    if selected_drone:
        is_default = (selected_drone["name"] == default_drone_name)
        default_tag_html = '<span class="default-badge">⭐ DEFAULT PROFILE</span>' if is_default else ""

        st.markdown(f"""
        <div class="drone-spec-card">
            <div class="drone-spec-header">
                <div>
                    <div class="drone-name">🛸 {selected_drone['name']}</div>
                    <div style="font-size: 0.88rem; color: var(--text-secondary); margin-top: 4px;">
                        Purpose: <strong>{selected_drone.get('purpose', 'General Operation')}</strong>
                    </div>
                </div>
                {default_tag_html}
            </div>
            <div class="spec-grid">
                <div class="spec-item">
                    <div class="spec-label">Width</div>
                    <div class="spec-value">{selected_drone['width_m']} m</div>
                </div>
                <div class="spec-item">
                    <div class="spec-label">Length</div>
                    <div class="spec-value">{selected_drone['length_m']} m</div>
                </div>
                <div class="spec-item">
                    <div class="spec-label">Footprint</div>
                    <div class="spec-value">{round(selected_drone['width_m'] * selected_drone['length_m'], 2)} m²</div>
                </div>
                <div class="spec-item">
                    <div class="spec-label">Required Clearance</div>
                    <div class="spec-value" style="color: var(--accent-blue);">{selected_drone['required_clearance_m']} m</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("No drone profile found.")

    st.markdown("---")

    # 2. Input Intake System Section
    st.markdown("### 📥 Input Source Intake")
    st.markdown("Select an aerial image, extract a drone video frame, or capture a live webcam feed for landing analysis.")

    intake_col1, intake_col2, intake_col3 = st.columns(3)

    # CARD 1: Upload Image
    with intake_col1:
        st.markdown("""
        <div class="input-card-title">📁 Upload Image</div>
        <div class="input-card-desc">Upload aerial image <code>[JPG / PNG]</code></div>
        """, unsafe_allow_html=True)
        
        uploaded_image = st.file_uploader(
            "Upload aerial image",
            type=["jpg", "jpeg", "png"],
            key="image_uploader",
            label_visibility="collapsed"
        )
        
        if uploaded_image is not None:
            file_bytes = np.asarray(
                bytearray(uploaded_image.getvalue()),
                dtype=np.uint8
            )
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if input_handler.is_valid_frame(frame):
                st.session_state["pending_uploaded_image"] = frame
                
                # Explicit confirmation button (Rule 1 & Rule 6)
                if st.button("USE THIS IMAGE", type="primary", key="btn_use_uploaded_image", use_container_width=True):
                    st.session_state["current_frame"] = frame.copy()
                    st.session_state["active_input_type"] = "image"
                    st.session_state["input_source"] = "image"
                    
                    reset_previous_analysis_state()
                    st.session_state.pop("scene_analysis_result", None)
                    st.session_state.pop("clearance_results", None)
                    st.session_state.pop("ranked_zones", None)
                    st.session_state.pop("landing_decision", None)
                    st.session_state.pop("selected_area_result", None)
                    
                    st.success("Uploaded image selected as active frame.")
                    st.toast("Active Source: Uploaded Image")
                    st.rerun()
            else:
                st.error("Unable to read the selected image.")

    # CARD 2: Upload Video
    with intake_col2:
        st.markdown("""
        <div class="input-card-title">🎥 Upload Video</div>
        <div class="input-card-desc">Upload drone footage <code>[MP4]</code></div>
        """, unsafe_allow_html=True)
        
        uploaded_vid = st.file_uploader(
            "Select Drone Footage MP4",
            type=["mp4"],
            key="input_vid_uploader",
            label_visibility="collapsed"
        )
        
        if uploaded_vid is not None:
            total_frames, fps, temp_path, vid_err = input_handler.get_video_metadata(uploaded_vid)
            if total_frames > 0 and temp_path:
                selected_idx = st.slider(
                    "Select Video Frame",
                    min_value=0,
                    max_value=total_frames - 1,
                    value=st.session_state.get("last_video_frame_idx", 0),
                    key="input_vid_slider"
                )
                vid_frame, frame_err = input_handler.extract_video_frame(temp_path, selected_idx)
                
                if input_handler.is_valid_frame(vid_frame):
                    video_slider_changed = (selected_idx != st.session_state.get("last_video_frame_idx"))
                    btn_use_vid = st.button("USE THIS VIDEO FRAME", type="primary", key="btn_use_video_frame", use_container_width=True)
                    
                    if btn_use_vid or video_slider_changed:
                        st.session_state["current_frame"] = vid_frame.copy()
                        st.session_state["active_input_type"] = "video"
                        st.session_state["input_source"] = "video"
                        st.session_state["last_video_frame_idx"] = selected_idx
                        
                        reset_previous_analysis_state()
                        st.session_state.pop("scene_analysis_result", None)
                        st.session_state.pop("clearance_results", None)
                        st.session_state.pop("ranked_zones", None)
                        st.session_state.pop("landing_decision", None)
                        st.session_state.pop("selected_area_result", None)
                        st.rerun()

                    elif st.session_state.get("active_input_type") == "video":
                        st.session_state["current_frame"] = vid_frame
                        st.session_state["last_video_frame_idx"] = selected_idx

                    st.caption(f"Frame {selected_idx + 1} of {total_frames} (FPS: {fps:.1f})")
                elif frame_err:
                    st.error(frame_err)
            elif vid_err:
                st.error(vid_err)

    # CARD 3: Live Camera
    with intake_col3:
        st.markdown("""
        <div class="input-card-title">📷 Live Camera</div>
        <div class="input-card-desc">Use live camera feed <code>[Capture Frame]</code></div>
        """, unsafe_allow_html=True)
        
        camera_file = st.camera_input("Capture live camera frame", key="input_camera_widget", label_visibility="collapsed")
        
        if camera_file is not None:
            current_cam_id = camera_file.file_id if hasattr(camera_file, "file_id") else (camera_file.name + str(len(camera_file.getvalue())))
            cam_captured_new = (current_cam_id != st.session_state.get("last_camera_id"))
            
            cam_frame, cam_err = input_handler.process_camera_input(camera_file)
            
            if input_handler.is_valid_frame(cam_frame):
                btn_use_cam = st.button("USE LIVE CAMERA FRAME", type="primary", key="btn_use_camera_frame", use_container_width=True)
                
                if cam_captured_new or btn_use_cam:
                    st.session_state["current_frame"] = cam_frame.copy()
                    st.session_state["active_input_type"] = "camera"
                    st.session_state["input_source"] = "camera"
                    st.session_state["last_camera_id"] = current_cam_id
                    
                    reset_previous_analysis_state()
                    st.session_state.pop("scene_analysis_result", None)
                    st.session_state.pop("clearance_results", None)
                    st.session_state.pop("ranked_zones", None)
                    st.session_state.pop("landing_decision", None)
                    st.session_state.pop("selected_area_result", None)
                    
                    st.success("Live camera frame captured & selected.")
                    st.toast("Active Source: Live Camera")
                    st.rerun()
                elif st.session_state.get("active_input_type") == "camera":
                    st.session_state["current_frame"] = cam_frame
            else:
                st.error(cam_err or "Unable to process the captured camera frame.")

    st.markdown("---")

    # 3. Current Frame Preview Section
    st.markdown("### 🖼️ CURRENT FRAME")

    current_frame = input_handler.get_current_frame()
    active_src_type = st.session_state.get("active_input_type") or st.session_state.get("input_source")
    
    if input_handler.is_valid_frame(current_frame):
        h, w, _ = current_frame.shape
        source_label_map = {
            "image": "Uploaded Image",
            "video": "Video Frame",
            "camera": "Live Camera"
        }
        source_display = source_label_map.get(active_src_type, "External Intake")

        st.markdown(f"""
        <div class="content-card" style="padding: 24px;">
            <div style="display: flex; gap: 24px; flex-wrap: wrap; margin-bottom: 16px;">
                <div><strong>Active Source:</strong> <span style="color: var(--accent-blue); font-weight: 700;">{source_display}</span></div>
                <div><strong>Resolution:</strong> {w} × {h} pixels</div>
                <div><strong>Status:</strong> <span style="color: var(--status-safe);">Ready for analysis</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        rgb_preview = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
        st.image(rgb_preview, caption=f"Active Frame Preview — {source_display} ({w}×{h})", width="stretch")
    else:
        st.info("No valid intake frame selected yet. Upload an image, select a video frame, or capture from live camera above.")

    st.markdown("---")

    # 4. Start AI Analysis Section
    st.markdown("### 🚀 AI Decision Pipeline")
    
    frame_ready = input_handler.is_valid_frame(current_frame)
    has_prev = bool(st.session_state.get("previous_decision")) and (st.session_state.get("input_source") in ["video", "camera"])
    btn_label = "🔄 RE-ANALYZE CURRENT FRAME" if has_prev else "🚀 START AI ANALYSIS"
    
    if st.button(btn_label, type="primary", disabled=not frame_ready, use_container_width=True, key="btn_start_ai_analysis"):
        with st.spinner("Analyzing scene hazards & evaluating open landing candidates..."):
            analysis_res = scene_analysis.analyze_scene(current_frame)
            st.session_state["scene_analysis_result"] = analysis_res

            hazards = analysis_res.get("hazards", [])
            candidates = analysis_res.get("candidate_zones", [])

            clearance_results = []
            ranked_zones = []
            if candidates and selected_drone and input_handler.is_valid_frame(current_frame):
                clearance_results = landing_engine.evaluate_all_candidate_clearances(
                    candidate_zones=candidates,
                    drone_profile=selected_drone,
                    frame_shape=current_frame.shape,
                    estimated_ground_width_m=estimated_ground_width_m
                )
                st.session_state["clearance_results"] = clearance_results

                ranked_zones = scoring.rank_zones(
                    candidate_zones=candidates,
                    drone_profile=selected_drone,
                    hazards=hazards,
                    clearance_results=clearance_results,
                    frame_shape=current_frame.shape
                )
                st.session_state["ranked_zones"] = ranked_zones

            decision_res = landing_engine.evaluate_landing_decision(
                ranked_zones=ranked_zones,
                safe_threshold=landing_engine.SAFE_THRESHOLD,
                emergency_mode=emergency_mode
            )
            st.session_state["landing_decision"] = decision_res

            # Calculate Re-Evaluation Alerts if previous analysis exists for video or camera
            if st.session_state.get("previous_decision") and st.session_state.get("input_source") in ["video", "camera"]:
                alerts = compute_re_evaluation_alerts(hazards, ranked_zones, decision_res)
                st.session_state["re_eval_alerts"] = alerts
                if alerts:
                    voice_assistant.check_and_speak_critical_alerts(alerts)
            else:
                st.session_state["re_eval_alerts"] = []

            # -------------------------------------------------------------
            # MISSION HISTORY LOGGING & PERSISTENCE
            # -------------------------------------------------------------
            curr_dec_type = decision_res.get("decision")
            rec_label = decision_res.get("recommended_zone")
            rec_score = decision_res.get("score", 0)
            rec_status = decision_res.get("status", "SAFE")
            
            if curr_dec_type == "RECOMMEND":
                current_rec_dict = {"zone": rec_label, "score": rec_score, "status": rec_status}
                current_final_status = "RECOMMENDED"
            elif curr_dec_type == "EMERGENCY_FALLBACK":
                current_rec_dict = {"zone": rec_label, "score": rec_score, "status": "EMERGENCY_CANDIDATE"}
                current_final_status = "EMERGENCY_CANDIDATE"
            else: # NO_SAFE_ZONE
                top_score = ranked_zones[0].get("score", 0) if ranked_zones else 0
                current_rec_dict = {"zone": None, "score": top_score, "status": "NO_SAFE_ZONE"}
                current_final_status = "NO_SAFE_ZONE"

            top_reasons = []
            if ranked_zones:
                top_z = next((z for z in ranked_zones if z.get("label") == rec_label), ranked_zones[0])
                top_reasons = top_z.get("reasons", [])

            is_ongoing_mission = (has_prev and bool(st.session_state.get("current_mission_record")))

            if is_ongoing_mission:
                mission_record = st.session_state["current_mission_record"]
                # Append obstacle update events if alerts exist
                for alert in st.session_state.get("re_eval_alerts", []):
                    a_type = alert.get("type")
                    if a_type in ["NEW_OBSTACLE", "RECOMMENDATION_CHANGED", "NO_LONGER_SAFE", "NEW_SAFE_AVAILABLE"]:
                        msg_str = alert.get("message", "")
                        hz_name = "obstacle"
                        if "(" in msg_str and ")" in msg_str:
                            try:
                                hz_name = msg_str.split("(")[1].split(")")[0]
                            except Exception:
                                pass
                        
                        mission_record["obstacle_updates"].append({
                            "event": a_type,
                            "hazard": hz_name,
                            "previous_zone": alert.get("prev_zone", st.session_state.get("previous_recommended_zone")),
                            "previous_score": alert.get("prev_score", st.session_state.get("previous_top_score", 0)),
                            "new_score": alert.get("curr_prev_score", rec_score),
                            "new_recommendation": alert.get("new_recommendation", rec_label)
                        })
                
                mission_record["final_status"] = current_final_status
                mission_record["final_recommendation"] = current_rec_dict
                mission_record["top_zone_reasons"] = top_reasons
                
                st.session_state["current_mission_record"] = mission_record
                history.update_mission(mission_record)
            else:
                new_mission_id = history.generate_mission_id()
                st.session_state["current_mission_id"] = new_mission_id
                
                mission_record = {
                    "mission_id": new_mission_id,
                    "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                    "drone_used": selected_drone.get("name", "Rescue Drone") if selected_drone else "Rescue Drone",
                    "input_type": st.session_state.get("input_source", "unknown"),
                    "candidate_count": len(candidates),
                    "initial_recommendation": current_rec_dict,
                    "obstacle_updates": [],
                    "final_status": current_final_status,
                    "final_recommendation": current_rec_dict,
                    "top_zone_reasons": top_reasons
                }
                
                st.session_state["current_mission_record"] = mission_record
                history.append_mission(mission_record)

            # Update session state with current analysis AFTER comparison & history logging
            curr_rec = decision_res.get("zone") if decision_res.get("decision") == "RECOMMEND" else None
            st.session_state["previous_ranked_zones"] = ranked_zones
            st.session_state["previous_recommended_zone"] = curr_rec
            st.session_state["previous_top_score"] = decision_res.get("score", 0)
            st.session_state["previous_decision"] = decision_res
            st.session_state["previous_hazards"] = hazards

            st.toast("Scene Understanding & Re-Evaluation complete!")

    # Display Main Analysis Dashboard if analysis results exist
    if "scene_analysis_result" in st.session_state and st.session_state["scene_analysis_result"]:
        res = st.session_state["scene_analysis_result"]
        hazards = res.get("hazards", [])
        candidates = res.get("candidate_zones", [])
        clearance_results = st.session_state.get("clearance_results", [])
        ranked_zones = st.session_state.get("ranked_zones", [])
        decision_res = st.session_state.get("landing_decision", {})

        recommended_zone = decision_res.get("zone") if decision_res.get("decision") == "RECOMMEND" else None
        rec_label = decision_res.get("recommended_zone")

        st.markdown("---")
        st.markdown("### 🎯 MAIN LANDING ANALYSIS DASHBOARD")

        # RENDER DYNAMIC RE-EVALUATION ALERTS BANNER
        if st.session_state.get("re_eval_alerts"):
            for alert in st.session_state["re_eval_alerts"]:
                a_type = alert.get("type")
                a_title = alert.get("title", "")
                a_msg = alert.get("message", "")
                a_new_rec = alert.get("new_recommendation")

                if a_type == "NEW_OBSTACLE":
                    st.warning(f"**{a_title}** — {a_msg}")
                elif a_type == "RECOMMENDATION_CHANGED":
                    prev_z = alert.get("prev_zone", "Zone")
                    prev_s = alert.get("prev_score", 0)
                    curr_s = alert.get("curr_prev_score", 0)
                    st.error(f"""
                    ### {a_title}
                    **{prev_z}: {prev_s} → {curr_s}**
                    
                    🔄 *Re-evaluating landing zones...*
                    
                    🏆 **New recommended landing zone: {a_new_rec}**
                    """)
                elif a_type == "NO_LONGER_SAFE":
                    prev_z = alert.get("prev_zone", "Zone")
                    prev_s = alert.get("prev_score", 0)
                    curr_s = alert.get("curr_prev_score", 0)
                    st.error(f"""
                    ### {a_title}
                    **{prev_z}: {prev_s} → {curr_s}**
                    
                    🔴 **NO SAFE LANDING ZONE**
                    
                    *Abort Landing / Continue Search*
                    """)
                elif a_type == "NEW_SAFE_AVAILABLE":
                    st.success(f"""
                    ### {a_title}
                    🏆 **Recommended Zone: {a_new_rec}**
                    """)
                elif a_type == "SCORE_UPDATE":
                    st.info(a_msg)

        # Analysis Mode Toggle
        analysis_mode = st.radio(
            "Analysis Mode",
            ["Automatic Detection", "Select Area"],
            horizontal=True,
            key="dashboard_analysis_mode_toggle"
        )
        st.session_state["analysis_mode"] = analysis_mode

        # 70% / 30% Column Layout
        left_col, right_col = st.columns([7, 3])

        # -------------------------------------------------------------
        # LEFT COLUMN (70% - VISUAL COMMAND VIEW)
        # -------------------------------------------------------------
        with left_col:
            if analysis_mode == "Automatic Detection":
                annotated_dash_frame = draw_analysis_dashboard_frame(
                    frame=current_frame,
                    hazards=hazards,
                    ranked_zones=ranked_zones,
                    recommended_zone_label=rec_label
                )
                if input_handler.is_valid_frame(annotated_dash_frame):
                    rgb_dash = cv2.cvtColor(annotated_dash_frame, cv2.COLOR_BGR2RGB)
                    st.image(rgb_dash, caption="Main Landing Analysis (Red: Hazards | Green: SAFE | Amber: CAUTION | Red: UNSAFE)", width="stretch")

                # Metric Summary Tiles
                m_col1, m_col2, m_col3 = st.columns(3)
                with m_col1:
                    st.metric("Hazards Detected", len(hazards), delta="Red Boxes" if hazards else "Clear", delta_color="inverse" if hazards else "normal")
                with m_col2:
                    st.metric("Candidate Zones Evaluated", len(candidates), delta=f"{len(ranked_zones)} Ranked")
                with m_col3:
                    st.metric("Detection Engine", "YOLOv8n + Edge Heuristics")

                # Clearance Assessment Breakdown Accordion
                with st.expander("📏 CANDIDATE ZONE CLEARANCE DETAILS", expanded=False):
                    for zone_res in clearance_results:
                        label = zone_res.get("label", "Candidate Zone")
                        clr = zone_res.get("clearance", {})
                        passed = clr.get("passed", False)
                        est_w = clr.get("estimated_zone_width_m", 0.0)
                        est_h = clr.get("estimated_zone_height_m", 0.0)
                        req_w = clr.get("required_width_m", 0.0)
                        req_h = clr.get("required_length_m", 0.0)
                        reason = clr.get("reason", "")

                        status_text = "PASSED" if passed else "REJECTED"
                        status_color = "#10B981" if passed else "#EF4444"

                        st.markdown(f"""
                        <div style="background-color: #111520; border: 1px solid #1E2638; border-radius: 8px; padding: 12px 16px; margin-bottom: 8px; border-left: 3px solid {status_color};">
                            <div style="display: flex; justify-content: space-between; font-weight: 700;">
                                <span>📍 {label}</span>
                                <span style="color: {status_color};">{status_text}</span>
                            </div>
                            <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 4px;">
                                Estimated: <strong>{est_w}m × {est_h}m</strong> | Required: <strong>{req_w}m × {req_h}m</strong>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

            else:  # Select Area Mode
                st.info("🎯 Use the range sliders below to select a custom rectangular area on the aerial frame for real-time safety evaluation.")
                
                h_img, w_img = current_frame.shape[:2]
                sl_col1, sl_col2 = st.columns(2)
                with sl_col1:
                    x_range = st.slider("Horizontal Range (X %)", 0, 100, (30, 60), key="select_area_x_slider")
                with sl_col2:
                    y_range = st.slider("Vertical Range (Y %)", 0, 100, (30, 60), key="select_area_y_slider")

                x1_px = int(w_img * (x_range[0] / 100.0))
                x2_px = int(w_img * (x_range[1] / 100.0))
                y1_px = int(h_img * (y_range[0] / 100.0))
                y2_px = int(h_img * (y_range[1] / 100.0))

                if x2_px - x1_px < 10:
                    x2_px = min(w_img, x1_px + 10)
                if y2_px - y1_px < 10:
                    y2_px = min(h_img, y1_px + 10)

                selected_bbox = [x1_px, y1_px, x2_px, y2_px]
                selected_cand = {"label": "Selected Area", "bbox": selected_bbox}

                sel_clr = landing_engine.check_clearance(selected_cand, selected_drone, current_frame.shape, estimated_ground_width_m)
                sel_score = scoring.score_zone(selected_cand, selected_drone, hazards, sel_clr, current_frame.shape)

                st.session_state["selected_area_result"] = {
                    "candidate": selected_cand,
                    "clearance": sel_clr,
                    "score": sel_score
                }

                annotated_sel_frame = draw_selected_area_frame(
                    frame=current_frame,
                    hazards=hazards,
                    selected_bbox=selected_bbox,
                    score_info=sel_score
                )

                if input_handler.is_valid_frame(annotated_sel_frame):
                    rgb_sel = cv2.cvtColor(annotated_sel_frame, cv2.COLOR_BGR2RGB)
                    st.image(rgb_sel, caption=f"Manual Selected Area ({x2_px - x1_px}×{y2_px - y1_px} px)", width="stretch")

        # -------------------------------------------------------------
        # RIGHT COLUMN (30% - COMPACT INTELLIGENCE CONSOLE)
        # -------------------------------------------------------------
        with right_col:
            if analysis_mode == "Automatic Detection":
                decision = decision_res.get("decision", "NO_SAFE_ZONE")

                if decision == "RECOMMEND" and recommended_zone:
                    rec_name = decision_res.get("recommended_zone", "Zone A")
                    rec_score = decision_res.get("score", 0)
                    rec_status = decision_res.get("status", "SAFE")
                    rec_reasons = recommended_zone.get("reasons", [])

                    reasons_bullets = "".join([f"<li style='margin-bottom: 4px;'>{r}</li>" for r in rec_reasons[:3]])

                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #064E3B 0%, #047857 100%); border: 1px solid #10B981; border-radius: 14px; padding: 20px; margin-bottom: 16px; box-shadow: 0 4px 12px rgba(16,185,129,0.2);">
                        <div style="font-size: 0.85rem; font-weight: 800; color: #A7F3D0; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px;">
                            🟢 RECOMMENDED LANDING ZONE
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px;">
                            <div style="font-size: 1.6rem; font-weight: 800; color: #FFFFFF;">{rec_name}</div>
                            <div style="font-size: 1.4rem; font-weight: 800; color: #34D399;">{rec_score} <span style="font-size: 0.85rem; color: #A7F3D0;">/ 100</span></div>
                        </div>
                        <div style="display: inline-block; background-color: rgba(255,255,255,0.2); color: #FFFFFF; padding: 4px 12px; border-radius: 12px; font-weight: 800; font-size: 0.8rem; margin-bottom: 12px;">
                            STATUS: {rec_status}
                        </div>
                        <ul style="margin: 0; padding-left: 18px; font-size: 0.88rem; color: #ECFDF5; list-style-type: none; margin-bottom: 10px;">
                            {reasons_bullets}
                        </ul>
                        <div style="font-size: 0.85rem; font-weight: 700; color: #D1FAE5; border-top: 1px solid rgba(255,255,255,0.15); padding-top: 8px;">
                            Proceed with operator confirmation.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                else:
                    cand_count = len(candidates)
                    clearance_passed_count = sum(1 for c in clearance_results if (c.get("passed", False) or c.get("clearance", {}).get("passed", False)))

                    if cand_count == 0:
                        card_title = "🔴 NO LANDING CANDIDATE DETECTED"
                        card_subtitle = "🔴 No landing candidate detected in the current frame."
                    elif clearance_passed_count == 0:
                        card_title = "🔴 NO SAFE LANDING ZONE"
                        card_subtitle = "Candidate regions were detected, but none satisfy the selected drone's clearance requirement."
                    else:
                        card_title = "🔴 NO SAFE LANDING ZONE"
                        card_subtitle = "Candidate zones were detected, but none satisfy the required safety threshold."

                    highest_name = decision_res.get("highest_candidate") or (candidates[0].get("label") if candidates else "None")
                    highest_score = decision_res.get("highest_score", 0)

                    highest_info_html = ""
                    if cand_count > 0:
                        highest_info_html = f"""
                        <div style="font-size: 1.05rem; font-weight: 700; color: #FFFFFF; margin-bottom: 4px;">
                            Highest Candidate: <strong>{highest_name}</strong>
                        </div>
                        <div style="font-size: 1.25rem; font-weight: 800; color: #F8FAFC; margin-bottom: 10px;">
                            Score: {highest_score} <span style="font-size: 0.85rem; color: #FCA5A5;">/ 100</span>
                        </div>
                        """

                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #7F1D1D 0%, #991B1B 100%); border: 1px solid #EF4444; border-radius: 14px; padding: 20px; margin-bottom: 16px; box-shadow: 0 4px 12px rgba(239,68,68,0.2);">
                        <div style="font-size: 0.9rem; font-weight: 800; color: #FCA5A5; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px;">
                            {card_title}
                        </div>
                        {highest_info_html}
                        <div style="font-size: 0.88rem; color: #FEE2E2; margin-bottom: 12px; line-height: 1.4;">
                            {card_subtitle}
                        </div>
                        <div style="background-color: rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.15); border-radius: 8px; padding: 10px 14px; color: #FEE2E2; font-size: 0.85rem; font-weight: 700;">
                            Recommendation: Abort Landing / Continue Search
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # EMERGENCY MODE CANDIDATE DISPLAY
                    em_cand = decision_res.get("emergency_candidate")
                    if em_cand is not None:
                        em_name = em_cand.get("label", highest_name)
                        em_score = em_cand.get("score", 0)
                        em_status = em_cand.get("status", "CAUTION")
                        em_reason = decision_res.get("emergency_reason", "Candidate score is below the normal safety threshold.")

                        em_status_clr = "#F59E0B" if em_status == "CAUTION" else "#EF4444"

                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #78350F 0%, #92400E 100%); border: 1px solid #F59E0B; border-radius: 14px; padding: 18px; margin-bottom: 16px; box-shadow: 0 4px 12px rgba(245,158,11,0.2);">
                            <div style="font-size: 0.85rem; font-weight: 800; color: #FDE68A; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px;">
                                ⚠ EMERGENCY CANDIDATE ONLY
                            </div>
                            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px;">
                                <div style="font-size: 1.4rem; font-weight: 800; color: #FFFFFF;">{em_name}</div>
                                <div style="font-size: 1.25rem; font-weight: 800; color: #FDE68A;">{em_score} <span style="font-size: 0.8rem; color: #FCD34D;">/ 100</span></div>
                            </div>
                            <div style="margin-bottom: 10px;">
                                <span style="background-color: rgba(0,0,0,0.3); color: {em_status_clr}; border: 1px solid {em_status_clr}; padding: 3px 10px; border-radius: 10px; font-weight: 800; font-size: 0.78rem;">
                                    STATUS: {em_status}
                                </span>
                            </div>
                            <div style="font-size: 0.85rem; color: #FEF3C7; margin-bottom: 8px;">
                                <strong>Reason:</strong> {em_reason}
                            </div>
                            <div style="font-size: 0.8rem; color: #FCD34D; font-style: italic; border-top: 1px solid rgba(255,255,255,0.15); padding-top: 6px;">
                                This zone does NOT meet the normal SafeLand safety threshold.
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                st.caption("SafeLand AI provides landing decision support only. Final landing authority remains with the authorized operator/control system.")

                # CANDIDATE RANKING LIST
                st.markdown("#### 📊 Candidate Ranking")
                if ranked_zones:
                    ranking_items_html = ""
                    for r_zone in ranked_zones:
                        rank = r_zone.get("rank", 1)
                        label = r_zone.get("label", "Zone")
                        score = r_zone.get("score", 0)
                        status = r_zone.get("status", "UNSAFE")

                        if status == "SAFE":
                            dot_color = "#10B981"
                        elif status == "CAUTION":
                            dot_color = "#F59E0B"
                        else:
                            dot_color = "#EF4444"

                        ranking_items_html += f"""
                        <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 14px; background-color: rgba(0,0,0,0.25); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; margin-bottom: 8px;">
                            <div style="display: flex; align-items: center; gap: 10px;">
                                <span style="font-size: 0.85rem; font-weight: 700; color: #64748B;">#{rank}</span>
                                <span style="display: inline-block; width: 10px; height: 10px; border-radius: 50%; background-color: {dot_color};"></span>
                                <span style="font-weight: 700; color: #F8FAFC;">{label}</span>
                            </div>
                            <div style="font-weight: 800; color: {dot_color};">
                                {score} <span style="font-size: 0.75rem; color: #64748B;">/ 100</span>
                            </div>
                        </div>
                        """
                    st.markdown(f'<div style="background-color: #111520; border: 1px solid #1E2638; border-radius: 14px; padding: 14px; margin-bottom: 16px;">{ranking_items_html}</div>', unsafe_allow_html=True)

                # VIEW EXPLANATION EXPANDER
                with st.expander("🔍 VIEW DETAILED EXPLANATIONS", expanded=False):
                    for r_zone in ranked_zones:
                        z_label = r_zone.get("label", "Zone")
                        z_score = r_zone.get("score", 0)
                        z_status = r_zone.get("status", "UNSAFE")
                        z_reasons = r_zone.get("reasons", [])

                        clr_tag = "#10B981" if z_status == "SAFE" else ("#F59E0B" if z_status == "CAUTION" else "#EF4444")
                        reasons_list = "".join([f"<li style='margin-bottom: 3px;'>{r}</li>" for r in z_reasons])

                        st.markdown(f"""
                        <div style="margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid rgba(255,255,255,0.06);">
                            <div style="display: flex; justify-content: space-between; font-size: 0.95rem; font-weight: 700; margin-bottom: 4px;">
                                <span>{z_label} — {z_score}</span>
                                <span style="color: {clr_tag};">{z_status}</span>
                            </div>
                            <ul style="margin: 0; padding-left: 16px; font-size: 0.82rem; color: #CBD5E1; list-style-type: none;">
                                {reasons_list}
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)

                # -------------------------------------------------------------
                # GROUNDED CHAT & VOICE ASSISTANT SECTION
                # -------------------------------------------------------------
                st.markdown("---")
                st.markdown("#### 💬 Ask SafeLand AI")

                prompt_to_submit = None

                # Voice Command Mic Button & Quick Suggestions Layout
                v_hdr1, v_hdr2 = st.columns([3, 1])
                with v_hdr1:
                    st.caption("Quick Questions:")
                with v_hdr2:
                    voice_clicked = st.button("🎙️ Speak", key="btn_voice_command", use_container_width=True, help="Capture voice command via microphone")

                q_col1, q_col2, q_col3 = st.columns(3)

                with q_col1:
                    if st.button("Why Zone A?", key="chat_btn_why_a", use_container_width=True):
                        prompt_to_submit = "Why Zone A?"

                with q_col2:
                    if st.button("Analyze Zone B", key="chat_btn_analyze_b", use_container_width=True):
                        prompt_to_submit = "Analyze Zone B"

                with q_col3:
                    if st.button("Detected hazards?", key="chat_btn_hazards", use_container_width=True):
                        prompt_to_submit = "Detected hazards?"

                # Voice Recognition Trigger
                if voice_clicked:
                    with st.spinner("🎙️ Listening... Speak your command..."):
                        transcription, err_msg = voice_assistant.listen_and_transcribe(timeout=3)
                    if transcription:
                        st.success(f"🎙️ Voice Command Captured: \"{transcription}\"")
                        prompt_to_submit = transcription
                    else:
                        st.warning(f"⚠️ {err_msg or 'Voice capture unavailable. Please use text chat.'}")

                with st.form("safeland_chat_form", clear_on_submit=True):
                    user_input = st.text_input(
                        "Chat Prompt",
                        placeholder="Ask about landing safety or click 🎙️ Speak...",
                        label_visibility="collapsed",
                        key="chat_input_text"
                    )
                    submit_chat = st.form_submit_button("Send 💬", use_container_width=True, type="primary")
                    if submit_chat and user_input.strip():
                        prompt_to_submit = user_input.strip()

                if prompt_to_submit:
                    st.session_state["chat_history"].append({"role": "user", "content": prompt_to_submit})

                    curr_ranked = st.session_state.get("ranked_zones", [])
                    curr_hazards = res.get("hazards", []) if "scene_analysis_result" in st.session_state and st.session_state["scene_analysis_result"] else []
                    curr_decision = st.session_state.get("landing_decision", {})
                    curr_clearance = st.session_state.get("clearance_results", [])

                    bot_reply = chat_assistant.answer_question(
                        question=prompt_to_submit,
                        ranked_zones=curr_ranked,
                        hazards=curr_hazards,
                        landing_decision=curr_decision,
                        clearance_results=curr_clearance,
                        emergency_mode=emergency_mode
                    )

                    st.session_state["chat_history"].append({"role": "assistant", "content": bot_reply})
                    
                    # Speak assistant's reply aloud offline (non-blocking)
                    voice_assistant.speak_text_offline(bot_reply)

                    st.rerun()

                if st.session_state.get("chat_history"):
                    st.markdown("<div style='max-height: 280px; overflow-y: auto; background-color: #0B0E17; border: 1px solid #1E2638; border-radius: 12px; padding: 12px; margin-top: 10px;'>", unsafe_allow_html=True)
                    for msg in st.session_state["chat_history"][-6:]:
                        role = msg.get("role")
                        content = msg.get("content", "")
                        if role == "user":
                            st.markdown(f"<div style='text-align: right; margin-bottom: 8px;'><span style='background-color: #1E293B; color: #F8FAFC; padding: 6px 12px; border-radius: 12px; font-size: 0.85rem; display: inline-block;'>🧑‍✈️ <strong>Operator:</strong> {content}</span></div>", unsafe_allow_html=True)
                        else:
                            content_html = content.replace("\n", "<br/>")
                            st.markdown(f"<div style='text-align: left; margin-bottom: 8px;'><div style='background-color: #0F172A; border: 1px solid #334155; color: #E2E8F0; padding: 10px 14px; border-radius: 12px; font-size: 0.85rem;'>🤖 <strong>SafeLand Assistant:</strong><br/>{content_html}</div></div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                    if st.button("Clear Chat 🗑️", key="btn_clear_chat", use_container_width=True):
                        st.session_state["chat_history"] = []
                        st.rerun()

            else:  # Select Area Right Panel
                if "selected_area_result" in st.session_state:
                    sel_res = st.session_state["selected_area_result"]
                    sel_clr = sel_res.get("clearance", {})
                    sel_score = sel_res.get("score", {})

                    score_val = sel_score.get("score", 0)
                    status_val = sel_score.get("status", "UNSAFE")
                    color_val = sel_score.get("color", "red")
                    reasons_val = sel_score.get("reasons", [])
                    passed_val = sel_clr.get("passed", False)

                    if color_val == "green" or status_val == "SAFE":
                        badge_clr = "#10B981"
                        badge_bg = "rgba(16, 185, 129, 0.15)"
                    elif color_val == "amber" or status_val == "CAUTION":
                        badge_clr = "#F59E0B"
                        badge_bg = "rgba(245, 158, 11, 0.15)"
                    else:
                        badge_clr = "#EF4444"
                        badge_bg = "rgba(239, 68, 68, 0.15)"

                    st.markdown("#### 📍 Selected Area Intelligence")

                    st.markdown(f"""
                    <div style="background-color: #111520; border: 1px solid #1E2638; border-radius: 14px; padding: 20px; margin-bottom: 16px; border-left: 5px solid {badge_clr};">
                        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 12px;">
                            <div style="font-size: 1.25rem; font-weight: 800; color: #F8FAFC;">Selected Area</div>
                            <div style="font-size: 1.4rem; font-weight: 800; color: {badge_clr};">{score_val} <span style="font-size: 0.85rem; color: #64748B;">/ 100</span></div>
                        </div>
                        <div style="margin-bottom: 14px;">
                            <span style="background-color: {badge_bg}; color: {badge_clr}; border: 1px solid {badge_clr}; padding: 4px 14px; border-radius: 12px; font-weight: 800; font-size: 0.8rem; letter-spacing: 0.05em;">
                                STATUS: {status_val}
                            </span>
                        </div>
                    """, unsafe_allow_html=True)

                    if score_val >= 80 and passed_val:
                        st.success("✅ SELECTED AREA PASSES CURRENT PROTOTYPE SAFETY CHECKS")
                    else:
                        st.error("🔴 SELECTED AREA NOT RECOMMENDED")

                    st.markdown(f"""
                        <div style="margin-top: 14px; font-size: 0.85rem; color: #94A3B8;">
                            <div>Estimated Size: <strong>{sel_clr.get('estimated_zone_width_m', 0.0)}m × {sel_clr.get('estimated_zone_height_m', 0.0)}m</strong></div>
                            <div>Required Clearance: <strong>{sel_clr.get('required_width_m', 0.0)}m × {sel_clr.get('required_length_m', 0.0)}m</strong></div>
                        </div>
                        <div style="margin-top: 14px; font-size: 0.88rem; font-weight: 700; color: #F8FAFC;">Safety Evaluation Reasons:</div>
                        <ul style="margin-top: 6px; padding-left: 18px; font-size: 0.85rem; color: #CBD5E1; list-style-type: none;">
                            {"".join([f"<li style='margin-bottom: 4px;'>{r}</li>" for r in reasons_val])}
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("---")
        st.caption("SafeLand AI provides prototype landing decision support. Final landing authority remains with the authorized operator/control system.")


# ==========================================
# PAGE 2: DRONE PROFILES MANAGEMENT
# ==========================================
elif st.session_state["current_page"] == "Drone Profiles":

    st.markdown("## 🚁 Drone Profiles & Specifications")
    st.markdown("Configure operational UAV dimensions, landing clearance requirements, and default drone profiles.")

    drones = drone_profiles.load_drone_profiles()
    if not isinstance(drones, list):
        drones = []
    drones = [d for d in drones if isinstance(d, dict) and "name" in d and isinstance(d["name"], str)]
    default_drone_name = drone_profiles.get_default_drone_name()

    tab1, tab2 = st.tabs(["📋 Registered Drones", "➕ Add New Drone"])

    # Tab 1: List & Set Default
    with tab1:
        st.markdown("### Registered Fleet Profiles")

        for idx, drone in enumerate(drones):
            is_default = (drone["name"] == default_drone_name)
            
            c1, c2, c3, c4, c5, c6 = st.columns([2, 1.2, 1.2, 1.5, 2.5, 1.5])
            
            with c1:
                st.markdown(f"**🛸 {drone['name']}**")
                if is_default:
                    st.caption("⭐ Default Profile")
            with c2:
                st.markdown(f"**Width:** {drone['width_m']} m")
            with c3:
                st.markdown(f"**Length:** {drone['length_m']} m")
            with c4:
                st.markdown(f"**Clearance:** {drone['required_clearance_m']} m")
            with c5:
                st.markdown(f"*{drone.get('purpose', 'General')}*")
            with c6:
                if is_default:
                    st.button("Active Default", key=f"def_btn_{idx}", disabled=True, use_container_width=True)
                else:
                    if st.button("Set as Default", key=f"set_def_btn_{idx}", use_container_width=True):
                        drone_profiles.set_default_drone_name(drone["name"])
                        st.toast(f"Updated default drone to: {drone['name']}")
                        st.rerun()
            st.divider()

    # Tab 2: Add New Drone Form
    with tab2:
        st.markdown("### Add New Drone Specification")
        with st.form("add_drone_form", clear_on_submit=True):
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                new_name = st.text_input("Drone Name *", placeholder="e.g. Heavy Cargo Drone X4")
                new_width = st.number_input("Width (meters) *", min_value=0.1, max_value=20.0, value=1.5, step=0.1)
            with f_col2:
                new_purpose = st.text_input("Mission Purpose *", placeholder="e.g. High-altitude mapping")
                new_length = st.number_input("Length (meters) *", min_value=0.1, max_value=20.0, value=1.5, step=0.1)

            calculated_default_clearance = drone_profiles.calculate_required_clearance(new_width, new_length)
            new_clearance = st.number_input(
                "Required Clearance Radius (meters)",
                min_value=0.5,
                max_value=50.0,
                value=float(calculated_default_clearance),
                step=0.5,
                help="Recommended minimum radius around landing site clear of obstacles."
            )

            submit_btn = st.form_submit_button("➕ Save Drone Profile", use_container_width=True, type="primary")

            if submit_btn:
                if not new_name.strip():
                    st.error("Please enter a valid Drone Name.")
                elif not new_purpose.strip():
                    st.error("Please enter a Mission Purpose.")
                else:
                    success = drone_profiles.add_drone_profile(
                        name=new_name.strip(),
                        width_m=new_width,
                        length_m=new_length,
                        purpose=new_purpose.strip(),
                        required_clearance_m=new_clearance
                    )
                    if success:
                        st.success(f"Successfully added drone profile: **{new_name.strip()}**!")
                        st.rerun()
                    else:
                        st.error("A drone with this name already exists. Please choose a unique name.")


# ==========================================
# PAGE 3: MISSION HISTORY SCREEN
# ==========================================
elif st.session_state["current_page"] == "History":

    st.markdown("## 📜 Mission History")
    st.markdown("Complete persistent log of SafeLand AI landing evaluations and dynamic re-evaluations.")

    missions = history.load_mission_history()

    if not missions:
        st.info("No mission history yet. Run an analysis to create the first record.")
    else:
        # Display most recent first
        sorted_missions = list(reversed(missions))

        st.markdown("### 📊 Mission Overview")
        
        # Summary Table Data
        table_rows = []
        for m in sorted_missions:
            m_id = m.get("mission_id", "Unknown")
            ts = m.get("timestamp", "")
            formatted_ts = ts.replace("T", " ")[:16] if (ts and len(ts) >= 16) else ts
            
            drone = m.get("drone_used", "Rescue Drone")
            inp = m.get("input_type", "unknown")
            cnt = m.get("candidate_count", 0)
            status = m.get("final_status", "ANALYSIS_COMPLETE")
            
            final_rec = m.get("final_recommendation", {})
            final_zone = final_rec.get("zone") if (isinstance(final_rec, dict) and final_rec.get("zone")) else "None"
            final_score = final_rec.get("score", 0) if isinstance(final_rec, dict) else 0

            table_rows.append({
                "Mission ID": m_id,
                "Timestamp": formatted_ts,
                "Drone": drone,
                "Input": str(inp).upper(),
                "Candidates": cnt,
                "Final Status": status,
                "Final Zone": final_zone,
                "Score": final_score
            })

        st.dataframe(table_rows, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### 🔍 Detailed Mission Logs")

        for m in sorted_missions:
            m_id = m.get("mission_id", "SL-000")
            drone = m.get("drone_used", "Drone")
            ts = m.get("timestamp", "")
            init_rec = m.get("initial_recommendation", {}) if isinstance(m.get("initial_recommendation"), dict) else {}
            obs_updates = m.get("obstacle_updates", []) if isinstance(m.get("obstacle_updates"), list) else []
            final_rec = m.get("final_recommendation", {}) if isinstance(m.get("final_recommendation"), dict) else {}
            final_status = m.get("final_status", "ANALYSIS_COMPLETE")
            reasons = m.get("top_zone_reasons", []) if isinstance(m.get("top_zone_reasons"), list) else []

            init_z = init_rec.get("zone") if init_rec.get("zone") else "No Safe Zone"
            init_score = init_rec.get("score", 0)
            init_status = init_rec.get("status", "UNSAFE")

            final_z = final_rec.get("zone") if final_rec.get("zone") else "No Safe Zone"
            final_score = final_rec.get("score", 0)
            final_st = final_rec.get("status", "UNSAFE")

            with st.expander(f"▼ {m_id} — {drone} ({ts})", expanded=False):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Initial Recommendation:**")
                    if init_rec.get("zone"):
                        st.write(f"🎯 **{init_z}** — Score: **{init_score}** ({init_status})")
                    else:
                        st.write(f"🔴 **NO SAFE LANDING ZONE** (Highest Score: {init_score})")

                with col_b:
                    st.markdown("**Final Status & Recommendation:**")
                    st.write(f"Status: **{final_status}**")
                    if final_rec.get("zone"):
                        st.write(f"🏆 **{final_z}** — Score: **{final_score}** ({final_st})")
                    else:
                        st.write(f"🔴 **NO SAFE LANDING ZONE** (Score: {final_score})")

                st.markdown("---")
                st.markdown("**Obstacle Updates:**")
                if obs_updates:
                    for update in obs_updates:
                        if not isinstance(update, dict):
                            continue
                        evt = update.get("event", "UPDATE")
                        hz = update.get("hazard", "hazard")
                        prev_z = update.get("previous_zone", "Zone")
                        prev_s = update.get("previous_score", 0)
                        new_s = update.get("new_score", 0)
                        new_rec = update.get("new_recommendation", "Zone")
                        st.warning(f"🚨 **{evt}**: {str(hz).capitalize()} detected near {prev_z} (Score: {prev_s} → {new_s}). New recommendation: **{new_rec}**")
                else:
                    st.caption("No obstacle-triggered updates during this mission.")

                st.markdown("---")
                st.markdown("**Top Zone Reasons:**")
                if reasons:
                    for r in reasons:
                        st.markdown(f"- {r}")
                else:
                    st.caption("No top zone reasons recorded.")


# ==========================================
# PAGE 4: MISSIONS OVERVIEW SCREEN
# ==========================================
elif st.session_state["current_page"] == "Missions":

    st.markdown("## 🚁 Missions")
    st.markdown("Active mission status overview and recent flight assessments.")

    # 1. Current / Active Mission Summary Section
    st.markdown("### 🎯 Current Mission Assessment")

    if "scene_analysis_result" in st.session_state and st.session_state["scene_analysis_result"]:
        res = st.session_state["scene_analysis_result"]
        candidates = res.get("candidate_zones", [])
        decision_res = st.session_state.get("landing_decision", {})
        
        active_drone_name = selected_drone.get("name", "Rescue Drone") if selected_drone else "Rescue Drone"
        input_src_val = str(st.session_state.get("input_source", "unknown")).upper()
        cand_count_val = len(candidates)
        
        dec_type = decision_res.get("decision")
        rec_zone_val = decision_res.get("recommended_zone") or decision_res.get("zone")
        rec_score_val = decision_res.get("score", 0)
        rec_status_val = decision_res.get("status", "UNSAFE")

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric("Selected Drone", active_drone_name)
        with m_col2:
            st.metric("Input Source", input_src_val)
        with m_col3:
            st.metric("Candidate Zones", cand_count_val)
        with m_col4:
            if dec_type == "RECOMMEND":
                st.metric("Recommendation", f"{rec_zone_val} ({rec_score_val}/100)", delta=rec_status_val)
            elif dec_type == "EMERGENCY_FALLBACK":
                st.metric("Recommendation", f"🚨 {rec_zone_val} ({rec_score_val}/100)", delta="EMERGENCY")
            else:
                st.metric("Recommendation", "🔴 NO SAFE ZONE", delta="UNSAFE")
    else:
        st.info("No active mission yet. Run an analysis from the Dashboard.")

    st.markdown("---")

    # 2. Compact Recent Missions Section
    st.markdown("### 📜 Recent Mission Records")
    missions = history.load_mission_history()
    if not missions:
        st.caption("No mission history records found.")
    else:
        recent_missions = list(reversed(missions))[:5]
        compact_rows = []
        for rm in recent_missions:
            rm_id = rm.get("mission_id", "Unknown")
            rm_ts = rm.get("timestamp", "")
            rm_ts_formatted = rm_ts.replace("T", " ")[:16] if len(rm_ts) >= 16 else rm_ts
            rm_drone = rm.get("drone_used", "Drone")
            rm_inp = str(rm.get("input_type", "unknown")).upper()
            rm_status = rm.get("final_status", "COMPLETE")
            
            rm_rec = rm.get("final_recommendation", {}) if isinstance(rm.get("final_recommendation"), dict) else {}
            rm_zone = rm_rec.get("zone") if rm_rec.get("zone") else "None"
            rm_score = rm_rec.get("score", 0)

            compact_rows.append({
                "Mission ID": rm_id,
                "Timestamp": rm_ts_formatted,
                "Drone": rm_drone,
                "Input": rm_inp,
                "Status": rm_status,
                "Final Zone": rm_zone,
                "Score": rm_score
            })

        st.dataframe(compact_rows, use_container_width=True, hide_index=True)


# ==========================================
# PAGE 5: OPERATIONAL SETTINGS SCREEN
# ==========================================
elif st.session_state["current_page"] == "Settings":

    st.markdown("## ⚙ Settings")
    st.markdown("Operational environment configuration, emergency mode controls, and landing thresholds.")

    st.markdown("### 📐 1. Estimated Ground Width")
    g_col1, g_col2 = st.columns([1, 2])
    with g_col1:
        st.metric("Current Frame Scale", f"{estimated_ground_width_m} m")
    with g_col2:
        st.info("Adjust full-frame ground width scaling using the **Operational Settings** slider in the sidebar.")

    st.markdown("---")
    st.markdown("### 🚨 2. Emergency Landing Mode")
    e_col1, e_col2 = st.columns([1, 2])
    with e_col1:
        st.metric("Emergency Mode Status", "ACTIVE 🚨" if emergency_mode else "OFF (Normal)")
    with e_col2:
        st.info("Emergency Mode can be toggled via the sidebar control to allow least-risk candidate selection when no zone meets normal safe thresholds.")

    st.markdown("---")
    st.markdown("### 🛡️ 3. Safe Threshold Display")
    s_col1, s_col2 = st.columns([1, 2])
    with s_col1:
        st.metric("Safe Threshold Constant", f"{landing_engine.SAFE_THRESHOLD} / 100")
    with s_col2:
        st.info(f"**Safe Threshold:** `{landing_engine.SAFE_THRESHOLD} / 100`\n\nZones scoring 80–100 are classified as **SAFE**. Scores 50–79 are **CAUTION**, and 0–49 are **UNSAFE**.")

    st.markdown("---")
    st.caption("⚠️ Estimated scale for demo purposes — real deployment requires camera calibration, altitude data or depth sensing.")
