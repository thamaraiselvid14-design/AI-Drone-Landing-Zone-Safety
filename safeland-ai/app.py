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
from typing import List, Dict, Any, Optional


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

if "drone_selection_mode" not in st.session_state:
    st.session_state["drone_selection_mode"] = "Default"

if "current_frame" not in st.session_state:
    st.session_state["current_frame"] = None

if "input_source" not in st.session_state:
    st.session_state["input_source"] = None

if "analysis_requested" not in st.session_state:
    st.session_state["analysis_requested"] = False

if "pending_uploaded_image" not in st.session_state:
    st.session_state["pending_uploaded_image"] = None

if "emergency_mode" not in st.session_state:
    st.session_state["emergency_mode"] = False

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
    if st.button("Dashboard", use_container_width=True, type="primary" if st.session_state["current_page"] == "Dashboard" else "secondary"):
        st.session_state["current_page"] = "Dashboard"
        st.rerun()

with nav_col3:
    if st.button("Drone Profiles", use_container_width=True, type="primary" if st.session_state["current_page"] == "Drone Profiles" else "secondary"):
        st.session_state["current_page"] = "Drone Profiles"
        st.rerun()

with nav_col4:
    if st.button("Missions", use_container_width=True, disabled=True):
        pass

with nav_col5:
    if st.button("History", use_container_width=True, disabled=True):
        pass

with nav_col6:
    if st.button("Settings", use_container_width=True, disabled=True):
        pass

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
                
                # Explicit confirmation button
                if st.button("USE THIS IMAGE", type="primary", key="btn_use_uploaded_image", use_container_width=True):
                    st.session_state["current_frame"] = frame.copy()
                    st.session_state["input_source"] = "image"
                    st.session_state.pop("scene_analysis_result", None)
                    st.session_state.pop("clearance_results", None)
                    st.session_state.pop("ranked_zones", None)
                    st.success("Image selected for analysis.")
                    st.toast("Image selected for analysis.")
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
                    value=0,
                    key="input_vid_slider"
                )
                vid_frame, frame_err = input_handler.extract_video_frame(temp_path, selected_idx)
                if input_handler.is_valid_frame(vid_frame):
                    st.session_state["current_frame"] = vid_frame
                    st.session_state["input_source"] = "video"
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
            cam_frame, cam_err = input_handler.process_camera_input(camera_file)
            if input_handler.is_valid_frame(cam_frame):
                st.session_state["current_frame"] = cam_frame
                st.session_state["input_source"] = "camera"
                st.success("Live camera frame captured successfully.")
            else:
                st.error(cam_err or "Unable to process the captured camera frame.")

    st.markdown("---")

    # 3. Current Frame Preview Section
    st.markdown("### 🖼️ CURRENT FRAME")

    current_frame = input_handler.get_current_frame()
    
    if input_handler.is_valid_frame(current_frame):
        h, w, _ = current_frame.shape
        source_label_map = {
            "image": "Upload Image",
            "video": "Upload Video",
            "camera": "Live Camera"
        }
        source_display = source_label_map.get(st.session_state.get("input_source"), "External Intake")

        st.markdown(f"""
        <div class="content-card" style="padding: 24px;">
            <div style="display: flex; gap: 24px; flex-wrap: wrap; margin-bottom: 16px;">
                <div><strong>Source:</strong> <span style="color: var(--accent-blue);">{source_display}</span></div>
                <div><strong>Resolution:</strong> {w} × {h} pixels</div>
                <div><strong>Status:</strong> <span style="color: var(--status-safe);">Ready for analysis</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        rgb_preview = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
        st.image(rgb_preview, caption=f"Active Frame Preview ({w}×{h})", width="stretch")
    else:
        st.info("No valid intake frame selected yet. Upload an image, select a video frame, or capture from live camera above.")

    st.markdown("---")

    # 4. Start AI Analysis Section
    st.markdown("### 🚀 AI Decision Pipeline")
    
    frame_ready = input_handler.is_valid_frame(current_frame)
    
    if st.button("🚀 START AI ANALYSIS", type="primary", disabled=not frame_ready, use_container_width=True, key="btn_start_ai_analysis"):
        with st.spinner("Analyzing scene hazards & evaluating open landing candidates..."):
            analysis_res = scene_analysis.analyze_scene(current_frame)
            st.session_state["scene_analysis_result"] = analysis_res
            st.toast("Scene Understanding analysis complete!")

    # Display Main Analysis Dashboard if analysis results exist
    if "scene_analysis_result" in st.session_state and st.session_state["scene_analysis_result"]:
        res = st.session_state["scene_analysis_result"]
        hazards = res.get("hazards", [])
        candidates = res.get("candidate_zones", [])

        # Evaluate clearances & safety scores
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

        # Safe Landing Threshold & Emergency Mode Evaluation Logic
        decision_res = landing_engine.evaluate_landing_decision(
            ranked_zones=ranked_zones,
            safe_threshold=landing_engine.SAFE_THRESHOLD,
            emergency_mode=emergency_mode
        )
        st.session_state["landing_decision"] = decision_res

        recommended_zone = decision_res.get("zone") if decision_res.get("decision") == "RECOMMEND" else None
        rec_label = decision_res.get("recommended_zone")

        st.markdown("---")
        st.markdown("### 🎯 MAIN LANDING ANALYSIS DASHBOARD")

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
                    highest_name = decision_res.get("highest_candidate") or "None"
                    highest_score = decision_res.get("highest_score", 0)

                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #7F1D1D 0%, #991B1B 100%); border: 1px solid #EF4444; border-radius: 14px; padding: 20px; margin-bottom: 16px; box-shadow: 0 4px 12px rgba(239,68,68,0.2);">
                        <div style="font-size: 0.9rem; font-weight: 800; color: #FCA5A5; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px;">
                            🔴 NO SAFE LANDING ZONE
                        </div>
                        <div style="font-size: 1.05rem; font-weight: 700; color: #FFFFFF; margin-bottom: 4px;">
                            Highest Candidate: <strong>{highest_name}</strong>
                        </div>
                        <div style="font-size: 1.25rem; font-weight: 800; color: #F8FAFC; margin-bottom: 10px;">
                            Score: {highest_score} <span style="font-size: 0.85rem; color: #FCA5A5;">/ 100</span>
                        </div>
                        <div style="font-size: 0.85rem; color: #FEE2E2; margin-bottom: 8px;">
                            Safety threshold not satisfied.
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
