"""Input Handler Module

Provides reusable helpers for image conversion, video frame extraction,
webcam frame capture, and frame validation for SafeLand AI.
"""

from typing import Optional, Tuple, Any, Dict
import numpy as np
import cv2
import tempfile
import os
import streamlit as st


def is_valid_frame(frame: Any) -> bool:
    """Validate if an object is a non-empty NumPy ndarray frame."""
    return isinstance(frame, np.ndarray) and frame.size > 0


def get_current_frame() -> Optional[np.ndarray]:
    """Return st.session_state["current_frame"] if it is a valid numpy ndarray, otherwise return None."""
    frame = st.session_state.get("current_frame", None)
    if is_valid_frame(frame):
        return frame
    return None


def process_uploaded_image(uploaded_file: Any) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Convert an uploaded image file (JPG/PNG) into a valid BGR NumPy ndarray.

    Args:
        uploaded_file: Streamlit UploadedFile object.

    Returns:
        Tuple of (bgr_frame, error_message).
    """
    if uploaded_file is None:
        return None, "No file provided."

    try:
        if hasattr(uploaded_file, "getvalue"):
            raw_bytes = uploaded_file.getvalue()
        else:
            if hasattr(uploaded_file, "seek"):
                uploaded_file.seek(0)
            raw_bytes = uploaded_file.read()

        file_bytes = np.frombuffer(raw_bytes, np.uint8)
        if file_bytes.size == 0:
            return None, "Uploaded image file is empty."
        
        bgr_frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if not is_valid_frame(bgr_frame):
            return None, "Failed to decode image. Please upload a valid JPG, JPEG, or PNG."
        
        return bgr_frame, None
    except Exception as e:
        return None, f"Image decoding error: {str(e)}"


def get_video_metadata(uploaded_file: Any) -> Tuple[int, float, Optional[str], Optional[str]]:
    """Save uploaded video file temporarily and extract total frame count and FPS.

    Args:
        uploaded_file: Streamlit UploadedFile object.

    Returns:
        Tuple of (total_frames, fps, temp_file_path, error_message).
    """
    if uploaded_file is None:
        return 0, 0.0, None, "No video file uploaded."

    try:
        # Create temp file
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"safeland_temp_{uploaded_file.file_id if hasattr(uploaded_file, 'file_id') else 'video'}.mp4")
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            cap.release()
            return 0, 0.0, None, "Unable to open MP4 video file."

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        if total_frames <= 0:
            return 0, 0.0, None, "Uploaded video contains no frames."

        return total_frames, fps if fps > 0 else 30.0, temp_path, None
    except Exception as e:
        return 0, 0.0, None, f"Video reading error: {str(e)}"


def extract_video_frame(temp_path: str, frame_index: int) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Extract a specific frame from a video file path.

    Args:
        temp_path: Path to temporary video file.
        frame_index: Target frame index (0 to total_frames - 1).

    Returns:
        Tuple of (bgr_frame, error_message).
    """
    if not temp_path or not os.path.exists(temp_path):
        return None, "Video temporary file not found."

    cap = cv2.VideoCapture(temp_path)
    if not cap.isOpened():
        cap.release()
        return None, "Failed to open video capture."

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()

        if not ret or not is_valid_frame(frame):
            return None, f"Could not read frame at index {frame_index}."

        return frame, None
    except Exception as e:
        cap.release()
        return None, f"Frame extraction error: {str(e)}"


def capture_webcam_frame(camera_index: int = 0) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Open webcam, capture one frame, release immediately, and return BGR frame.

    Args:
        camera_index: Device index for VideoCapture (default: 0).

    Returns:
        Tuple of (bgr_frame, error_message).
    """
    try:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            cap.release()
            return None, "Camera unavailable. Please check camera permissions or use Image/Video input."

        ret, frame = cap.read()
        cap.release()

        if not ret or not is_valid_frame(frame):
            return None, "Failed to capture frame from live camera feed."

        return frame, None
    except Exception as e:
        return None, f"Webcam error: {str(e)}"


def process_camera_input(camera_file: Any) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Convert Streamlit camera_input object bytes into a valid BGR NumPy ndarray frame.

    Args:
        camera_file: Streamlit camera_input object.

    Returns:
        Tuple of (bgr_frame, error_message).
    """
    if camera_file is None:
        return None, "No camera input provided."

    try:
        file_bytes = np.asarray(bytearray(camera_file.getvalue()), dtype=np.uint8)
        bgr_frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if not is_valid_frame(bgr_frame):
            return None, "Unable to process the captured camera frame."

        return bgr_frame, None
    except Exception as e:
        return None, f"Camera frame processing error: {str(e)}"

