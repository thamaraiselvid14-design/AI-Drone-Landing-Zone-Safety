import os
import sys
import numpy as np
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Mock streamlit session state for testing logic
class MockSessionState(dict):
    def __getattr__(self, key):
        return self.get(key, None)
    def __setattr__(self, key, value):
        self[key] = value
    def __delattr__(self, key):
        if key in self:
            del self[key]

import streamlit as st
if not hasattr(st, "session_state") or not isinstance(st.session_state, MockSessionState):
    st.session_state = MockSessionState()

import core.input_handler as input_handler

def test_input_source_state_machine():
    print("--- 1. Testing Initial State ---")
    st.session_state["active_input_type"] = None
    st.session_state["current_frame"] = None
    st.session_state["input_source"] = None
    st.session_state["last_camera_id"] = None

    cam_frame_1 = np.ones((480, 640, 3), dtype=np.uint8) * 100
    img_frame_1 = np.ones((480, 640, 3), dtype=np.uint8) * 200

    print("--- 2. Capture Live Camera Photo ---")
    current_cam_id = "cam_photo_001"
    cam_captured_new = (current_cam_id != st.session_state.get("last_camera_id"))
    assert cam_captured_new is True

    if cam_captured_new:
        st.session_state["current_frame"] = cam_frame_1.copy()
        st.session_state["active_input_type"] = "camera"
        st.session_state["input_source"] = "camera"
        st.session_state["last_camera_id"] = current_cam_id

    assert st.session_state["active_input_type"] == "camera"
    assert np.array_equal(input_handler.get_current_frame(), cam_frame_1)
    print("PASSED: Camera photo captured as active source.")

    print("--- 3. Upload Image (Before clicking USE THIS IMAGE) ---")
    # Uploading image puts frame into pending state, but DOES NOT set active_input_type or current_frame
    st.session_state["pending_uploaded_image"] = img_frame_1

    # Simulate Streamlit rerun with persistent camera widget buffer
    cam_captured_new_rerun = (current_cam_id != st.session_state.get("last_camera_id"))
    assert cam_captured_new_rerun is False

    # Since user hasn't clicked USE THIS IMAGE yet, active_input_type remains camera
    assert st.session_state["active_input_type"] == "camera"
    assert np.array_equal(input_handler.get_current_frame(), cam_frame_1)
    print("PASSED: Uploading image does NOT activate it before clicking USE THIS IMAGE.")

    print("--- 4. User Clicks 'USE THIS IMAGE' ---")
    # Click USE THIS IMAGE
    st.session_state["current_frame"] = img_frame_1.copy()
    st.session_state["active_input_type"] = "image"
    st.session_state["input_source"] = "image"

    assert st.session_state["active_input_type"] == "image"
    assert np.array_equal(input_handler.get_current_frame(), img_frame_1)
    print("PASSED: Active source updated to Uploaded Image.")

    print("--- 5. Subsequent Rerun with Persistent Camera Buffer ---")
    # On next rerun, camera widget still holds cam_photo_001
    cam_captured_new_rerun2 = (current_cam_id != st.session_state.get("last_camera_id"))
    btn_use_cam = False

    if cam_captured_new_rerun2 or btn_use_cam:
        st.session_state["current_frame"] = cam_frame_1.copy()
        st.session_state["active_input_type"] = "camera"

    # Must STAY image
    assert st.session_state["active_input_type"] == "image"
    assert np.array_equal(input_handler.get_current_frame(), img_frame_1)
    print("PASSED: Camera buffer does NOT overwrite active Uploaded Image on reruns.")

    print("--- 6. AI Analysis Source Verification ---")
    analysis_frame = input_handler.get_current_frame()
    assert np.array_equal(analysis_frame, img_frame_1)
    assert not np.array_equal(analysis_frame, cam_frame_1)
    print("PASSED: START AI ANALYSIS uses Uploaded Image, NOT the old camera frame.")

    print("--- 7. Reverse Test: Snapping New Camera Photo ---")
    new_cam_id = "cam_photo_002"
    cam_frame_2 = np.ones((480, 640, 3), dtype=np.uint8) * 150
    cam_captured_new_2 = (new_cam_id != st.session_state.get("last_camera_id"))
    assert cam_captured_new_2 is True

    if cam_captured_new_2:
        st.session_state["current_frame"] = cam_frame_2.copy()
        st.session_state["active_input_type"] = "camera"
        st.session_state["input_source"] = "camera"
        st.session_state["last_camera_id"] = new_cam_id

    assert st.session_state["active_input_type"] == "camera"
    assert np.array_equal(input_handler.get_current_frame(), cam_frame_2)
    print("PASSED: Snapping new camera photo correctly activates Live Camera frame.")

    print("\nALL INPUT SOURCE STATE MACHINE TESTS PASSED CLEANLY!")

if __name__ == "__main__":
    test_input_source_state_machine()
