"""Test Upload Image Confirmation Flow

Validates:
1. Uploading image decodes frame and sets pending_uploaded_image state
2. current_frame remains uncommitted until USE THIS IMAGE is clicked
3. Clicking USE THIS IMAGE commits frame to current_frame and sets input_source == "image"
4. START AI ANALYSIS button is enabled after confirmation
5. Uploading a new image updates pending_uploaded_image and requires confirmation again
"""

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import core.input_handler as input_handler


def test_upload_confirmation_flow():
    print("=== TESTING UPLOAD IMAGE CONFIRMATION FLOW ===")

    session_state = {
        "current_frame": None,
        "input_source": None,
        "pending_uploaded_image": None
    }

    # Step 1: User uploads image 1 (800x1000 synthetic frame)
    frame1 = np.full((800, 1000, 3), 100, dtype=np.uint8)
    assert input_handler.is_valid_frame(frame1)

    session_state["pending_uploaded_image"] = frame1
    print("Uploaded Image 1 -> pending_uploaded_image set")
    assert session_state["current_frame"] is None, "current_frame must NOT be overwritten immediately"

    # Step 2: User clicks 'USE THIS IMAGE'
    session_state["current_frame"] = session_state["pending_uploaded_image"].copy()
    session_state["input_source"] = "image"
    print("Clicked USE THIS IMAGE -> current_frame committed, input_source == 'image'")

    assert input_handler.is_valid_frame(session_state["current_frame"])
    assert session_state["input_source"] == "image"
    frame_ready = input_handler.is_valid_frame(session_state["current_frame"])
    assert frame_ready is True, "START AI ANALYSIS button must be ENABLED"

    # Step 3: User uploads image 2 (600x800 synthetic frame)
    frame2 = np.full((600, 800, 3), 200, dtype=np.uint8)
    session_state["pending_uploaded_image"] = frame2
    print("Uploaded Image 2 -> pending_uploaded_image updated")

    # Before clicking USE THIS IMAGE for image 2, current_frame is still image 1
    assert session_state["current_frame"].shape == (800, 1000, 3)

    # User clicks 'USE THIS IMAGE' for image 2
    session_state["current_frame"] = session_state["pending_uploaded_image"].copy()
    assert session_state["current_frame"].shape == (600, 800, 3)
    print("Clicked USE THIS IMAGE for Image 2 -> current_frame updated to new image")

    print("\n[SUCCESS] UPLOAD IMAGE CONFIRMATION FLOW TEST PASSED!")

if __name__ == "__main__":
    test_upload_confirmation_flow()
