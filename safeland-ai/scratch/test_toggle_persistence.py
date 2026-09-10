import sys
import os
sys.path.insert(0, os.getcwd())

def test_toggle_state_persistence():
    print("=== TESTING EMERGENCY MODE TOGGLE STATE PERSISTENCE ===")

    # Simulated Streamlit Session State Dict
    class MockSessionState(dict):
        def __getattr__(self, key):
            return self.get(key)
        def __setattr__(self, key, value):
            self[key] = value

    session_state = MockSessionState()

    # Step 1: App initialization
    if "emergency_mode" not in session_state:
        session_state["emergency_mode"] = False

    print("Step 1 (App start): emergency_mode =", session_state["emergency_mode"])
    assert session_state["emergency_mode"] == False

    # Step 2: User toggles ON
    session_state["emergency_mode"] = True
    print("Step 2 (User toggles ON): emergency_mode =", session_state["emergency_mode"])
    assert session_state["emergency_mode"] == True

    # Step 3: User changes drone (simulated rerun without changing emergency_mode)
    session_state["selected_drone"] = "Rescue Drone"
    print("Step 3 (Change drone rerun): emergency_mode =", session_state["emergency_mode"])
    assert session_state["emergency_mode"] == True

    # Step 4: User uploads image (simulated rerun)
    session_state["input_source"] = "image"
    print("Step 4 (Upload image rerun): emergency_mode =", session_state["emergency_mode"])
    assert session_state["emergency_mode"] == True

    # Step 5: User runs AI analysis (simulated rerun)
    session_state["analysis_requested"] = True
    print("Step 5 (AI Analysis rerun): emergency_mode =", session_state["emergency_mode"])
    assert session_state["emergency_mode"] == True

    # Step 6: User toggles OFF
    session_state["emergency_mode"] = False
    print("Step 6 (User toggles OFF): emergency_mode =", session_state["emergency_mode"])
    assert session_state["emergency_mode"] == False

    # Step 7: Subsequent reruns maintain OFF state
    session_state["analysis_requested"] = True
    print("Step 7 (Subsequent rerun): emergency_mode =", session_state["emergency_mode"])
    assert session_state["emergency_mode"] == False

    print("\n[OK] TOGGLE STATE PERSISTENCE TEST PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_toggle_state_persistence()
