"""Test Missions Page State Retrieval Logic

Validates:
A. Open Missions directly after app startup.
B. Default drone exists.
C. No default drone exists.
D. User selected a drone on Dashboard.
E. Default drone was removed.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import core.drone_profiles as drone_profiles


def get_missions_drone_name(session_state: dict) -> str:
    """Replicates the safe drone retrieval logic on the Missions page."""
    selected_drone_obj = session_state.get("selected_drone")
    if selected_drone_obj is None:
        default_name = drone_profiles.get_default_drone_name()
        if default_name:
            selected_drone_obj = drone_profiles.get_drone_by_name(default_name)

    if isinstance(selected_drone_obj, dict):
        return selected_drone_obj.get("name", "No active drone selected.")
    else:
        return "No active drone selected."


def run_tests():
    print("=== TESTING MISSIONS PAGE DRONE RETRIEVAL LOGIC ===")

    # Scenario B: Default drone exists (Rescue Drone)
    drone_profiles.set_default_drone_name("Rescue Drone")
    st_state_b = {}
    name_b = get_missions_drone_name(st_state_b)
    assert name_b == "Rescue Drone", f"Scenario B Failed: Expected 'Rescue Drone', got '{name_b}'"
    print("[PASS] Scenario B: Default drone exists -> 'Rescue Drone'")

    # Scenario C: No default drone exists
    drone_profiles.remove_default_drone_name()
    st_state_c = {}
    name_c = get_missions_drone_name(st_state_c)
    assert name_c == "No active drone selected.", f"Scenario C Failed: Expected 'No active drone selected.', got '{name_c}'"
    print("[PASS] Scenario C: No default drone exists -> 'No active drone selected.'")

    # Scenario D: User selected a drone on Dashboard (e.g. Surveillance Drone)
    st_state_d = {
        "selected_drone": drone_profiles.get_drone_by_name("Surveillance Drone"),
        "selected_drone_name": "Surveillance Drone"
    }
    name_d = get_missions_drone_name(st_state_d)
    assert name_d == "Surveillance Drone", f"Scenario D Failed: Expected 'Surveillance Drone', got '{name_d}'"
    print("[PASS] Scenario D: User selected drone on Dashboard -> 'Surveillance Drone'")

    # Scenario E: Default drone was removed
    drone_profiles.remove_default_drone_name()
    st_state_e = {}
    name_e = get_missions_drone_name(st_state_e)
    assert name_e == "No active drone selected.", f"Scenario E Failed: Expected 'No active drone selected.', got '{name_e}'"
    print("[PASS] Scenario E: Default drone removed -> 'No active drone selected.'")

    # Scenario A: Open Missions directly after app startup with default present
    drone_profiles.set_default_drone_name("Rescue Drone")
    st_state_a = {}
    name_a = get_missions_drone_name(st_state_a)
    assert name_a == "Rescue Drone", f"Scenario A Failed: Expected 'Rescue Drone', got '{name_a}'"
    print("[PASS] Scenario A: Open Missions directly after app startup -> 'Rescue Drone'")

    print("\nALL MISSIONS PAGE DRONE RETRIEVAL TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_tests()
