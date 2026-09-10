"""Test Default Drone Profile Removal and Selection Behavior

Validates requirements A through G:
A. Rescue Drone = default.
B. Click Remove Default.
C. Confirm no default exists.
D. Set Delivery Drone as default.
E. Confirm only Delivery Drone becomes default.
F. Remove Delivery Drone as default.
G. Confirm all drones return to "Set as Default".
"""

import sys
from pathlib import Path
import os
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import core.drone_profiles as drone_profiles

def run_test():
    print("=== TESTING DEFAULT DRONE BEHAVIOR (STEPS A to G) ===")

    # Step A: Rescue Drone = default
    drone_profiles.set_default_drone_name("Rescue Drone")
    default_name = drone_profiles.get_default_drone_name()
    assert default_name == "Rescue Drone", f"Step A Failed: Expected Rescue Drone, got {default_name}"
    rescue_drone = drone_profiles.get_drone_by_name(default_name)
    assert rescue_drone is not None and rescue_drone["name"] == "Rescue Drone"
    print("[PASS] Step A: Rescue Drone set as default")

    # Step B: Remove Default
    removed = drone_profiles.remove_default_drone_name()
    assert removed is True, "Step B Failed: remove_default_drone_name returned False"
    print("[PASS] Step B: Remove Default called successfully")

    # Step C: Confirm no default exists
    default_name_after = drone_profiles.get_default_drone_name()
    assert default_name_after is None, f"Step C Failed: Expected None, got {default_name_after}"
    drone_by_none = drone_profiles.get_drone_by_name(default_name_after)
    assert drone_by_none is None, f"Step C Failed: get_drone_by_name(None) should return None, got {drone_by_none}"

    drones = drone_profiles.load_drone_profiles()
    for d in drones:
        is_default = (default_name_after is not None and d["name"] == default_name_after)
        assert not is_default, f"Step C Failed: Drone {d['name']} still marked as default"
    print("[PASS] Step C: Confirmed no default drone exists")

    # Step D: Set Delivery Drone as default
    set_deliv = drone_profiles.set_default_drone_name("Delivery Drone")
    assert set_deliv is True
    print("[PASS] Step D: Set Delivery Drone as default")

    # Step E: Confirm only Delivery Drone becomes default
    deliv_default = drone_profiles.get_default_drone_name()
    assert deliv_default == "Delivery Drone", f"Step E Failed: Expected Delivery Drone, got {deliv_default}"

    for d in drones:
        is_default = (deliv_default is not None and d["name"] == deliv_default)
        if d["name"] == "Delivery Drone":
            assert is_default is True, "Step E Failed: Delivery Drone should be default"
        else:
            assert is_default is False, f"Step E Failed: Drone {d['name']} should NOT be default"
    print("[PASS] Step E: Confirmed ONLY Delivery Drone is default")

    # Step F: Remove Delivery Drone as default
    removed_f = drone_profiles.remove_default_drone_name()
    assert removed_f is True
    print("[PASS] Step F: Remove Delivery Drone as default called")

    # Step G: Confirm all drones return to "Set as Default" (no default drone exists)
    final_default = drone_profiles.get_default_drone_name()
    assert final_default is None, f"Step G Failed: Expected None, got {final_default}"
    for d in drones:
        is_default = (final_default is not None and d["name"] == final_default)
        assert is_default is False, f"Step G Failed: Drone {d['name']} is still marked as default"
    print("[PASS] Step G: Confirmed all drones return to 'Set as Default'")

    # Reset default back to Rescue Drone for demo baseline
    drone_profiles.set_default_drone_name("Rescue Drone")
    print("\nALL TESTS (A THROUGH G) PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_test()
