"""Test Streamlit Clearance Flow Integration

Validates:
1. Candidate zone generation via scene_analysis
2. Clearance assessment via landing_engine for Surveillance Drone vs Rescue Drone
3. Ground scale slider adjustments
4. Edge cases
"""

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import core.drone_profiles as drone_profiles
import core.scene_analysis as scene_analysis
import core.landing_engine as landing_engine


def test_clearance_integration():
    print("=== TESTING INTEGRATION CLEARANCE FLOW ===")

    # 1. Create a dummy synthetic frame (800x1000 pixels BGR)
    frame = np.full((800, 1000, 3), 120, dtype=np.uint8)

    # 2. Analyze scene
    res = scene_analysis.analyze_scene(frame)
    candidates = res.get("candidate_zones", [])
    print(f"Candidates found: {len(candidates)}")
    assert len(candidates) > 0, "Scene analysis should propose candidate zones for synthetic frame"

    # 3. Load Drones
    drones = drone_profiles.load_drone_profiles()
    surveillance_drone = drone_profiles.get_drone_by_name("Surveillance Drone")
    rescue_drone = drone_profiles.get_drone_by_name("Rescue Drone")

    print(f"Surveillance Drone Specs: {surveillance_drone}")
    print(f"Rescue Drone Specs: {rescue_drone}")

    # 4. Evaluate clearances at default 10.0m scale
    print("\n--- Evaluating at 10.0m Ground Scale ---")
    surv_eval = landing_engine.evaluate_all_candidate_clearances(
        candidate_zones=candidates,
        drone_profile=surveillance_drone,
        frame_shape=frame.shape,
        estimated_ground_width_m=10.0
    )

    rescue_eval = landing_engine.evaluate_all_candidate_clearances(
        candidate_zones=candidates,
        drone_profile=rescue_drone,
        frame_shape=frame.shape,
        estimated_ground_width_m=10.0
    )

    for i, (s_zone, r_zone) in enumerate(zip(surv_eval, rescue_eval)):
        label = s_zone["label"]
        s_clr = s_zone["clearance"]
        r_clr = r_zone["clearance"]

        print(f"\n{label}:")
        print(f"  Surveillance Drone ({s_clr['required_width_m']}m x {s_clr['required_length_m']}m): Estimated {s_clr['estimated_zone_width_m']}m x {s_clr['estimated_zone_height_m']}m -> PASSED={s_clr['passed']}")
        print(f"  Rescue Drone ({r_clr['required_width_m']}m x {r_clr['required_length_m']}m): Estimated {r_clr['estimated_zone_width_m']}m x {r_clr['estimated_zone_height_m']}m -> PASSED={r_clr['passed']}")

    # Verification: Zone candidate dimensions (280px out of 1000px = 2.8m at 10m scale)
    # Surveillance Drone requires 1.5m -> 2.8m >= 1.5m -> PASSED
    # Rescue Drone requires 3.5m -> 2.8m < 3.5m -> REJECTED
    assert surv_eval[0]["clearance"]["passed"] is True, "Zone A should PASS for Surveillance Drone at 10m scale"
    assert rescue_eval[0]["clearance"]["passed"] is False, "Zone A should REJECT for Rescue Drone at 10m scale"

    # 5. Evaluate at 20.0m Ground Scale
    print("\n--- Evaluating at 20.0m Ground Scale ---")
    rescue_eval_20m = landing_engine.evaluate_all_candidate_clearances(
        candidate_zones=candidates,
        drone_profile=rescue_drone,
        frame_shape=frame.shape,
        estimated_ground_width_m=20.0
    )
    # At 20m scale, 280px = 5.6m >= 3.5m -> PASSED
    print(f"Zone A at 20m scale for Rescue Drone: Estimated {rescue_eval_20m[0]['clearance']['estimated_zone_width_m']}m -> PASSED={rescue_eval_20m[0]['clearance']['passed']}")
    assert rescue_eval_20m[0]["clearance"]["passed"] is True, "Zone A should PASS for Rescue Drone at 20m scale"

    print("\n[SUCCESS] INTEGRATION CLEARANCE FLOW COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    test_clearance_integration()
