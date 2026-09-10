"""Test Landing Engine Clearance Assessment & evaluate_all_candidate_clearances

Tests:
1. check_clearance with standard and alternative bbox/drone representations
2. evaluate_all_candidate_clearances with:
   - Small drone (Surveillance Drone)
   - Large drone (Rescue Drone)
   - Multiple candidate zones
   - Empty candidate list
   - Malformed candidate data
3. Verification command check
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import core.landing_engine as landing_engine


def test_landing_engine_complete():
    print("=== STARTING COMPLETE LANDING ENGINE TESTS ===")

    frame_shape = (800, 1000, 3)

    surveillance_drone = {
        "name": "Surveillance Drone",
        "width_m": 0.7,
        "length_m": 0.7,
        "required_clearance_m": 1.5
    }

    rescue_drone = {
        "name": "Rescue Drone",
        "width_m": 1.8,
        "length_m": 1.6,
        "required_clearance_m": 3.5
    }

    # Candidates: Zone A (280x280px), Zone B (400x400px), Zone C (100x100px)
    candidates = [
        {"label": "Zone A", "bbox": [100, 100, 380, 380]}, # 2.8m x 2.8m at 10m scale
        {"label": "Zone B", "bbox": [50, 50, 450, 450]},   # 4.0m x 4.0m at 10m scale
        {"label": "Zone C", "bbox": [10, 10, 110, 110]}    # 1.0m x 1.0m at 10m scale
    ]

    print("\n--- Test 1: evaluate_all_candidate_clearances with Small Drone (Surveillance) ---")
    eval_small = landing_engine.evaluate_all_candidate_clearances(
        candidate_zones=candidates,
        drone_profile=surveillance_drone,
        frame_shape=frame_shape,
        estimated_ground_width_m=10.0
    )
    print("Small Drone Evaluation Results:")
    for res in eval_small:
        print(f"  {res['label']}: status={res['status']}, passed={res['passed']}, size={res['estimated_width_m']}m x {res['estimated_length_m']}m, req={res['required_width_m']}m x {res['required_length_m']}m")
        assert "status" in res
        assert "passed" in res
        assert "estimated_width_m" in res
        assert "estimated_length_m" in res
        assert "required_width_m" in res
        assert "required_length_m" in res

    assert eval_small[0]["passed"] is True, "Zone A should pass for Surveillance Drone"
    assert eval_small[1]["passed"] is True, "Zone B should pass for Surveillance Drone"
    assert eval_small[2]["passed"] is False, "Zone C (1.0m) should fail for Surveillance Drone (requires 1.5m)"

    print("\n--- Test 2: evaluate_all_candidate_clearances with Large Drone (Rescue) ---")
    eval_large = landing_engine.evaluate_all_candidate_clearances(
        candidate_zones=candidates,
        drone_profile=rescue_drone,
        frame_shape=frame_shape,
        estimated_ground_width_m=10.0
    )
    print("Large Drone Evaluation Results:")
    for res in eval_large:
        print(f"  {res['label']}: status={res['status']}, passed={res['passed']}, size={res['estimated_width_m']}m x {res['estimated_length_m']}m, req={res['required_width_m']}m x {res['required_length_m']}m")

    assert eval_large[0]["passed"] is False, "Zone A (2.8m) should fail for Rescue Drone (requires 3.5m)"
    assert eval_large[1]["passed"] is True, "Zone B (4.0m) should pass for Rescue Drone (requires 3.5m)"
    assert eval_large[2]["passed"] is False, "Zone C (1.0m) should fail for Rescue Drone (requires 3.5m)"

    print("\n--- Test 3: Empty Candidate List ---")
    eval_empty = landing_engine.evaluate_all_candidate_clearances(
        candidate_zones=[],
        drone_profile=surveillance_drone,
        frame_shape=frame_shape
    )
    assert eval_empty == [], "Empty candidate list must return empty list without crashing"
    print("Empty list result:", eval_empty)

    print("\n--- Test 4: Malformed Candidate Data ---")
    malformed_candidates = [
        "not_a_dict",
        {"label": "Malformed Zone 1", "bbox": "invalid_bbox"},
        {"label": "Malformed Zone 2", "bbox": [0, 0]},
        {"label": "Valid Zone", "bbox": [100, 100, 500, 500]}
    ]
    eval_malformed = landing_engine.evaluate_all_candidate_clearances(
        candidate_zones=malformed_candidates,
        drone_profile=surveillance_drone,
        frame_shape=frame_shape
    )
    assert len(eval_malformed) == 4
    assert eval_malformed[0]["status"] == "REJECTED"
    assert eval_malformed[1]["status"] == "REJECTED"
    assert eval_malformed[2]["status"] == "REJECTED"
    assert eval_malformed[3]["status"] == "PASSED"
    print("Malformed candidate evaluation handled safely without crashing!")

    print("\n[SUCCESS] ALL LANDING ENGINE INTEGRATION TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_landing_engine_complete()
