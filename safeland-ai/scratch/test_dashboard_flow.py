"""Test Main Analysis Dashboard Workflows

Validates:
1. Automatic Detection Mode workflow & Recommendation Rule (SAFE + Passed clearance required)
2. No Safe Landing Zone fallback condition
3. Select Area Mode workflow & Selected Area Safety Rule
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
import core.scoring as scoring


def test_dashboard_workflows():
    print("=== TESTING MAIN ANALYSIS DASHBOARD WORKFLOWS ===")

    frame = np.full((800, 1000, 3), 120, dtype=np.uint8)

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

    # 1. Automatic Detection Workflow with Surveillance Drone (Safe Zone Expected)
    print("\n--- 1. Automatic Detection (Surveillance Drone) ---")
    candidates = [
        {"label": "Zone A", "bbox": [100, 100, 380, 380]}, # 2.8m x 2.8m -> SAFE (Score 100)
        {"label": "Zone B", "bbox": [500, 100, 650, 250]}   # 1.5m x 1.5m
    ]
    clrs_surv = landing_engine.evaluate_all_candidate_clearances(candidates, surveillance_drone, frame.shape, 10.0)
    ranked_surv = scoring.rank_zones(candidates, surveillance_drone, [], clrs_surv, frame.shape)

    rec_zone_surv = None
    for rz in ranked_surv:
        clr = rz.get("clearance", {})
        if clr.get("passed") and rz.get("score", 0) >= 80:
            rec_zone_surv = rz
            break

    print(f"Surveillance Drone Recommended Zone: {rec_zone_surv['label'] if rec_zone_surv else 'None'} (Score: {rec_zone_surv['score'] if rec_zone_surv else 0})")
    assert rec_zone_surv is not None, "Zone A should be recommended for Surveillance Drone"
    assert rec_zone_surv["label"] == "Zone A"
    assert rec_zone_surv["score"] >= 80

    # 2. No Safe Landing Zone Condition (Rescue Drone on same frame where Zone A is 2.8m < 3.5m required)
    print("\n--- 2. No Safe Landing Zone Test (Rescue Drone) ---")
    clrs_rescue = landing_engine.evaluate_all_candidate_clearances(candidates, rescue_drone, frame.shape, 10.0)
    ranked_rescue = scoring.rank_zones(candidates, rescue_drone, [], clrs_rescue, frame.shape)

    rec_zone_rescue = None
    for rz in ranked_rescue:
        clr = rz.get("clearance", {})
        if clr.get("passed") and rz.get("score", 0) >= 80:
            rec_zone_rescue = rz
            break

    print(f"Rescue Drone Recommended Zone: {rec_zone_rescue['label'] if rec_zone_rescue else 'None'}")
    assert rec_zone_rescue is None, "Rescue Drone should have NO recommended zone because clearance failed"
    assert ranked_rescue[0]["score"] <= 49, "Highest candidate score for Rescue Drone must be <= 49 (UNSAFE)"

    # 3. Select Area Workflow (Manual Selection)
    print("\n--- 3. Select Area Workflow ---")
    # Select large open region [200, 200, 700, 700] (500x500px -> 5.0m x 5.0m at 10m scale)
    selected_cand = {
        "label": "Selected Area",
        "bbox": [200, 200, 700, 700]
    }
    sel_clr = landing_engine.check_clearance(selected_cand, rescue_drone, frame.shape, 10.0)
    sel_score = scoring.score_zone(selected_cand, rescue_drone, [], sel_clr, frame.shape)

    print(f"Selected Area Result (Rescue Drone): Clearance Passed={sel_clr['passed']}, Score={sel_score['score']}, Status={sel_score['status']}")
    assert sel_clr["passed"] is True, "Large 5.0m selected area should pass clearance for Rescue Drone"
    assert sel_score["score"] >= 80, "Large open selected area with no hazards should score >= 80"
    assert sel_score["status"] == "SAFE"

    print("\n[SUCCESS] ALL MAIN ANALYSIS DASHBOARD WORKFLOW TESTS PASSED!")

if __name__ == "__main__":
    test_dashboard_workflows()
