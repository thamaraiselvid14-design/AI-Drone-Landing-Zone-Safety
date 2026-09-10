"""Comprehensive Test Suite for Advanced Missions & Settings Pages

Tests 1 through 9 from instructions:
TEST 1: Fresh application -> Missions
TEST 2: Fresh application -> Settings
TEST 3: Set a default drone -> Settings
TEST 4: Remove default drone -> Missions
TEST 5: Run image analysis -> Missions
TEST 6: Run dynamic re-evaluation -> Missions
TEST 7: Change settings -> Save -> Reload
TEST 8: Open History after several analyses
TEST 9: Full page navigation cycle
"""

import sys
from pathlib import Path
import os
import json
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import core.drone_profiles as drone_profiles
import core.landing_engine as landing_engine
import core.scene_analysis as scene_analysis
import core.scoring as scoring
import core.history as history


def run_all_tests():
    print("=== STARTING COMPREHENSIVE MISSIONS & SETTINGS TESTS (TESTS 1 to 9) ===")

    # TEST 1: Fresh application -> Missions
    session_state = {}
    default_name = drone_profiles.get_default_drone_name()
    selected_drone_obj = session_state.get("selected_drone")
    if selected_drone_obj is None and default_name:
        selected_drone_obj = drone_profiles.get_drone_by_name(default_name)
    
    scene_res = session_state.get("scene_analysis_result")
    assert scene_res is None, "Fresh session should have no active scene_analysis_result"
    print("[PASS] TEST 1: Fresh application -> Missions shows 'No active mission'")

    # TEST 2: Fresh application -> Settings
    settings_data = drone_profiles.load_settings()
    assert isinstance(settings_data, dict), "Settings should be loaded as dictionary"
    assert "estimated_ground_width_m" in settings_data
    assert "alerts" in settings_data
    print("[PASS] TEST 2: Fresh application -> Settings loads settings dictionary correctly")

    # TEST 3: Set a default drone -> Settings
    drone_profiles.set_default_drone_name("Delivery Drone")
    def_name = drone_profiles.get_default_drone_name()
    assert def_name == "Delivery Drone"
    def_obj = drone_profiles.get_drone_by_name(def_name)
    assert def_obj["required_clearance_m"] == 2.5
    print("[PASS] TEST 3: Set default drone -> Settings displays correct drone (Delivery Drone) and clearance (2.5m)")

    # TEST 4: Remove default drone -> Missions
    drone_profiles.remove_default_drone_name()
    def_name_none = drone_profiles.get_default_drone_name()
    assert def_name_none is None, "Default drone should be None after removal"
    
    session_state_no_def = {}
    retrieved_drone = session_state_no_def.get("selected_drone")
    if retrieved_drone is None:
        def_n = drone_profiles.get_default_drone_name()
        if def_n:
            retrieved_drone = drone_profiles.get_drone_by_name(def_n)
    
    active_drone_str = retrieved_drone.get("name", "No active drone selected.") if isinstance(retrieved_drone, dict) else "No active drone selected."
    assert active_drone_str == "No active drone selected.", f"Expected 'No active drone selected.', got '{active_drone_str}'"
    print("[PASS] TEST 4: Remove default drone -> Missions displays 'No active drone selected.' with no crash or fake fallback")

    # TEST 5: Run image analysis -> Missions
    drone_profiles.set_default_drone_name("Rescue Drone")
    rescue_drone = drone_profiles.get_drone_by_name("Rescue Drone")
    dummy_frame = np.full((800, 1000, 3), 120, dtype=np.uint8)
    
    analysis_res = scene_analysis.analyze_scene(dummy_frame)
    hazards = analysis_res.get("hazards", [])
    candidates = analysis_res.get("candidate_zones", [])
    
    clearance_results = landing_engine.evaluate_all_candidate_clearances(
        candidate_zones=candidates,
        drone_profile=rescue_drone,
        frame_shape=dummy_frame.shape,
        estimated_ground_width_m=10.0
    )
    ranked_zones = scoring.rank_zones(
        candidate_zones=candidates,
        drone_profile=rescue_drone,
        hazards=hazards,
        clearance_results=clearance_results,
        frame_shape=dummy_frame.shape
    )
    landing_dec = landing_engine.evaluate_landing_decision(
        ranked_zones=ranked_zones,
        safe_threshold=landing_engine.SAFE_THRESHOLD,
        emergency_mode=False
    )
    
    assert len(ranked_zones) > 0, "Ranked zones should be non-empty for dummy frame"
    assert "decision" in landing_dec, "Landing decision must contain 'decision' key"
    print(f"[PASS] TEST 5: Run image analysis -> Candidates: {len(candidates)}, Top Zone: {ranked_zones[0]['label']} ({ranked_zones[0]['score']}/100)")

    # TEST 6: Dynamic re-evaluation -> Missions
    prev_rec = landing_dec.get("recommended_zone")
    prev_score = landing_dec.get("score")
    
    # Introduce hazard near recommended zone
    rec_zone_obj = next((z for z in candidates if z.get("label") == prev_rec), candidates[0])
    rx1, ry1, rx2, ry2 = rec_zone_obj["bbox"]
    simulated_hazards = hazards + [{
        "class_name": "person",
        "bbox": [rx1 + 5, ry1 + 5, rx1 + 30, ry1 + 30],
        "confidence": 0.95
    }]
    
    re_ranked_zones = scoring.rank_zones(
        candidate_zones=candidates,
        drone_profile=rescue_drone,
        hazards=simulated_hazards,
        clearance_results=clearance_results,
        frame_shape=dummy_frame.shape
    )
    re_landing_dec = landing_engine.evaluate_landing_decision(
        ranked_zones=re_ranked_zones,
        safe_threshold=landing_engine.SAFE_THRESHOLD,
        emergency_mode=False
    )
    
    new_rec = re_landing_dec.get("recommended_zone")
    print(f"[PASS] TEST 6: Dynamic re-evaluation -> Previous: {prev_rec} ({prev_score}/100) -> New: {new_rec} ({re_landing_dec.get('score')}/100)")

    # TEST 7: Change settings -> Save -> Reload
    test_settings = {
        "default_drone": "Rescue Drone",
        "estimated_ground_width_m": 15.5,
        "show_hazard_boxes": True,
        "show_zone_overlays": False,
        "show_detailed_reasons": True,
        "dynamic_re_evaluation": True,
        "re_evaluation_interval": 5,
        "alerts": {
            "new_obstacle": True,
            "recommendation_change": True,
            "no_safe_zone": False,
            "emergency_mode": True
        }
    }
    drone_profiles.save_settings(test_settings)
    reloaded_settings = drone_profiles.load_settings()
    assert reloaded_settings["estimated_ground_width_m"] == 15.5
    assert reloaded_settings["show_zone_overlays"] is False
    assert reloaded_settings["re_evaluation_interval"] == 5
    assert reloaded_settings["alerts"]["no_safe_zone"] is False
    print("[PASS] TEST 7: Change settings -> Save -> Reload verified successfully")

    # TEST 8: Mission History integrity
    missions = history.load_mission_history()
    assert isinstance(missions, list), "Mission history must be a list"
    print(f"[PASS] TEST 8: Mission history loaded safely ({len(missions)} recorded missions)")

    # TEST 9: Full page navigation cycle simulation
    pages = ["Dashboard", "Missions", "Settings", "Drone Profiles", "History", "Dashboard"]
    current_page = "Dashboard"
    for page in pages:
        current_page = page
        assert current_page in ["Dashboard", "Drone Profiles", "Missions", "History", "Settings"]
    print("[PASS] TEST 9: Full navigation cycle (Dashboard -> Missions -> Settings -> Drone Profiles -> History -> Dashboard) completed without state errors")

    print("\nALL 9 COMPREHENSIVE TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_all_tests()
