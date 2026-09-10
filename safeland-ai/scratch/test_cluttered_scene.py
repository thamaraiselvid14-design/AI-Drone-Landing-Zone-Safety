import sys
import os
sys.path.insert(0, os.getcwd())

import numpy as np
import cv2
import core.scene_analysis as scene_analysis
import core.landing_engine as landing_engine
import core.scoring as scoring
import core.drone_profiles as drone_profiles

def run_cluttered_test():
    print("=== DENSELY CLUTTERED IMAGE TEST ===")
    
    # 1. Generate a synthetic cluttered frame with high edge density & multiple mock hazards
    h, w = 480, 640
    frame = np.full((h, w, 3), 120, dtype=np.uint8)
    
    # Add dense noise/texture to lower openness scores across the entire frame
    np.random.seed(42)
    noise = np.random.randint(-40, 40, (h, w, 3), dtype=np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Draw several grid lines to simulate dense clutter / structures
    for x in range(0, w, 40):
        cv2.line(frame, (x, 0), (x, h), (30, 30, 30), 2)
    for y in range(0, h, 40):
        cv2.line(frame, (0, y), (w, y), (30, 30, 30), 2)

    # 2. Scene analysis
    hazards = [
        {"class_name": "car", "confidence": 0.91, "bbox": [100, 100, 250, 250]},
        {"class_name": "person", "confidence": 0.88, "bbox": [300, 150, 380, 280]},
        {"class_name": "truck", "confidence": 0.95, "bbox": [400, 250, 580, 420]}
    ]
    
    cand_zones = [
        {"bbox": [50, 50, 200, 200]},
        {"bbox": [280, 100, 420, 240]},
        {"bbox": [400, 50, 550, 200]}
    ]
    
    drone = drone_profiles.load_drone_profiles()[0]
    
    clearance_results = landing_engine.evaluate_all_candidate_clearances(
        candidate_zones=cand_zones,
        drone_profile=drone,
        frame_shape=frame.shape,
        estimated_ground_width_m=10.0
    )
    
    ranked_zones = scoring.rank_zones(
        candidate_zones=cand_zones,
        drone_profile=drone,
        hazards=hazards,
        clearance_results=clearance_results,
        frame_shape=frame.shape
    )
    
    print(f"Evaluated {len(ranked_zones)} ranked zones in cluttered scene:")
    for z in ranked_zones:
        safe_reasons = [r.encode('ascii', 'ignore').decode('ascii') for r in z.get('reasons', [])]
        print(f"  {z['label']}: score={z['score']}, status={z['status']}, reasons={safe_reasons}")
        assert z['score'] < landing_engine.SAFE_THRESHOLD, f"Cluttered zone score {z['score']} should be below {landing_engine.SAFE_THRESHOLD}"

    # TEST A: Emergency Mode OFF
    res_off = landing_engine.evaluate_landing_decision(ranked_zones, emergency_mode=False)
    print("\n--- Emergency Mode OFF ---")
    print("Decision:", res_off["decision"])
    print("Recommended Zone:", res_off["recommended_zone"])
    print("Highest Candidate:", res_off["highest_candidate"], f"({res_off['highest_score']}/100)")
    print("Emergency Candidate:", res_off["emergency_candidate"])
    
    assert res_off["decision"] == "NO_SAFE_ZONE"
    assert res_off["recommended_zone"] is None
    assert res_off["highest_candidate"] is not None
    assert res_off["emergency_candidate"] is None

    # TEST B: Emergency Mode ON
    res_on = landing_engine.evaluate_landing_decision(ranked_zones, emergency_mode=True)
    print("\n--- Emergency Mode ON ---")
    print("Decision:", res_on["decision"])
    print("Recommended Zone:", res_on["recommended_zone"])
    print("Highest Candidate:", res_on["highest_candidate"], f"({res_on['highest_score']}/100)")
    print("Emergency Candidate:", res_on["emergency_candidate"]["label"])
    print("Emergency Status:", res_on["emergency_candidate"]["status"])
    safe_em_reason = res_on["emergency_reason"].encode('ascii', 'ignore').decode('ascii') if res_on["emergency_reason"] else ""
    print("Emergency Reason:", safe_em_reason)
    
    assert res_on["decision"] == "NO_SAFE_ZONE"
    assert res_on["recommended_zone"] is None
    assert res_on["emergency_candidate"] is not None
    assert res_on["emergency_candidate"]["status"] != "SAFE"  # MUST NOT BE RELABELED SAFE!
    assert res_on["emergency_reason"] is not None

    print("\n[OK] DENSELY CLUTTERED SCENE TEST PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_cluttered_test()
