import sys
import os
sys.path.insert(0, os.getcwd())

import core.landing_engine as landing_engine

def run_tests():
    print("=== RUNNING SAFE LANDING THRESHOLD & EMERGENCY MODE TESTS ===")
    
    # TEST 1: Zone A = 92 (clearance passed), Zone B = 70 (clearance passed)
    # Expected: Recommend Zone A
    t1_zones = [
        {"label": "Zone A", "score": 92, "status": "SAFE", "clearance": {"passed": True}, "reasons": ["✓ Sufficient clearance"]},
        {"label": "Zone B", "score": 70, "status": "CAUTION", "clearance": {"passed": True}, "reasons": ["⚠ Vehicle nearby"]}
    ]
    res1 = landing_engine.evaluate_landing_decision(t1_zones, emergency_mode=False)
    print("TEST 1 (Zone A=92, B=70, clearance passed):", res1["decision"], "| Recommended:", res1["recommended_zone"])
    assert res1["decision"] == "RECOMMEND"
    assert res1["recommended_zone"] == "Zone A"
    assert res1["score"] == 92
    assert res1["status"] == "SAFE"

    # TEST 2: Zone A = 79 (clearance passed), Zone B = 65 (clearance passed), Emergency Mode OFF
    # Expected: NO SAFE LANDING ZONE
    t2_zones = [
        {"label": "Zone A", "score": 79, "status": "CAUTION", "clearance": {"passed": True}, "reasons": ["⚠ Small margin"]},
        {"label": "Zone B", "score": 65, "status": "CAUTION", "clearance": {"passed": True}, "reasons": ["⚠ Person nearby"]}
    ]
    res2 = landing_engine.evaluate_landing_decision(t2_zones, emergency_mode=False)
    print("TEST 2 (Zone A=79, B=65, Emergency OFF):", res2["decision"], "| Highest:", res2["highest_candidate"], "| Emergency cand:", res2["emergency_candidate"])
    assert res2["decision"] == "NO_SAFE_ZONE"
    assert res2["recommended_zone"] is None
    assert res2["highest_candidate"] == "Zone A"
    assert res2["highest_score"] == 79
    assert res2["emergency_candidate"] is None

    # TEST 3: Same as TEST 2, Emergency Mode ON
    # Expected: NO SAFE LANDING ZONE plus ⚠ Emergency Candidate Only = Zone A
    res3 = landing_engine.evaluate_landing_decision(t2_zones, emergency_mode=True)
    print("TEST 3 (Zone A=79, B=65, Emergency ON):", res3["decision"], "| Highest:", res3["highest_candidate"], "| Emergency cand:", res3["emergency_candidate"]["label"], "| Status:", res3["emergency_candidate"]["status"])
    assert res3["decision"] == "NO_SAFE_ZONE"
    assert res3["recommended_zone"] is None
    assert res3["highest_candidate"] == "Zone A"
    assert res3["emergency_candidate"] is not None
    assert res3["emergency_candidate"]["label"] == "Zone A"
    assert res3["emergency_candidate"]["status"] != "SAFE"  # NEVER labeled SAFE!

    # TEST 4: Zone A = 90 (clearance failed), Zone B = 84 (clearance passed)
    # Expected: Recommend Zone B
    t4_zones = [
        {"label": "Zone A", "score": 90, "status": "SAFE", "clearance": {"passed": False, "reason": "Insufficient width"}, "reasons": ["Insufficient clearance"]},
        {"label": "Zone B", "score": 84, "status": "SAFE", "clearance": {"passed": True}, "reasons": ["✓ Sufficient clearance"]}
    ]
    res4 = landing_engine.evaluate_landing_decision(t4_zones, emergency_mode=False)
    print("TEST 4 (Zone A=90 clearance failed, B=84 clearance passed):", res4["decision"], "| Recommended:", res4["recommended_zone"])
    assert res4["decision"] == "RECOMMEND"
    assert res4["recommended_zone"] == "Zone B"
    assert res4["score"] == 84

    # TEST 5: All zones below 80 and all clearance failed, Emergency Mode OFF
    # Expected: NO SAFE LANDING ZONE
    t5_zones = [
        {"label": "Zone A", "score": 45, "status": "UNSAFE", "clearance": {"passed": False}, "reasons": ["❌ Hazard inside zone"]},
        {"label": "Zone B", "score": 40, "status": "UNSAFE", "clearance": {"passed": False}, "reasons": ["❌ Clearance rejected"]}
    ]
    res5 = landing_engine.evaluate_landing_decision(t5_zones, emergency_mode=False)
    print("TEST 5 (All below 80 & clearance failed, Emergency OFF):", res5["decision"], "| Emergency cand:", res5["emergency_candidate"])
    assert res5["decision"] == "NO_SAFE_ZONE"
    assert res5["recommended_zone"] is None
    assert res5["emergency_candidate"] is None

    # TEST 6: All zones below 80, Emergency Mode ON
    # Expected: Show highest-ranked candidate as emergency-only fallback, NOT labeled SAFE
    res6 = landing_engine.evaluate_landing_decision(t5_zones, emergency_mode=True)
    print("TEST 6 (All below 80, Emergency ON):", res6["decision"], "| Emergency cand:", res6["emergency_candidate"]["label"], "| Status:", res6["emergency_candidate"]["status"])
    assert res6["decision"] == "NO_SAFE_ZONE"
    assert res6["recommended_zone"] is None
    assert res6["emergency_candidate"] is not None
    assert res6["emergency_candidate"]["label"] == "Zone A"
    assert res6["emergency_candidate"]["status"] != "SAFE"

    # TEST 7: No candidate zones
    # Expected: NO SAFE LANDING ZONE Continue Search
    res7 = landing_engine.evaluate_landing_decision([], emergency_mode=True)
    print("TEST 7 (No candidate zones):", res7["decision"], "| Highest:", res7["highest_candidate"])
    assert res7["decision"] == "NO_SAFE_ZONE"
    assert res7["recommended_zone"] is None
    assert res7["highest_candidate"] is None

    print("\n[OK] ALL 7 UNIT TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_tests()
