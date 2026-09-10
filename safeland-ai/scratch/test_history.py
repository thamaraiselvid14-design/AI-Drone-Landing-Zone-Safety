import os
import sys
import json
import shutil
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import core.history as history

TEST_JSON_PATH = os.path.join(PROJECT_ROOT, "scratch", "temp_mission_history.json")

def test_history_all():
    print("--- 1. Testing Corrupt / Missing Storage Safety ---")
    if os.path.exists(TEST_JSON_PATH):
        os.remove(TEST_JSON_PATH)

    # Missing file
    res = history.load_mission_history(TEST_JSON_PATH)
    assert res == [], f"Expected [], got {res}"

    # Empty file
    with open(TEST_JSON_PATH, "w") as f:
        f.write("")
    res = history.load_mission_history(TEST_JSON_PATH)
    assert res == [], "Empty file should return []"

    # Malformed file
    with open(TEST_JSON_PATH, "w") as f:
        f.write("{invalid json...")
    res = history.load_mission_history(TEST_JSON_PATH)
    assert res == [], "Malformed file should return []"

    print("PASSED: Corrupt / missing storage handling.\n")

    print("--- 2. Testing ID Generation & Sequential Increment ---")
    m1_id = history.generate_mission_id(TEST_JSON_PATH)
    print(f"Generated ID 1: {m1_id}")
    assert m1_id.startswith("SL-")

    rec1 = {
        "mission_id": m1_id,
        "timestamp": "2026-09-11T01:30:00",
        "drone_used": "Rescue Drone",
        "input_type": "image",
        "candidate_count": 3,
        "initial_recommendation": {"zone": "Zone A", "score": 92, "status": "SAFE"},
        "obstacle_updates": [],
        "final_status": "RECOMMENDED",
        "final_recommendation": {"zone": "Zone A", "score": 92, "status": "SAFE"},
        "top_zone_reasons": ["✓ Sufficient clearance"]
    }
    history.append_mission(rec1, TEST_JSON_PATH)

    m2_id = history.generate_mission_id(TEST_JSON_PATH)
    print(f"Generated ID 2: {m2_id}")
    assert m2_id != m1_id, "IDs must be unique"
    assert int(m2_id.split("-")[-1]) == int(m1_id.split("-")[-1]) + 1, "Sequential increment failed"

    print("PASSED: Mission ID generation.\n")

    print("--- 3. Testing Duplicate Prevention & Record Update ---")
    rec1_updated = dict(rec1)
    rec1_updated["obstacle_updates"].append({
        "event": "NEW_OBSTACLE",
        "hazard": "person",
        "previous_zone": "Zone A",
        "previous_score": 92,
        "new_score": 28,
        "new_recommendation": "Zone B"
    })
    rec1_updated["final_recommendation"] = {"zone": "Zone B", "score": 86, "status": "SAFE"}

    # Update existing record
    history.update_mission(rec1_updated, TEST_JSON_PATH)

    loaded = history.load_mission_history(TEST_JSON_PATH)
    assert len(loaded) == 1, f"Expected 1 record after update, got {len(loaded)}"
    assert loaded[0]["final_recommendation"]["zone"] == "Zone B"
    assert len(loaded[0]["obstacle_updates"]) == 1

    print("PASSED: Duplicate prevention & in-place update.\n")

    print("--- 4. Testing Schema Validation for Required Fields ---")
    required_keys = [
        "mission_id", "timestamp", "drone_used", "input_type",
        "candidate_count", "initial_recommendation", "obstacle_updates",
        "final_status", "final_recommendation", "top_zone_reasons"
    ]
    for key in required_keys:
        assert key in loaded[0], f"Missing key '{key}' in mission record"

    print("PASSED: Record schema validation.\n")

    print("--- ALL HISTORY UNIT TESTS PASSED CLEANLY ---")

if __name__ == "__main__":
    test_history_all()
