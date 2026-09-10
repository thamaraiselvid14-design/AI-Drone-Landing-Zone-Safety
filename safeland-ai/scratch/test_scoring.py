"""Test Safety Scoring and Zone Ranking Engine (core/scoring.py)

Validates:
TEST 1: Passed clearance + no nearby hazards -> high score (80-100, SAFE)
TEST 2: Passed clearance + one nearby car/person -> lower score (50-79, CAUTION)
TEST 3: Hazard moved closer to zone -> score decreases
TEST 4: Additional nearby hazard added -> score decreases
TEST 5: Clearance rejected -> low score (<=49) and UNSAFE
TEST 6: No hazards -> maximum hazard contribution
TEST 7: Multiple zones -> sorted highest score to lowest score with 1-based ranks
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import core.scoring as scoring
import core.landing_engine as landing_engine


def test_scoring_engine():
    print("=== STARTING SCORING & RANKING ENGINE TESTS ===")

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

    # Candidate zone A: 280x280px at (100, 100, 380, 380) -> 2.8m x 2.8m at 10m scale
    cand_zone_a = {
        "label": "Zone A",
        "bbox": [100, 100, 380, 380]
    }

    clr_surv_passed = landing_engine.check_clearance(cand_zone_a, surveillance_drone, frame_shape, 10.0)
    clr_rescue_failed = landing_engine.check_clearance(cand_zone_a, rescue_drone, frame_shape, 10.0)

    # TEST 1: Passed clearance + no nearby hazards
    print("\n--- TEST 1: Passed clearance + no nearby hazards ---")
    score_t1 = scoring.score_zone(cand_zone_a, surveillance_drone, [], clr_surv_passed, frame_shape)
    print(f"Test 1 Result: score={score_t1['score']}, status={score_t1['status']}, color={score_t1['color']}")
    assert isinstance(score_t1["score"], int)
    assert 80 <= score_t1["score"] <= 100
    assert score_t1["status"] == "SAFE"
    assert score_t1["color"] == "green"
    assert "✓ Sufficient landing clearance" in score_t1["reasons"]
    assert "✓ No nearby dynamic hazards" in score_t1["reasons"]

    # TEST 2: Passed clearance + one nearby car
    print("\n--- TEST 2: Passed clearance + one nearby car ---")
    hazards_t2 = [{
        "class_name": "car",
        "confidence": 0.85,
        "bbox": [420, 100, 550, 200] # ~40px distance
    }]
    score_t2 = scoring.score_zone(cand_zone_a, surveillance_drone, hazards_t2, clr_surv_passed, frame_shape)
    print(f"Test 2 Result: score={score_t2['score']}, status={score_t2['status']}")
    assert score_t2["score"] < score_t1["score"], "Score with nearby car should be lower than no hazards"
    assert any("Car detected" in r for r in score_t2["reasons"])

    # TEST 3: Hazard moved closer to zone
    print("\n--- TEST 3: Hazard moved closer to zone ---")
    hazards_t3_closer = [{
        "class_name": "car",
        "confidence": 0.85,
        "bbox": [385, 100, 515, 200] # ~5px distance (much closer!)
    }]
    score_t3 = scoring.score_zone(cand_zone_a, surveillance_drone, hazards_t3_closer, clr_surv_passed, frame_shape)
    print(f"Test 3 Result: Far score={score_t2['score']} vs Closer score={score_t3['score']}")
    assert score_t3["score"] < score_t2["score"], "Moving hazard closer must decrease the score"

    # TEST 4: Additional nearby hazard added
    print("\n--- TEST 4: Additional nearby hazard added ---")
    hazards_t4_multiple = [
        {"class_name": "car", "confidence": 0.85, "bbox": [420, 100, 550, 200]},
        {"class_name": "person", "confidence": 0.90, "bbox": [100, 420, 180, 500]}
    ]
    score_t4 = scoring.score_zone(cand_zone_a, surveillance_drone, hazards_t4_multiple, clr_surv_passed, frame_shape)
    print(f"Test 4 Result: 1 hazard score={score_t2['score']} vs 2 hazards score={score_t4['score']}")
    assert score_t4["score"] < score_t2["score"], "Adding another nearby hazard must decrease score"

    # TEST 5: Clearance rejected
    print("\n--- TEST 5: Clearance rejected ---")
    score_t5 = scoring.score_zone(cand_zone_a, rescue_drone, [], clr_rescue_failed, frame_shape)
    print(f"Test 5 Result: score={score_t5['score']}, status={score_t5['status']}")
    assert score_t5["score"] <= 49, "Rejected clearance score must be <= 49"
    assert score_t5["status"] == "UNSAFE"
    assert score_t5["color"] == "red"
    assert "✕ Insufficient landing clearance" in score_t5["reasons"]

    # TEST 6: No hazards
    print("\n--- TEST 6: No hazards ---")
    score_t6 = scoring.score_zone(cand_zone_a, surveillance_drone, [], clr_surv_passed, frame_shape)
    assert score_t6["score_breakdown"]["distance_factor"] == 25.0
    assert score_t6["score_breakdown"]["count_factor"] == 20.0

    # TEST 7: Multiple zones & ranking
    print("\n--- TEST 7: Multiple zones ranking ---")
    zones = [
        {"label": "Zone C", "bbox": [10, 10, 110, 110]},   # Small 1.0m zone
        {"label": "Zone A", "bbox": [100, 100, 380, 380]}, # 2.8m zone
        {"label": "Zone B", "bbox": [50, 50, 450, 450]}    # 4.0m zone
    ]
    clr_results = landing_engine.evaluate_all_candidate_clearances(zones, surveillance_drone, frame_shape, 10.0)
    ranked = scoring.rank_zones(zones, surveillance_drone, [], clr_results, frame_shape)

    print("Ranked Zones Result:")
    for r in ranked:
        print(f"  Rank #{r['rank']} - {r['label']}: Score={r['score']}, Status={r['status']}")

    assert len(ranked) == 3
    assert ranked[0]["rank"] == 1
    assert ranked[1]["rank"] == 2
    assert ranked[2]["rank"] == 3
    assert ranked[0]["score"] >= ranked[1]["score"] >= ranked[2]["score"], "Ranked zones must be sorted descending by score"

    # Verify no unsupported claims in any reasons
    unsupported_terms = ["tree", "water", "flood", "debris", "slope"]
    for r in ranked:
        for reason in r["reasons"]:
            for term in unsupported_terms:
                assert term not in reason.lower(), f"Unsupported term '{term}' found in reason: '{reason}'"

    print("\n[SUCCESS] ALL SCORING & ZONE RANKING ENGINE TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_scoring_engine()
