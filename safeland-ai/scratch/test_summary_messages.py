import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def evaluate_summary_message(candidates, clearance_results, decision_res, safe_threshold=80):
    cand_count = len(candidates)
    clearance_passed_count = sum(1 for c in clearance_results if (c.get("passed", False) or c.get("clearance", {}).get("passed", False)))
    decision = decision_res.get("decision", "NO_SAFE_ZONE")

    if decision == "RECOMMEND" and decision_res.get("recommended_zone"):
        return "NORMAL_RECOMMENDED_CARD"
    elif cand_count == 0:
        return "🔴 No landing candidate detected in the current frame."
    elif clearance_passed_count == 0:
        return "Candidate regions were detected, but none satisfy the selected drone's clearance requirement."
    else:
        return "Candidate zones were detected, but none satisfy the required safety threshold."

def test_all_scenarios():
    print("Testing Summary Messages for all 4 scenarios...")

    # Scenario 1: candidate_zones count == 0
    msg1 = evaluate_summary_message(
        candidates=[],
        clearance_results=[],
        decision_res={"decision": "NO_SAFE_ZONE"}
    )
    print(f"Scenario 1: {msg1.encode('ascii', 'ignore').decode('ascii')}")
    assert msg1 == "🔴 No landing candidate detected in the current frame."

    # Scenario 2: candidates exist BUT all clearance checks fail
    msg2 = evaluate_summary_message(
        candidates=[{"label": "Zone A"}, {"label": "Zone B"}],
        clearance_results=[{"label": "Zone A", "passed": False}, {"label": "Zone B", "passed": False}],
        decision_res={"decision": "NO_SAFE_ZONE", "highest_candidate": "Zone A", "highest_score": 35}
    )
    print(f"Scenario 2: {msg2}")
    assert msg2 == "Candidate regions were detected, but none satisfy the selected drone's clearance requirement."

    # Scenario 3: some zones pass clearance BUT all scores are below SAFE_THRESHOLD (80)
    msg3 = evaluate_summary_message(
        candidates=[{"label": "Zone A"}, {"label": "Zone B"}],
        clearance_results=[{"label": "Zone A", "passed": True}, {"label": "Zone B", "passed": False}],
        decision_res={"decision": "NO_SAFE_ZONE", "highest_candidate": "Zone A", "highest_score": 65}
    )
    print(f"Scenario 3: {msg3}")
    assert msg3 == "Candidate zones were detected, but none satisfy the required safety threshold."

    # Scenario 4: at least one zone is safe (score >= 80 and passed clearance)
    msg4 = evaluate_summary_message(
        candidates=[{"label": "Zone A"}, {"label": "Zone B"}],
        clearance_results=[{"label": "Zone A", "passed": True}],
        decision_res={"decision": "RECOMMEND", "recommended_zone": "Zone A", "score": 92}
    )
    print(f"Scenario 4: {msg4}")
    assert msg4 == "NORMAL_RECOMMENDED_CARD"

    print("\nALL SUMMARY MESSAGE SCENARIO TESTS PASSED CLEANLY!")

if __name__ == "__main__":
    test_all_scenarios()
