import sys
import os
sys.path.insert(0, os.getcwd())

import core.chat_assistant as chat_assistant

def run_chat_tests():
    print("=== RUNNING GROUNDED CHAT ASSISTANT UNIT TESTS ===")

    # TEST 1: No analysis available
    t1_reply = chat_assistant.answer_question("Which zone is safest?")
    print("TEST 1 (No analysis):", t1_reply[:60] + "...")
    assert "Run AI analysis first" in t1_reply

    # Setup mock analysis state
    ranked_zones = [
        {"label": "Zone A", "score": 92, "status": "SAFE", "clearance": {"passed": True}, "reasons": ["✓ Sufficient landing clearance", "✓ High distance from hazards"]},
        {"label": "Zone B", "score": 68, "status": "CAUTION", "clearance": {"passed": True}, "reasons": ["⚠ Car detected nearby", "⚠ Limited landing margin"]},
        {"label": "Zone C", "score": 31, "status": "UNSAFE", "clearance": {"passed": False, "reason": "Insufficient width"}, "reasons": ["❌ Insufficient landing clearance", "❌ Person inside zone"]}
    ]
    hazards = [
        {"class_name": "car", "confidence": 0.9, "bbox": [100, 100, 200, 200]},
        {"class_name": "person", "confidence": 0.85, "bbox": [300, 300, 350, 400]}
    ]
    landing_dec = {
        "decision": "RECOMMEND",
        "recommended_zone": "Zone A",
        "score": 92,
        "status": "SAFE",
        "zone": ranked_zones[0]
    }

    def safe_print(title, text):
        safe_t = text.encode('ascii', 'ignore').decode('ascii')
        print(f"\n{title}:\n {safe_t}")

    # TEST 2: Why Zone A?
    t2_reply = chat_assistant.answer_question("Why Zone A?", ranked_zones, hazards, landing_dec)
    safe_print("TEST 2 (Why Zone A?)", t2_reply)
    assert "Zone A" in t2_reply
    assert "92/100" in t2_reply
    assert "Sufficient landing clearance" in t2_reply

    # TEST 3: Can I land at Zone C? (UNSAFE)
    t3_reply = chat_assistant.answer_question("Can I land at Zone C?", ranked_zones, hazards, landing_dec)
    safe_print("TEST 3 (Can I land at Zone C?)", t3_reply)
    assert "UNSAFE" in t3_reply
    assert "31/100" in t3_reply

    # TEST 4: Which zone is safest?
    t4_reply = chat_assistant.answer_question("Which zone is safest?", ranked_zones, hazards, landing_dec)
    safe_print("TEST 4 (Which zone is safest?)", t4_reply)
    assert "Zone A" in t4_reply
    assert "92/100" in t4_reply

    # TEST 5: Detected hazards?
    t5_reply = chat_assistant.answer_question("Detected hazards?", ranked_zones, hazards, landing_dec)
    safe_print("TEST 5 (Detected hazards?)", t5_reply)
    assert "car" in t5_reply
    assert "person" in t5_reply

    # TEST 6: Analyze Zone B
    t6_reply = chat_assistant.answer_question("Analyze Zone B", ranked_zones, hazards, landing_dec)
    safe_print("TEST 6 (Analyze Zone B)", t6_reply)
    assert "Zone B" in t6_reply
    assert "68/100" in t6_reply
    assert "CAUTION" in t6_reply

    # TEST 7: Dynamic re-ranking changes Zone A -> Zone B
    ranked_zones_updated = [
        {"label": "Zone B", "score": 86, "status": "SAFE", "clearance": {"passed": True}, "reasons": ["✓ Sufficient landing clearance"]},
        {"label": "Zone A", "score": 28, "status": "UNSAFE", "clearance": {"passed": False}, "reasons": ["❌ Hazard inside zone"]}
    ]
    landing_dec_updated = {
        "decision": "RECOMMEND",
        "recommended_zone": "Zone B",
        "score": 86,
        "status": "SAFE",
        "zone": ranked_zones_updated[0]
    }
    t7_reply = chat_assistant.answer_question("Which zone is safest?", ranked_zones_updated, hazards, landing_dec_updated)
    safe_print("TEST 7 (Dynamic re-rank Grounding - Which zone is safest?)", t7_reply)
    assert "Zone B" in t7_reply
    assert "86/100" in t7_reply

    # TEST 8: NO SAFE LANDING ZONE case
    unsafe_dec = {
        "decision": "NO_SAFE_ZONE",
        "recommended_zone": None,
        "highest_candidate": "Zone A",
        "highest_score": 68
    }
    t8_reply = chat_assistant.answer_question("Which zone is safest?", ranked_zones[1:], hazards, unsafe_dec)
    safe_print("TEST 8 (NO SAFE LANDING ZONE)", t8_reply)
    assert "NO SAFE LANDING ZONE" in t8_reply

    # TEST 9: Emergency Mode active
    em_cand = {"label": "Zone B", "score": 68, "status": "CAUTION"}
    em_dec = {
        "decision": "NO_SAFE_ZONE",
        "recommended_zone": None,
        "highest_candidate": "Zone B",
        "highest_score": 68,
        "emergency_candidate": em_cand
    }
    t9_reply = chat_assistant.answer_question("Which zone is safest?", ranked_zones[1:], hazards, em_dec, emergency_mode=True)
    safe_print("TEST 9 (Emergency Mode active)", t9_reply)
    assert "emergency candidate only" in t9_reply

    print("\n[OK] ALL 9 CHAT ASSISTANT UNIT TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_chat_tests()
