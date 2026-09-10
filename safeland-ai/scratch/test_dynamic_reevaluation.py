import sys
import os
sys.path.insert(0, os.getcwd())

import core.landing_engine as landing_engine
import core.scoring as scoring
import core.drone_profiles as drone_profiles

def run_tests():
    print("=== RUNNING DYNAMIC RE-EVALUATION UNIT TESTS ===")

    # Mock Session State for unit testing
    class MockSessionState(dict):
        def __getattr__(self, key):
            return self.get(key)
        def __setattr__(self, key, value):
            self[key] = value

    state = MockSessionState()
    
    def reset_state():
        state["previous_ranked_zones"] = None
        state["previous_recommended_zone"] = None
        state["previous_top_score"] = None
        state["previous_decision"] = None
        state["previous_hazards"] = None

    def compute_alerts(curr_hazards, curr_ranked_zones, curr_decision):
        alerts = []
        prev_decision = state.get("previous_decision")
        prev_rec = state.get("previous_recommended_zone")

        if not prev_decision or not isinstance(prev_decision, dict):
            return alerts

        prev_dec_type = prev_decision.get("decision")
        prev_rec_label = prev_rec.get("label") if prev_rec else None
        prev_rec_score = prev_rec.get("score", 0) if prev_rec else 0
        prev_rec_bbox = prev_rec.get("bbox") if prev_rec else None

        curr_dec_type = curr_decision.get("decision")
        curr_rec = curr_decision.get("zone") if curr_dec_type == "RECOMMEND" else None
        curr_rec_label = curr_decision.get("recommended_zone")
        curr_rec_score = curr_decision.get("score", 0) if curr_rec else 0

        # Hazard intersection check
        if prev_rec_bbox and len(prev_rec_bbox) == 4:
            new_hazards = []
            for hz in curr_hazards:
                hz_box = hz.get("bbox")
                hz_class = hz.get("class_name", "hazard")
                if landing_engine.hazard_intersects_zone(hz_box, prev_rec_bbox):
                    new_hazards.append(hz_class)
            if new_hazards:
                alerts.append({
                    "type": "NEW_OBSTACLE",
                    "title": "🚨 NEW OBSTACLE DETECTED",
                    "message": f"New obstacle ({', '.join(set(new_hazards))}) detected near previous recommended zone ({prev_rec_label})."
                })

        # Decision / recommendation comparison
        if prev_dec_type == "RECOMMEND" and curr_dec_type == "RECOMMEND" and prev_rec_label == curr_rec_label:
            if prev_rec_score != curr_rec_score:
                alerts.append({
                    "type": "SCORE_UPDATE",
                    "title": f"Zone {prev_rec_label} Safety Score Updated",
                    "message": f"Zone {prev_rec_label} score updated: {prev_rec_score} -> {curr_rec_score}"
                })
        elif prev_dec_type == "RECOMMEND" and curr_dec_type == "RECOMMEND" and prev_rec_label != curr_rec_label:
            prev_z_in_curr = next((z for z in curr_ranked_zones if z.get("label") == prev_rec_label), None)
            curr_score_of_prev = prev_z_in_curr.get("score", 0) if prev_z_in_curr else 0
            alerts.append({
                "type": "RECOMMENDATION_CHANGED",
                "title": "🚨 LANDING RECOMMENDATION CHANGED",
                "prev_zone": prev_rec_label,
                "prev_score": prev_rec_score,
                "curr_prev_score": curr_score_of_prev,
                "new_recommendation": curr_rec_label
            })
        elif prev_dec_type == "RECOMMEND" and curr_dec_type == "NO_SAFE_ZONE":
            prev_z_in_curr = next((z for z in curr_ranked_zones if z.get("label") == prev_rec_label), None)
            curr_score_of_prev = prev_z_in_curr.get("score", 0) if prev_z_in_curr else 0
            alerts.append({
                "type": "NO_LONGER_SAFE",
                "title": "🚨 PREVIOUS LANDING ZONE NO LONGER SAFE",
                "prev_zone": prev_rec_label,
                "prev_score": prev_rec_score,
                "curr_prev_score": curr_score_of_prev
            })
        elif prev_dec_type == "NO_SAFE_ZONE" and curr_dec_type == "RECOMMEND":
            alerts.append({
                "type": "NEW_SAFE_AVAILABLE",
                "title": "🟢 SAFE LANDING CANDIDATE NOW AVAILABLE",
                "new_recommendation": curr_rec_label
            })

        return alerts

    def update_state(hazards, ranked_zones, decision_res):
        curr_rec = decision_res.get("zone") if decision_res.get("decision") == "RECOMMEND" else None
        state["previous_ranked_zones"] = ranked_zones
        state["previous_recommended_zone"] = curr_rec
        state["previous_top_score"] = decision_res.get("score", 0)
        state["previous_decision"] = decision_res
        state["previous_hazards"] = hazards

    # -------------------------------------------------------------
    # TEST 1: First Analysis (No alerts expected)
    # -------------------------------------------------------------
    reset_state()
    hz1 = []
    zones1 = [
        {"label": "Zone A", "score": 92, "status": "SAFE", "bbox": [100, 100, 250, 250], "clearance": {"passed": True}},
        {"label": "Zone B", "score": 85, "status": "SAFE", "bbox": [300, 100, 450, 250], "clearance": {"passed": True}}
    ]
    dec1 = landing_engine.evaluate_landing_decision(zones1)
    alerts1 = compute_alerts(hz1, zones1, dec1)
    update_state(hz1, zones1, dec1)

    print("TEST 1 (First Analysis): Alerts =", len(alerts1))
    assert len(alerts1) == 0
    assert dec1["decision"] == "RECOMMEND"
    assert dec1["recommended_zone"] == "Zone A"

    # -------------------------------------------------------------
    # TEST 2: Same scene, same top zone (No false change alert)
    # -------------------------------------------------------------
    dec2 = landing_engine.evaluate_landing_decision(zones1)
    alerts2 = compute_alerts(hz1, zones1, dec2)
    update_state(hz1, zones1, dec2)

    print("TEST 2 (Same Scene Rerun): Alerts =", len(alerts2))
    assert len(alerts2) == 0

    # -------------------------------------------------------------
    # TEST 3: Score decreases on top zone (Zone A 92 -> 84)
    # -------------------------------------------------------------
    zones3 = [
        {"label": "Zone A", "score": 84, "status": "SAFE", "bbox": [100, 100, 250, 250], "clearance": {"passed": True}},
        {"label": "Zone B", "score": 80, "status": "SAFE", "bbox": [300, 100, 450, 250], "clearance": {"passed": True}}
    ]
    dec3 = landing_engine.evaluate_landing_decision(zones3)
    alerts3 = compute_alerts(hz1, zones3, dec3)
    update_state(hz1, zones3, dec3)

    print("TEST 3 (Score decrease): Alerts =", [a["type"] for a in alerts3])
    assert len(alerts3) == 1
    assert alerts3[0]["type"] == "SCORE_UPDATE"

    # -------------------------------------------------------------
    # TEST 4: New hazard enters previous recommended Zone A
    # -------------------------------------------------------------
    hz4 = [{"class_name": "car", "confidence": 0.9, "bbox": [110, 110, 180, 180]}]
    zones4 = [
        {"label": "Zone B", "score": 85, "status": "SAFE", "bbox": [300, 100, 450, 250], "clearance": {"passed": True}},
        {"label": "Zone A", "score": 28, "status": "UNSAFE", "bbox": [100, 100, 250, 250], "clearance": {"passed": True}}
    ]
    dec4 = landing_engine.evaluate_landing_decision(zones4)
    alerts4 = compute_alerts(hz4, zones4, dec4)
    update_state(hz4, zones4, dec4)

    print("TEST 4 (New obstacle in Zone A & Zone B becomes top): Alerts =", [a["type"] for a in alerts4])
    assert any(a["type"] == "NEW_OBSTACLE" for a in alerts4)
    assert any(a["type"] == "RECOMMENDATION_CHANGED" for a in alerts4)
    rec_change_alert = next(a for a in alerts4 if a["type"] == "RECOMMENDATION_CHANGED")
    assert rec_change_alert["new_recommendation"] == "Zone B"
    assert rec_change_alert["prev_score"] == 84
    assert rec_change_alert["curr_prev_score"] == 28

    # -------------------------------------------------------------
    # TEST 5: All zones become unsafe (Zone B drops to 45)
    # -------------------------------------------------------------
    zones5 = [
        {"label": "Zone B", "score": 45, "status": "UNSAFE", "bbox": [300, 100, 450, 250], "clearance": {"passed": False}},
        {"label": "Zone A", "score": 20, "status": "UNSAFE", "bbox": [100, 100, 250, 250], "clearance": {"passed": False}}
    ]
    dec5 = landing_engine.evaluate_landing_decision(zones5)
    alerts5 = compute_alerts([], zones5, dec5)
    update_state([], zones5, dec5)

    print("TEST 5 (All zones become unsafe): Alerts =", [a["type"] for a in alerts5])
    assert any(a["type"] == "NO_LONGER_SAFE" for a in alerts5)
    assert dec5["decision"] == "NO_SAFE_ZONE"

    # -------------------------------------------------------------
    # TEST 6: Previously no safe zone, later safe zone appears (Zone B becomes 90)
    # -------------------------------------------------------------
    zones6 = [
        {"label": "Zone B", "score": 90, "status": "SAFE", "bbox": [300, 100, 450, 250], "clearance": {"passed": True}},
        {"label": "Zone A", "score": 30, "status": "UNSAFE", "bbox": [100, 100, 250, 250], "clearance": {"passed": False}}
    ]
    dec6 = landing_engine.evaluate_landing_decision(zones6)
    alerts6 = compute_alerts([], zones6, dec6)
    update_state([], zones6, dec6)

    print("TEST 6 (Safe zone appears from no-safe): Alerts =", [a["type"] for a in alerts6])
    assert any(a["type"] == "NEW_SAFE_AVAILABLE" for a in alerts6)
    assert dec6["decision"] == "RECOMMEND"
    assert dec6["recommended_zone"] == "Zone B"

    print("\n[OK] ALL DYNAMIC RE-EVALUATION UNIT TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_tests()
