import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import core.history as history

sample_missions = [
    {
        "mission_id": "SL-20260911-001",
        "timestamp": "2026-09-11T01:15:00",
        "drone_used": "Rescue Drone",
        "input_type": "image",
        "candidate_count": 3,
        "initial_recommendation": {
            "zone": "Zone A",
            "score": 92,
            "status": "SAFE"
        },
        "obstacle_updates": [],
        "final_status": "RECOMMENDED",
        "final_recommendation": {
            "zone": "Zone A",
            "score": 92,
            "status": "SAFE"
        },
        "top_zone_reasons": [
            "✓ Sufficient landing clearance (Required: 1.70m, Available: ~4.52m × 4.52m)",
            "✓ No nearby dynamic hazards detected inside perimeter",
            "✓ Large landing margin relative to drone dimensions"
        ]
    },
    {
        "mission_id": "SL-20260911-002",
        "timestamp": "2026-09-11T01:22:00",
        "drone_used": "Heavy Cargo Drone",
        "input_type": "video",
        "candidate_count": 3,
        "initial_recommendation": {
            "zone": "Zone A",
            "score": 92,
            "status": "SAFE"
        },
        "obstacle_updates": [
            {
                "event": "NEW_OBSTACLE",
                "hazard": "person",
                "previous_zone": "Zone A",
                "previous_score": 92,
                "new_score": 28,
                "new_recommendation": "Zone B"
            },
            {
                "event": "RECOMMENDATION_CHANGED",
                "hazard": "person",
                "previous_zone": "Zone A",
                "previous_score": 92,
                "new_score": 28,
                "new_recommendation": "Zone B"
            }
        ],
        "final_status": "RECOMMENDED",
        "final_recommendation": {
            "zone": "Zone B",
            "score": 86,
            "status": "SAFE"
        },
        "top_zone_reasons": [
            "✓ Sufficient landing clearance (Required: 4.24m, Available: ~4.52m × 4.52m)",
            "✓ Clear of dynamic hazards",
            "✓ Highest ranked safe candidate following Zone A score drop"
        ]
    },
    {
        "mission_id": "SL-20260911-003",
        "timestamp": "2026-09-11T01:28:00",
        "drone_used": "Rescue Drone",
        "input_type": "camera",
        "candidate_count": 2,
        "initial_recommendation": {
            "zone": None,
            "score": 45,
            "status": "NO_SAFE_ZONE"
        },
        "obstacle_updates": [],
        "final_status": "NO_SAFE_ZONE",
        "final_recommendation": {
            "zone": None,
            "score": 45,
            "status": "NO_SAFE_ZONE"
        },
        "top_zone_reasons": []
    }
]

def main():
    history.save_mission_history(sample_missions)
    print(f"Populated {len(sample_missions)} sample missions into data/mission_history.json")

if __name__ == "__main__":
    main()
