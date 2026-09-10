"""History Module for SafeLand AI

Manages persistent logging, retrieval, and updates of mission history and
landing decision records using JSON storage.
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

_MODULE_DIR = os.path.dirname(globals().get("__file__", "."))
DEFAULT_HISTORY_PATH = os.path.abspath(os.path.join(_MODULE_DIR, "..", "data", "mission_history.json"))


def load_mission_history(filepath: str = DEFAULT_HISTORY_PATH) -> List[Dict[str, Any]]:
    """Retrieve all past mission landing logs from JSON storage safely.

    Args:
        filepath: Path to mission_history.json.

    Returns:
        List of mission dictionaries. Always returns a list.
    """
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("missions"), list):
                return data["missions"]
            elif isinstance(data, list):
                return data
            return []
    except Exception:
        return []


def save_mission_history(missions: List[Dict[str, Any]], filepath: str = DEFAULT_HISTORY_PATH) -> bool:
    """Save mission list safely to JSON file.

    Args:
        missions: List of mission record dictionaries.
        filepath: Target JSON file path.

    Returns:
        True if successfully written, False otherwise.
    """
    if not isinstance(missions, list):
        return False
    try:
        dir_path = os.path.dirname(os.path.abspath(filepath))
        os.makedirs(dir_path, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"missions": missions}, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


def append_mission(record: Dict[str, Any], filepath: str = DEFAULT_HISTORY_PATH) -> bool:
    """Append a new mission record or update existing record by mission_id safely.

    Args:
        record: Mission record dictionary.
        filepath: Target JSON file path.

    Returns:
        True if successfully written, False otherwise.
    """
    if not isinstance(record, dict) or not record.get("mission_id"):
        return False
    missions = load_mission_history(filepath)
    
    # Check for existing mission_id to prevent duplicates
    existing_index = next((i for i, m in enumerate(missions) if isinstance(m, dict) and m.get("mission_id") == record["mission_id"]), -1)
    if existing_index >= 0:
        missions[existing_index] = record
    else:
        missions.append(record)
        
    return save_mission_history(missions, filepath)


def update_mission(record: Dict[str, Any], filepath: str = DEFAULT_HISTORY_PATH) -> bool:
    """Update an existing mission record by mission_id."""
    return append_mission(record, filepath)


def generate_mission_id(filepath: str = DEFAULT_HISTORY_PATH) -> str:
    """Generate a unique readable mission ID (e.g. SL-20260911-001).

    Args:
        filepath: Target JSON file path to check existing IDs.

    Returns:
        Unique mission ID string.
    """
    date_str = datetime.now().strftime("%Y%m%d")
    prefix = f"SL-{date_str}-"
    missions = load_mission_history(filepath)
    
    existing_ids = {m.get("mission_id", "") for m in missions if isinstance(m, dict)}
    count = 1
    while True:
        candidate_id = f"{prefix}{count:03d}"
        if candidate_id not in existing_ids:
            return candidate_id
        count += 1
