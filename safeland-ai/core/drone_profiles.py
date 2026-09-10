"""Drone Profiles Module

Handles loading, storing, and managing drone physical profiles, clearances,
and operational constraints for landing zone analysis.
"""

from typing import Dict, List, Optional, Any
import json
import os

_MODULE_DIR = os.path.dirname(globals().get("__file__", "."))
PROFILES_FILE = os.path.abspath(os.path.join(_MODULE_DIR, "..", "data", "drone_profiles.json"))
SETTINGS_FILE = os.path.abspath(os.path.join(_MODULE_DIR, "..", "data", "settings.json"))

DEMO_DRONES = [
    {
        "name": "Rescue Drone",
        "width_m": 1.8,
        "length_m": 1.6,
        "purpose": "Search & Rescue Operations",
        "required_clearance_m": 3.5
    },
    {
        "name": "Delivery Drone",
        "width_m": 1.2,
        "length_m": 1.0,
        "purpose": "Medical & Cargo Delivery",
        "required_clearance_m": 2.5
    },
    {
        "name": "Surveillance Drone",
        "width_m": 0.7,
        "length_m": 0.7,
        "purpose": "Reconnaissance & Aerial Inspection",
        "required_clearance_m": 1.5
    }
]


def calculate_required_clearance(width_m: float, length_m: float) -> float:
    """Calculate recommended safety clearance radius based on drone dimensions."""
    max_dim = max(width_m, length_m)
    return round(max_dim * 1.8, 1)


def save_all_drone_profiles(drones: List[Dict[str, Any]], filepath: str = PROFILES_FILE) -> bool:
    """Save the full list of drone profiles into JSON storage."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    try:
        clean_drones = [
            d for d in drones
            if isinstance(d, dict) and "name" in d and isinstance(d["name"], str)
        ]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"drones": clean_drones}, f, indent=2)
        return True
    except Exception:
        return False


def load_drone_profiles(filepath: str = PROFILES_FILE) -> List[Dict[str, Any]]:
    """Load all drone profiles from JSON storage safely.

    Returns:
        List of valid drone profile dictionaries. Guaranteed to be a list of dicts.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    def _sanitize(raw_data: Any) -> List[Dict[str, Any]]:
        raw_list = []
        if isinstance(raw_data, dict):
            raw_list = raw_data.get("drones", [])
        elif isinstance(raw_data, list):
            raw_list = raw_data

        if not isinstance(raw_list, list):
            return []

        clean = []
        for item in raw_list:
            if isinstance(item, dict) and "name" in item and isinstance(item["name"], str) and item["name"].strip():
                clean.append(item)
        return clean

    if not os.path.exists(filepath):
        save_all_drone_profiles(DEMO_DRONES, filepath)
        return _sanitize(DEMO_DRONES)

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        drones = _sanitize(data)
        if not drones:
            save_all_drone_profiles(DEMO_DRONES, filepath)
            return _sanitize(DEMO_DRONES)
        return drones
    except Exception:
        save_all_drone_profiles(DEMO_DRONES, filepath)
        return _sanitize(DEMO_DRONES)



def add_drone_profile(
    name: str,
    width_m: float,
    length_m: float,
    purpose: str,
    required_clearance_m: Optional[float] = None,
    filepath: str = PROFILES_FILE
) -> bool:
    """Add a new drone profile to storage.

    Args:
        name: Unique name of the drone.
        width_m: Width in meters.
        length_m: Length in meters.
        purpose: Mission purpose description.
        required_clearance_m: Required landing clearance in meters (optional).

    Returns:
        True if successfully added, False if duplicate or failed to write.
    """
    drones = load_drone_profiles(filepath)
    
    # Avoid exact duplicate names
    for d in drones:
        if d.get("name", "").strip().lower() == name.strip().lower():
            return False

    if required_clearance_m is None or required_clearance_m <= 0:
        required_clearance_m = calculate_required_clearance(width_m, length_m)

    new_drone = {
        "name": name.strip(),
        "width_m": float(width_m),
        "length_m": float(length_m),
        "purpose": purpose.strip(),
        "required_clearance_m": float(required_clearance_m)
    }
    drones.append(new_drone)
    return save_all_drone_profiles(drones, filepath)


def get_default_drone_name(filepath: str = SETTINGS_FILE) -> Optional[str]:
    """Return the name of the configured default drone from data/settings.json.

    If no default drone is configured, or if the stored drone name no longer
    exists in drone_profiles.json, return None.
    """
    if not os.path.exists(filepath):
        return None

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                return None

            drone_name = data.get("default_drone")
            if not drone_name or not isinstance(drone_name, str):
                return None

            # Verify that the stored drone name exists in drone_profiles.json
            drones = load_drone_profiles()
            existing_names = [d.get("name") for d in drones if isinstance(d, dict) and "name" in d]
            if drone_name in existing_names:
                return drone_name
            return None
    except Exception:
        return None


def set_default_drone_name(drone_name: str, filepath: str = SETTINGS_FILE) -> bool:
    """Save the default drone selection to settings.json."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"default_drone": drone_name}, f, indent=2)
        return True
    except Exception:
        return False


def set_default_drone(drone_name: str, filepath: str = SETTINGS_FILE) -> bool:
    """Alias for set_default_drone_name."""
    return set_default_drone_name(drone_name, filepath)


def get_drone_by_name(drone_name: Optional[str], filepath: str = PROFILES_FILE) -> Optional[Dict[str, Any]]:
    """Find and return a drone dictionary by name."""
    drones = load_drone_profiles(filepath)
    if not drones:
        return None

    if not drone_name:
        return drones[0]

    for d in drones:
        if isinstance(d, dict) and d.get("name") == drone_name:
            return d
    return drones[0]
