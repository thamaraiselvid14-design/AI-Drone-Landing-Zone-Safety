import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_page_routing():
    print("Testing page routing consistency...")
    valid_pages = ["Dashboard", "Drone Profiles", "Missions", "History", "Settings"]
    
    # Verify app.py contains all 5 page branches
    with open(PROJECT_ROOT / "app.py", "r", encoding="utf-8") as f:
        content = f.read()

    for page in valid_pages:
        assert f'"{page}"' in content, f"Page string '{page}' not found in app.py"
        print(f"OK: Page '{page}' present in app.py routing")

    print("\nALL NAVIGATION ROUTING TESTS PASSED CLEANLY!")

if __name__ == "__main__":
    test_page_routing()
