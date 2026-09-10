import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.voice_assistant as voice_assistant

def test_voice_module():
    print("Testing clean_text_for_speech...")
    raw_text = "🤖 **SafeLand Assistant**: Zone A is SAFE! ✓ (Score: 95/100)"
    cleaned = voice_assistant.clean_text_for_speech(raw_text)
    print(f"Cleaned text: '{cleaned}'")
    assert "SafeLand Assistant" in cleaned
    assert "*" not in cleaned
    assert "✓" not in cleaned

    print("\nTesting speak_text_offline (non-blocking)...")
    voice_assistant.speak_text_offline("Testing SafeLand AI offline voice synthesis.")

    print("\nTesting check_and_speak_critical_alerts...")
    sample_alerts = [
        {
            "type": "NEW_OBSTACLE",
            "title": "🚨 NEW OBSTACLE DETECTED",
            "message": "New obstacle (person) detected near previous recommended landing zone (Zone A).",
            "color": "amber"
        },
        {
            "type": "RECOMMENDATION_CHANGED",
            "title": "🚨 LANDING RECOMMENDATION CHANGED",
            "message": "Previous recommended landing zone Zone A score dropped: 90 -> 40.",
            "new_recommendation": "Zone B",
            "color": "red"
        }
    ]
    voice_assistant.check_and_speak_critical_alerts(sample_alerts)

    print("\nTesting listen_and_transcribe timeout/fallback handling...")
    # Will timeout quickly or report hardware status without crashing
    text, err = voice_assistant.listen_and_transcribe(timeout=1)
    print(f"Transcription result: text='{text}', err='{err}'")

    print("\nAll voice assistant tests passed successfully!")

if __name__ == "__main__":
    test_voice_module()
