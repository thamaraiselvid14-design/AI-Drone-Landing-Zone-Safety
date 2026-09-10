"""Voice Assistant Module for SafeLand AI

Provides:
1. Speech-to-text voice recognition using speech_recognition (Google Web Speech API).
2. Offline text-to-speech synthesis using pyttsx3.
3. Critical spoken alert narration for high-priority landing safety events.
4. Robust hardware/software error handling and text fallback.
"""

import threading
import re
from typing import Tuple, Optional, Dict, Any, List

_TTS_ENGINE = None

def get_tts_engine():
    """Lazy initialize pyttsx3 engine safely."""
    global _TTS_ENGINE
    if _TTS_ENGINE is not None:
        return _TTS_ENGINE
    try:
        import pyttsx3
        _TTS_ENGINE = pyttsx3.init()
        # Set comfortable rate and volume
        _TTS_ENGINE.setProperty('rate', 165)
        _TTS_ENGINE.setProperty('volume', 0.9)
        return _TTS_ENGINE
    except Exception:
        return None


def clean_text_for_speech(text: str) -> str:
    """Strip markdown formatting and emojis for clear speech synthesis."""
    if not text or not isinstance(text, str):
        return ""
    
    # Remove markdown bold/italics/headers
    clean = re.sub(r'[*#_`]', '', text)
    # Remove emojis / non-ascii symbols
    clean = re.sub(r'[^\x00-\x7F]+', ' ', clean)
    # Replace multiple spaces/newlines
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean


def _speak_worker(text: str):
    """Background worker for non-blocking TTS playback."""
    clean_msg = clean_text_for_speech(text)
    if not clean_msg:
        return
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 165)
        engine.setProperty('volume', 0.9)
        engine.say(clean_msg)
        engine.runAndWait()
        engine.stop()
    except Exception:
        pass


def speak_text_offline(text: str, non_blocking: bool = True):
    """Speak text using pyttsx3 offline text-to-speech engine.

    Args:
        text: Text message to speak aloud.
        non_blocking: Run TTS in a background thread to prevent UI freezing.
    """
    if not text:
        return
    
    if non_blocking:
        t = threading.Thread(target=_speak_worker, args=(text,), daemon=True)
        t.start()
    else:
        _speak_worker(text)


def listen_and_transcribe(timeout: int = 3) -> Tuple[Optional[str], Optional[str]]:
    """Capture audio from microphone and transcribe using SpeechRecognition (Google API).

    Args:
        timeout: Maximum seconds to wait for speech input before timing out.

    Returns:
        Tuple of (transcribed_text, error_message).
    """
    try:
        import speech_recognition as sr
    except Exception:
        return None, "SpeechRecognition package unavailable. Use text chat input."

    try:
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 300
        recognizer.dynamic_energy_threshold = True

        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=5)
            
        transcription = recognizer.recognize_google(audio)
        if transcription and transcription.strip():
            return transcription.strip(), None
        return None, "No spoken audio recognized. Please try again or use text chat."

    except sr.WaitTimeoutError:
        return None, "Microphone listening timed out. Speak immediately after clicking 🎙️ mic button or use text fallback."
    except sr.UnknownValueError:
        return None, "Could not understand spoken audio. Please speak clearly or use text input."
    except sr.RequestError as e:
        return None, f"Speech recognition service error: {str(e)}"
    except Exception as e:
        return None, f"Microphone hardware error: {str(e)}. Use text chat input."


def check_and_speak_critical_alerts(re_eval_alerts: List[Dict[str, Any]]):
    """Speak critical high-priority alerts tied to Phase 8 dynamic re-evaluation events.

    Triggers:
    1. "Warning, person detected near the recommended landing zone" (or obstacle)
    2. "Zone [X] is now the highest ranked candidate." (when recommendation changes)

    Skips minor score changes.
    """
    if not re_eval_alerts or not isinstance(re_eval_alerts, list):
        return

    for alert in re_eval_alerts:
        if not isinstance(alert, dict):
            continue
            
        a_type = alert.get("type")
        new_rec = alert.get("new_recommendation")
        msg = alert.get("message", "").lower()

        if a_type == "NEW_OBSTACLE":
            if "person" in msg:
                alert_speech = "Warning, person detected near the recommended landing zone."
            else:
                alert_speech = "Warning, obstacle detected near the recommended landing zone."
            speak_text_offline(alert_speech)
            break

        elif a_type == "RECOMMENDATION_CHANGED" and new_rec:
            alert_speech = f"Zone {new_rec} is now the highest ranked candidate."
            speak_text_offline(alert_speech)
            break

        elif a_type == "NEW_SAFE_AVAILABLE" and new_rec:
            alert_speech = f"Zone {new_rec} is now the highest ranked candidate."
            speak_text_offline(alert_speech)
            break
