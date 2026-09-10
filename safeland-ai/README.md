# SafeLand AI

> **Adaptive & Explainable AI-Based Drone Landing Zone Detection and Safety Assessment**

SafeLand AI is a 24-hour hackathon prototype providing real-time AI decision support to evaluate, score, and rank potential autonomous and semi-autonomous drone landing zones.

---

## 📌 Project Overview

- **Project Name:** SafeLand AI
- **Subtitle:** Adaptive & Explainable Drone Landing Intelligence
- **Current Status:** **Phase 1 — Project setup and UI foundation.**

SafeLand AI combines computer vision, multi-factor spatial hazard detection, dynamic risk scoring, and explainable AI reasoning to ensure unmanned aerial vehicles (UAVs) land safely in varied terrain and emergency scenarios.

---

## 🛠️ Tech Stack

- **Language:** Python 3.11+
- **Frontend UI Framework:** Streamlit
- **Computer Vision & Image Processing:** OpenCV (`opencv-python`), Pillow (`PIL`), NumPy
- **Storage:** JSON (`data/drone_profiles.json`, `data/mission_history.json`)

---

## 📁 Project Structure

```
safeland-ai/
│
├── app.py                   # Streamlit Web Application Entry Point
│
├── core/                    # Core Python Modules
│   ├── __init__.py          # Package Initializer
│   ├── drone_profiles.py    # Drone Profile & Dimensions Management
│   ├── scene_analysis.py    # Computer Vision & Scene Processing
│   ├── landing_engine.py    # Candidate Zone Generation & Decision Engine
│   ├── scoring.py           # Composite Safety & Risk Scoring
│   └── history.py           # Mission Log Persistence & History
│
├── assets/                  # Static Assets & Images
│
├── data/                    # Storage Layer
│   ├── drone_profiles.json  # Saved Drone Specifications
│   └── mission_history.json # Historical Mission Assessment Logs
│
├── requirements.txt         # Pinned Dependencies
└── README.md                # Documentation
```

---

## 🚀 Quick Start & How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit Application

```bash
streamlit run app.py
```

Open your browser to the local URL provided by Streamlit (typically `http://localhost:8501`).

---

## 🗺️ Roadmap & Future Phases

Future development phases will introduce:

- 🚁 **Drone Profiles:** Custom UAV dimensions, clearance radii, wind tolerances, and landing gear specs.
- 📷 **Multi-Input Feed:** Support for Image Upload, Video Upload, and Live Camera streams.
- 🔍 **Scene Analysis:** Computer vision terrain segmentation, obstacle mapping, and slope estimation.
- 🎯 **Candidate Landing Zones:** Automated candidate region proposal and clearance verification.
- 📊 **Safety Scoring:** Multi-factor safety index calculation and risk categorization (SAFE / RISKY / UNSAFE).
- 🏆 **Zone Ranking:** Multi-candidate ranking and dynamic re-ranking based on changing telemetry.
- 💬 **Chat/Voice Support:** Natural language and voice query assistant for pilot landing decisions.
- 📜 **Mission History:** Persistent mission log review and analytical reporting.
