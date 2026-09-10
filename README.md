# 🚁 SafeLand AI – AI Drone Landing Zone Safety

SafeLand AI is an AI-assisted drone landing-zone safety prototype that analyzes images, video frames, and live-camera captures to identify and rank potential landing zones.

The system evaluates candidate zones based on detected hazards, available clearance, drone requirements, and safety scores. It recommends the safest available zone while keeping the **final landing decision with the human operator or drone control system**.

---

## 🎯 Project Objective

The main objective of SafeLand AI is to help drones identify safer landing areas in complex or changing environments.

The system is designed to:

- Detect possible obstacles and hazards.
- Identify candidate landing zones.
- Evaluate whether a zone has enough landing clearance.
- Calculate safety scores for each candidate.
- Rank landing zones from safest to least safe.
- Reject unsafe landing zones.
- Re-evaluate zones when the environment changes.
- Support emergency decision-making when no safe zone is available.

---

## 💡 Problem Statement

Drone landing becomes difficult in unknown, cluttered, or dynamically changing environments.

A landing area that initially appears safe may become unsafe because of:

- People
- Vehicles
- Objects
- Insufficient landing space
- Newly appearing obstacles
- Changes in the scene

SafeLand AI provides an intelligent decision-support system that continuously evaluates candidate landing areas and recommends safer alternatives.

---

## ⚙️ How SafeLand AI Works

```text
Image / Video / Live Camera
          ↓
     Frame Processing
          ↓
 AI-Based Scene Analysis
          ↓
   Hazard Detection
          ↓
Candidate Zone Generation
          ↓
Landing Clearance Check
          ↓
 Safety Score Calculation
          ↓
    Zone Ranking
          ↓
 Safety Threshold Check
          ↓
Landing Recommendation
          ↓
Dynamic Re-Evaluation
```

---

## ✨ Key Features

### 📥 Multiple Input Sources

SafeLand AI supports:

- Image upload
- Drone video upload and frame selection
- Live camera capture

### 🔍 Scene & Hazard Analysis

The system analyzes the selected frame to detect potential hazards and understand whether candidate regions are suitable for landing.

### 📍 Candidate Landing Zones

Potential landing areas are generated and evaluated individually.

Example:

```text
Zone A
Zone B
Zone C
Zone D
```

### 📏 Drone-Specific Clearance Checking

SafeLand AI considers the selected drone's dimensions and required landing clearance.

The system can therefore reject a visually clear area when the physical space is not sufficient for the selected drone.

### 📊 Safety Scoring & Ranking

Each candidate landing zone receives a safety score.

Example:

```text
Zone A → 92/100
Zone B → 81/100
Zone C → 63/100
Zone D → 38/100
```

The zones are ranked based on their calculated safety.

---

## 🔴 No Safe Landing Zone

SafeLand AI uses a configurable safety threshold.

If the highest-ranked candidate does not satisfy the required safety threshold, the system does not provide it as a normal landing recommendation.

Instead, it displays:

```text
🔴 NO SAFE LANDING ZONE

Safety threshold not satisfied.

Recommendation:
Abort landing / continue searching.
```

This prevents the system from presenting the "best among unsafe zones" as though it were genuinely safe.

---

## 🚨 Emergency Mode

Emergency Mode is provided for situations where no candidate satisfies the normal safety requirement.

When Emergency Mode is enabled:

1. The system first performs normal safety analysis.
2. If a safe zone exists, the normal recommendation is used.
3. If no safe zone exists, the system identifies the least-risk candidate.
4. The candidate is clearly labelled:

```text
⚠ Emergency candidate only
```

5. The interface also explains why the candidate remains risky.

Emergency Mode **does not convert an unsafe zone into a safe zone**. It only provides additional decision support when normal safe-landing criteria cannot be satisfied.

---

## 🔄 Dynamic Re-Evaluation

SafeLand AI can re-analyze a new camera or video frame when environmental conditions change.

For example:

```text
Initial Analysis:

Zone A → 92/100
🏆 Recommended: Zone A
```

If a new obstacle appears:

```text
🚨 NEW OBSTACLE DETECTED

🔄 Re-evaluating landing zones...

Zone A: 92 → 28

🏆 New recommended landing zone: Zone B
```

Previous analysis results are maintained using Streamlit session state so the system can compare the old and new rankings.

---

## 💬 Ask SafeLand AI

The dashboard includes a context-aware SafeLand AI assistant.

Unlike a generic chatbot, the assistant uses the **current landing analysis results**, including:

- Zone scores
- Zone ranking
- Detected hazards
- Safety reasons
- Rejection reasons

Example questions:

```text
Why Zone A?

Can I land at Zone B?

Which zone is safest?

Detected hazards?
```

The answers therefore depend on the current analyzed scene.

---

## 🚁 Drone Profiles

SafeLand AI supports different UAV configurations.

Example profiles include:

### Rescue Drone
- Width: 1.8 m
- Length: 1.6 m
- Required Clearance: 3.5 m
- Application: Search & Rescue Operations

### Delivery Drone
- Width: 1.2 m
- Length: 1.0 m
- Required Clearance: 2.5 m
- Application: Medical & Cargo Delivery

### Surveillance Drone
- Width: 0.7 m
- Length: 0.7 m
- Required Clearance: 1.5 m
- Application: Reconnaissance & Aerial Inspection

Different drone dimensions can result in different landing-zone decisions.

---

## 📜 Mission History

Completed analyses are recorded in:

```text
data/mission_history.json
```

Mission records can contain information such as:

- Mission ID
- Timestamp
- Selected drone
- Input source
- Number of candidate zones
- Initial recommendation
- Obstacle-triggered updates
- Final status
- Safety reasons

The History page allows previous analyses to be reviewed.

---

## 🧠 Technologies Used

### Python

Python is the main programming language used for the application's analysis logic, scoring, data processing, and backend functionality.

### Streamlit

Streamlit is used to create the interactive web dashboard and connect the Python analysis pipeline directly with the user interface.

### OpenCV

OpenCV is used for computer-vision operations such as:

- Reading images
- Processing frames
- Video-frame handling
- Image annotation
- Bounding-box operations
- Visual analysis

### NumPy

NumPy provides efficient numerical and array operations for image and analysis data.

### Pillow

Pillow is used for image loading, conversion, manipulation, and compatibility between uploaded images and the processing pipeline.

### Ultralytics / YOLO

YOLO-based object detection is used to identify supported objects and hazards in the analyzed scene.

### JSON

JSON files provide lightweight local storage for prototype information such as:

```text
data/drone_profiles.json
data/mission_history.json
data/settings.json
```

### Git & GitHub

Git is used for version control and GitHub is used to store and manage the project's source code.

### Render

Render is used to deploy the Streamlit application and provide public web access.

---

## 🗂️ Project Structure

```text
safeland-ai/
│
├── app.py
├── requirements.txt
├── README.md
│
├── core/
│   ├── __init__.py
│   ├── drone_profiles.py
│   ├── history.py
│   ├── input_handler.py
│   ├── landing_engine.py
│   ├── scene_analysis.py
│   └── scoring.py
│
├── data/
│   ├── drone_profiles.json
│   ├── mission_history.json
│   └── settings.json
│
├── assets/
│   └── test_scene_*.jpg
│
└── .streamlit/
    └── config.toml
```

---

## 💻 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/thamaraiselvid14-design/AI-Drone-Landing-Zone-Safety.git
```

### 2. Open the project directory

```bash
cd AI-Drone-Landing-Zone-Safety/safeland-ai
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start SafeLand AI

```bash
streamlit run app.py
```

### 5. Open in browser

Streamlit normally starts the application at:

```text
http://localhost:8501
```

---

## 🌐 Live Application

SafeLand AI is deployed using Render.

**Live Demo:**

https://ai-drone-landing-zone-safety.onrender.com

---

## 🧪 Example Test Scenario

### Initial Scene

1. Capture or upload a relatively clear scene.
2. Run AI analysis.
3. Check candidate-zone rankings.
4. Note the recommended zone.

### Dynamic Obstacle Test

1. Capture another frame.
2. Introduce a visible obstacle/person into the previously recommended region.
3. Re-analyze the frame.
4. Check whether its safety score decreases.
5. Verify whether another zone becomes the recommendation.

### Unsafe Scene Test

1. Provide a densely cluttered scene.
2. Run the analysis.
3. Confirm that candidates below the safety threshold are rejected.
4. Verify the `NO SAFE LANDING ZONE` warning.
5. Enable Emergency Mode.
6. Verify that only a clearly labelled emergency candidate is provided.

---

## ⚠️ Safety Disclaimer

SafeLand AI is currently a **prototype and decision-support system**.

The application provides landing-zone recommendations based on the available visual input and configured analysis rules.

It does **not autonomously authorize a drone to land**.

The final landing decision remains with the:

- Human operator,
- UAV flight controller, or
- Authorized control system.

Real-world deployment would require additional validation and hardware information such as accurate camera calibration, altitude/depth sensing, UAV telemetry, environmental sensing, and safety-certified control integration.

---

## 🚀 Future Enhancements

Future development can include:

- Depth-camera / LiDAR integration
- GPS and altitude integration
- Real-time drone telemetry
- Improved terrain classification
- More advanced obstacle tracking
- Continuous real-time video analysis
- Edge-AI deployment onboard UAVs
- Weather and wind-condition integration
- Improved explainable AI
- Real-world flight testing

---

## 👩‍💻 Project

**SafeLand AI**  
AI-Assisted Drone Landing Zone Safety & Dynamic Re-Evaluation System

Built as an intelligent UAV landing decision-support prototype.
