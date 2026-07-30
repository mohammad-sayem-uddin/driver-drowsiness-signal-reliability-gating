# Driver Drowsiness Detection System - Development Setup Guide

Welcome to the development environment for your **Research-Oriented Driver Drowsiness Detection System**. This workspace is pre-configured and structured specifically for macOS (compatible with Intel and Apple Silicon M1/M2/M3 Macs).

Follow this comprehensive, step-by-step guide to set up your environment, install libraries, and run diagnostics.

---

## 📂 Project Architecture

The directory has been pre-initialized with a clean, modular structure standard for research and production computer vision applications:

```text
Driver Drowsiness/
├── .vscode/
│   └── settings.json           # VS Code Python interpreter & linting configuration
├── requirements.txt            # Stable, version-controlled library requirements
├── test_webcam.py              # Environment diagnostic & hardware testing utility
├── README.md                   # Setup guide and architectural overview
└── src/                        # Core codebase package
    ├── __init__.py             # Package marker
    ├── main.py                 # Primary entry point (runs the monitoring HUD)
    ├── detector.py             # Math processor (EAR, MAR formulas, & state machine)
    └── utils/                  # Sub-utilities
        ├── __init__.py         # Sub-package marker
        ├── landmark_indices.py # High-precision MediaPipe Face Mesh index mappings
        └── audio_alert.py      # Audio system controller (Pygame loop & beep synthesizer)
```

---

## 🛠️ Step-by-Step Environment Setup

### Step 1: Open Terminal in the Project Folder
Open your macOS Terminal and navigate to the project directory:
```bash
cd "/Users/sayemuddin/Desktop/Driver Drowsiness"
```

### Step 2: Create a Python Virtual Environment
Creating a virtual environment (`venv`) prevents library version conflicts with other projects.
On macOS, Python 3 is typically run as `python3`:
```bash
python3 -m venv .venv
```

### Step 3: Activate the Virtual Environment
Activate the environment so that any packages you install are placed inside your local `.venv` rather than globally:
```bash
source .venv/bin/activate
```
*Note: Your terminal prompt should now be prefixed with `(.venv)`, indicating activation.*

### Step 4: Upgrade Pip
Ensure you have the latest package installer to avoid compile errors on macOS wheels:
```bash
pip install --upgrade pip
```

### Step 5: Install Dependencies
Install all required libraries using the pre-configured `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## 📦 Dependency Explanations

Here is why each package is essential to your research-oriented system:

| Library | Version | Purpose in Drowsiness Detection |
| :--- | :--- | :--- |
| **OpenCV (`opencv-python`)** | `>=4.8.0` | Handles webcam access, frame-by-frame video capture, converting color channels (BGR to RGB), and drawing the beautiful real-time HUD overlays on your screen. |
| **MediaPipe (`mediapipe`)** | `>=0.10.0` | Google's highly optimized machine learning framework. We utilize its **Face Mesh** pipeline to detect 478 highly accurate 3D landmarks in real time, bypassing the need to train a custom face landmarks model. |
| **NumPy (`numpy`)** | `>=1.24.0, <2.0` | A high-performance mathematical library. It scales facial landmark vectors, processes array operations for OpenCV drawings, and solves vector equations. |
| **Pygame (`pygame`)** | `>=2.5.0` | Provides a low-latency, multi-threaded audio mixing engine (`pygame.mixer`). Crucial for playing warning sirens/beeps on a background thread without lagging the computer vision frame rates. |

---

## 💻 VS Code Configuration

Your workspace includes a `.vscode/settings.json` file which pre-configures VS Code:
1. **Auto-select Virtual Environment**: VS Code will automatically detect and bind to the `.venv` Python interpreter when you open this directory.
2. **Auto-Format on Save**: Enabled using the modern Python **Black Formatter** integration.
3. **Optimized Exclusions**: Hides cache folders and system logs while keeping `.venv` accessible.

**Recommended VS Code Extensions to Install:**
- `ms-python.python` (Python Language Support)
- `ms-python.vscode-pylance` (High-performance type-checking & auto-imports)
- `ms-python.black-formatter` (Clean PEP8 code formatting)

---

## 🧪 Testing and Verification

We have provided **two separate scripts** to test your setup:

### Test 1: Full System Diagnostics (`test_webcam.py`)
Run the standalone diagnostic utility to verify library installations, camera stream resolution, audio output, and basic Face Mesh capabilities:
```bash
python3 test_webcam.py
```
* **Interactive Elements:**
  * Press **`s`** to play a dynamically synthesized alarm tone (verifies Pygame audio).
  * Press **`q`** or **`ESC`** to exit.

### Test 2: The Main Application Monitor (`src/main.py`)
Once diagnostics pass, run the fully realized Driver Drowsiness HUD:
```bash
python3 -m src.main
```
* **Features Demonstrated:**
  * **Real-time EAR & MAR calculation** overlaid directly on your screen.
  * **Automatic Alarm sound triggering** when eyes remain closed past the baseline frames.
  * **Yawn detection alert** when the mouth opens past the aspect ratio threshold.
  * **Interactive visual indicators** outlining your eyes and facial contours.

---

## ⚠️ Common macOS Installation Errors & Fixes

### 1. Apple Silicon (M1/M2/M3) MediaPipe Install Errors
* **Symptom:** `pip install mediapipe` fails or says `No matching distribution found`.
* **Reason:** Historically, MediaPipe did not distribute pre-compiled wheels for Apple Silicon for older Python versions.
* **Fixes:**
  * Ensure you are using **Python 3.9, 3.10, or 3.11** (MediaPipe has full arm64 wheels for these versions).
  * Make sure your pip is upgraded: `pip install --upgrade pip`
  * If using Python 3.12+ and experiencing issues, install the newer, verified wheels directly:
    ```bash
    pip install mediapipe --prefer-binary
    ```

### 2. Camera Access Denied (`cv2.VideoCapture` returns false or blank screen)
* **Symptom:** Terminal outputs error saying camera could not be opened, or python process hangs.
* **Reason:** macOS has tight security sandboxing. The terminal emulator (Terminal, iTerm2) or VS Code does not have permission to access your webcam.
* **Fixes:**
  1. Open **System Settings** on your Mac.
  2. Navigate to **Privacy & Security** -> **Camera**.
  3. Locate your terminal application (e.g., Terminal, iTerm) or **VS Code** in the list and toggle the switch to **ON**.
  4. Fully restart the terminal application and try running the script again.

### 3. Pygame Mixer Initialization Errors
* **Symptom:** Pygame outputs `pygame.error: No available audio device` or similar mixer error.
* **Reason:** Pygame cannot access your CoreAudio drivers or you have no default output audio device connected.
* **Fixes:**
  * Ensure your Mac is not set to a disconnected Bluetooth headset or output.
  * Our `audio_alert.py` wrapper catches this exception and handles it gracefully by turning off the sound output but letting the video and mathematical analysis run smoothly without crashing!

### 4. Numpy 2.0 Compatibility Issues
* **Symptom:** `AttributeError: module 'numpy' has no attribute '...'` upon importing cv2 or mediapipe.
* **Reason:** NumPy 2.0 (released in 2024) introduced breaking changes. Older pre-built wheels of OpenCV or MediaPipe might crash when run with NumPy 2.0.
* **Fixes:**
  * We pre-empted this by capping NumPy in your `requirements.txt` to `<2.0.0` (`numpy>=1.24.0,<2.0.0`).
  * If you accidentally upgraded numpy, force install the safe v1 series:
    ```bash
    pip install "numpy<2.0.0"
    ```

---

## 📈 Next Steps for Research & Expansion
Once your environment is verified, you are ready to expand this system into a publication-ready or production-ready detection tool. You can explore:
1. **EAR & MAR Custom Thresholding:** Collecting personal baseline ratios through a calibration phase.
2. **Head Pose Estimation:** Using landmarks to calculate roll, pitch, and yaw to detect distraction (gaze deviation from windshield).
3. **Infrared Camera Testing:** Testing the system in pitch-black night driving simulation conditions.
