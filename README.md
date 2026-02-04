# 🖐️ AI Hand Mouse

A real-time **computer-vision based hand-tracking mouse controller** built with Python, OpenCV, and MediaPipe.  
This project transforms hand gestures into precise cursor movements and system actions, creating a touch-free spatial interaction experience inspired by modern AR/VR interfaces.

The system uses MediaPipe’s hand-landmark detection to map finger motion into smooth screen navigation while applying adaptive filtering, gesture recognition, and visual overlays for a professional interaction workflow.

---

## ✨ Features

- 🎯 **Precision Cursor Control** — Index finger tracking with adaptive smoothing
- 🔒 **Gesture Lock Mode** — Open palm pauses cursor movement to prevent accidental actions
- 🧲 **Hand-Attached Grid & Angle Detection** — Visual spatial reference with real-time rotation tracking
- 🖱️ **Gesture-Based Mouse Actions**
  - Pinch (Thumb + Index) → Left Click / Drag
  - Index + Ring → Right Click
- 🖐️ **Two-Finger Scroll Gesture**
- 📊 **Velocity Trail Visualization** for motion feedback
- ⚡ **Dynamic Stability Engine** — adjusts smoothing based on hand speed and rotation

---

## 🧠 How It Works

The application captures webcam frames and processes them using MediaPipe’s Hand Landmarker model.  
Landmark coordinates are mapped to screen space and refined through:

- Angle-aware motion smoothing
- Speed-adaptive interpolation
- Gesture distance thresholds
- Active control region calibration

This allows fluid, low-jitter control even during hand rotation or rapid movement.

---

## 🛠️ Tech Stack

- **Python**
- **OpenCV** — video capture & rendering
- **MediaPipe Tasks API** — hand landmark tracking
- **PyAutoGUI** — cursor & system control
- **NumPy** — interpolation & math operations
- **AppleScript (macOS)** — system volume integration

---

## 🎮 Gesture Controls

| Gesture                | Action              |
|------------------------|---------------------|
| Move Index Finger      | Cursor Movement     |
| Thumb + Index Pinch    | Left Click / Drag   |
| Index + Ring Pinch     | Right Click         |
| Index + Middle Close   | Scroll              |
| Open Palm              | Lock Mode           |
| Thumb + Pinky          | Volume Up           |
| Index + Pinky          | Volume Down         |

---

## 🚀 Use Cases

- Touchless computer interaction
- HCI / Computer Vision demos
- Accessibility experimentation
- Gesture-controlled UI prototypes
- Portfolio projects showcasing spatial interfaces

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/ai-gesture-hand-mouse.git
cd ai-gesture-hand-mouse
pip install -r requirements.txt
