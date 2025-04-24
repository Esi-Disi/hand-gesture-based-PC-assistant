# 🖐️ Hand Gesture-Based PC Assistant

A real-time hand gesture recognition assistant that allows you to control your PC using intuitive hand movements captured via webcam.

## 🎯 Features
- **Media Control**: Play, pause, skip tracks, or go back using gestures.
- **System Volume Control**: Raise or lower volume by just moving your hand.
- **Cursor Control**: Move your mouse cursor and click using finger gestures.
- **Scrolling**: Scroll pages with a two-finger gesture.
- **Gesture Activation/Deactivation**: Use gestures to start or stop assistant functions.
- **Real-time Feedback**: Visual feedback for gesture recognition and system status.

## 🛠️ Tech Stack
- **Python**
- **MediaPipe**: Hand tracking and landmark detection
- **OpenCV**: Webcam capture and visualization
- **pyautogui**: Simulate mouse and keyboard events
- **pycaw**: Windows volume control API

## 📷 Gestures Overview
| Gesture | Action |
|--------|--------|
|  Thumbs Up | Activate System |
|  Thumbs Down | Deactivate System |
|  Fist | Pause Media |
|  Open Palm | Play/Pause Media |
|  Victory w/ Thumb | Next/Previous Track |
|  'L' Shape with your indx finger and thumb | Enable Cursor Mode |
|  Two Fingers | Enable Scroll Mode |
|  Three Fingers | Enable Volume Mode |
|  Middle Finger | Exit Assistant |

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Esi-Disi/hand-gesture-based-pc-assistant.git
cd hand-gesture-pc-assistant
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. 3. Run the Assistant
```bash
python assistant.py
```
