# ISL Recognition - Setup Guide

## For ALL Team Members (Python 3.11, 3.14, 3.15)

### Option 1 - Auto Setup (Double click)
Just double-click `setup.bat` — it does everything!

### Option 2 - Manual Setup in Anaconda Prompt

```
conda create -n isl-env python=3.11
conda activate isl-env
pip install -r requirements.txt
python app.py
```

### In VS Code
1. Press Ctrl+Shift+P
2. Type: Python: Select Interpreter
3. Choose: isl-env

## Controls
| Key | Action |
|-----|--------|
| V   | Voice command |
| ESC | Quit |
| K   | Keypoint logging mode |
| H   | Point history logging mode |
| N   | Normal mode |

## ISL Gestures Supported
| Gesture | Meaning |
|---------|---------|
| Open Hand | Hello |
| Close/Fist | Thank You |
| Pointer | Yes |
| OK sign | Good |
| Stop | Stop |
| Clockwise move | Please |
| Counter Clockwise | Sorry |
| Move | Come Here |

## Features
- Real-time hand gesture detection
- Indian Sign Language meanings displayed
- Text-to-Speech speaks the meaning aloud
- Voice command with Google Speech Recognition
- FPS counter
