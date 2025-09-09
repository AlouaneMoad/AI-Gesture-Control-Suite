AI Gesture Control Suite
A sophisticated Human-Computer Interaction (HCI) application that allows you to control your computer's core functions using real-time hand gestures. Developed in Python, this project leverages the power of computer vision to create an intuitive, hands-free user experience.

This project was developed as an advanced exploration into HCI, demonstrating a robust and fluid alternative to traditional input devices.

Key Features
🖱️ Full Cursor Control: Smooth, responsive mouse movement by simply pointing your index finger.

🤏 Advanced Click & Drag: An intelligent system distinguishes between a quick "tap" pinch for a single click and a "held" pinch for seamless drag-and-drop.

✌️ Multi-Gesture Clicks: Perform double-clicks using a distinct and reliable finger combination.

🔊 Two-Handed Volume Control: Use the distance between your two index fingers to intuitively manage your system's master volume.

🖥️ Cross-Platform: Fully compatible with Windows, macOS, and Linux.

Gesture Guide
<img width="966" height="461" alt="image" src="https://github.com/user-attachments/assets/03297273-96d2-4a40-a0b3-831288fe63a2" />

Technologies Used
Python 3

OpenCV: For real-time camera feed processing.

MediaPipe: For high-fidelity hand and finger landmark detection.

pynput: For direct control of the mouse.

NumPy: For numerical operations and smoothing.

screeninfo: For multi-monitor screen dimension detection.

pycaw (Windows): For native system volume control.

Setup & Installation
Clone the repository:

git clone https://github.com/AlouaneMoad/AI-Gesture-Control-Suite.git
cd AI-Gesture-Control-Suite

Create a virtual environment and install dependencies:

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install the required packages
pip install opencv-python mediapipe pynput numpy screeninfo pycaw

Run the application:

python AI-Gesture-Control-Suite.py

Press 'q' with the camera window active to quit.

Acknowledgements
Developed using a combination of my own code and AI-generated code from Google's Gemini, which I directed and refined.
