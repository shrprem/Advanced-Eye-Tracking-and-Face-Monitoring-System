# Advanced Eye Tracking and Face Monitoring System

📌 **Project Overview**

This project uses a camera to watch a person's face and eyes in real time. It detects where the person is looking (left, right, or center), if their eyes close, and if their head moves too much. The goal is to help monitor attention and alert users if they get distracted.

📝 **Problem Statement**

Keeping track of where someone is looking and if they are paying attention can be hard. Current systems lack clear alerts and easy-to-use interfaces. This project solves that by providing real-time eye tracking with visual and sound alerts in an easy interface.

⚙️ **Approach**

- Use MediaPipe and OpenCV to detect facial landmarks.
- Analyze eye position to tell gaze direction.
- Detect if eyes are closed and head moves excessively.
- Show the camera feed live with drawn landmarks.
- Provide an intimidating, dark-themed user interface.
- Alert with beeps when attention is lost.

### User Interface

- **Camera Feed Panel**  
  Shows live video with facial landmarks highlighted.  
  ![Camera Feed](images/Screenshot 2025-09-14 123646.png)

- **Status Panel**  
  Displays current detection status like "Eyes Closed", "Looking Away", and threat levels.  
  ![Status Panel](images/ui_status_panel.png)

- **Control Buttons**  
  - *INITIATE SURVEILLANCE*: Starts eye and face tracking.  
  - *TERMINATE SURVEILLANCE*: Stops the camera and tracking.  
  - *CYCLE CAMERA INPUT*: Switches between available cameras.  
  ![Control Buttons](images/ui_controls.png)

📂 **Project Files**

| File                             | Description                                  |
|----------------------------------|----------------------------------------------|
| `advanced_surveillance_system.py`| Main Python program with GUI and detection.  |
| `requirements.txt`               | List of Python package dependencies.         |
| `images/`                        | Folder for screenshots used in README.       |

🚀 **How to Use**

1. Make sure Python 3.8 or higher is installed.
2. Install the required packages: pip install -r requirements.txt
3. Run the program with: python advanced_surveillance_system.py
