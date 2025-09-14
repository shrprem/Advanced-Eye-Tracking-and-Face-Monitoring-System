
import cv2
import mediapipe as mp
import winsound
import math
import threading
import time
import tkinter as tk
from tkinter import ttk, Canvas, Frame, Label, Button, StringVar
from PIL import Image, ImageTk, ImageFilter, ImageEnhance
import numpy as np

class SurveillanceSystem:
    """Advanced Eye Tracking and Face Monitoring System"""

    def __init__(self):
        # MediaPipe initialization
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils

        # Eye tracking constants
        self.LEFT_EYE_CORNERS = [33, 133]
        self.RIGHT_EYE_CORNERS = [362, 263]
        self.LEFT_IRIS = 468
        self.RIGHT_IRIS = 473

        # System parameters
        self.movement_threshold = 0.008
        self.eye_open_threshold = 0.0012
        self.camera_index = 0

        # State variables
        self.is_monitoring = False
        self.capture_device = None
        self.previous_nose_position = None
        self.frame_buffer = None
        self.last_update_time = 0
        self.fps_counter = 0
        self.fps_display = 0

        # Thread management
        self.monitoring_thread = None
        self.ui_update_lock = threading.Lock()

        # Initialize UI
        self.setup_surveillance_interface()

    def calculate_distance(self, point_a, point_b):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point_a.x - point_b.x)**2 + (point_a.y - point_b.y)**2)

    def analyze_gaze_direction(self, face_landmarks, eye_side="left"):
        """Advanced gaze direction analysis with precision tracking"""
        try:
            if eye_side == "left":
                corner_left = face_landmarks.landmark[33]
                corner_right = face_landmarks.landmark[133]
                iris_center = face_landmarks.landmark[468]
            else:
                corner_left = face_landmarks.landmark[362]
                corner_right = face_landmarks.landmark[263]
                iris_center = face_landmarks.landmark[473]

            eye_width = self.calculate_distance(corner_left, corner_right)
            iris_position = self.calculate_distance(corner_left, iris_center)

            if eye_width == 0:
                return "TRACKING_ERROR", False

            gaze_ratio = iris_position / eye_width

            # Enhanced detection with finer thresholds
            if gaze_ratio < 0.25:
                return "EXTREME_LEFT", True
            elif gaze_ratio < 0.35:
                return "LEFT", False
            elif gaze_ratio > 0.75:
                return "EXTREME_RIGHT", True
            elif gaze_ratio > 0.65:
                return "RIGHT", False
            else:
                return "CENTER", False

        except (IndexError, ZeroDivisionError):
            return "DETECTION_FAILED", True

    def detect_eye_state(self, face_landmarks):
        """Determine if eyes are open or closed with high accuracy"""
        try:
            # Left eye vertical distance
            left_upper = face_landmarks.landmark[self.LEFT_EYE_CORNERS[0]]
            left_lower = face_landmarks.landmark[self.LEFT_EYE_CORNERS[1]]
            left_opening = self.calculate_distance(left_upper, left_lower)

            # Right eye vertical distance
            right_upper = face_landmarks.landmark[self.RIGHT_EYE_CORNERS[0]]
            right_lower = face_landmarks.landmark[self.RIGHT_EYE_CORNERS[1]]
            right_opening = self.calculate_distance(right_upper, right_lower)

            return (left_opening > self.eye_open_threshold and 
                   right_opening > self.eye_open_threshold)
        except IndexError:
            return False

    def monitor_head_movement(self, face_landmarks):
        """Track excessive head movement that might indicate distraction"""
        try:
            current_nose = face_landmarks.landmark[1]  # Nose tip

            if self.previous_nose_position is not None:
                movement = self.calculate_distance(current_nose, self.previous_nose_position)
                if movement > self.movement_threshold:
                    self.previous_nose_position = current_nose
                    return True, movement

            self.previous_nose_position = current_nose
            return False, 0.0
        except IndexError:
            return False, 0.0

    def trigger_alert(self, alert_type="standard", duration=200):
        """Multi-level alert system"""
        alert_frequencies = {
            "standard": (800, duration),
            "warning": (1200, duration * 2),
            "critical": (1500, duration * 3)
        }

        freq, dur = alert_frequencies.get(alert_type, (800, 200))
        try:
            winsound.Beep(freq, dur)
        except:
            pass  # Silent fail if sound system unavailable

    def process_video_frame(self):
        """Main video processing pipeline with optimized performance"""
        if not self.capture_device or not self.capture_device.isOpened():
            return

        with self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        ) as face_mesh:

            while self.is_monitoring and self.capture_device.isOpened():
                frame_captured, raw_frame = self.capture_device.read()
                if not frame_captured:
                    continue

                # Performance optimization - limit FPS
                current_time = time.time()
                if current_time - self.last_update_time < 1/30:  # 30 FPS limit
                    continue

                self.last_update_time = current_time
                self.fps_counter += 1

                # Process frame
                rgb_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
                detection_results = face_mesh.process(rgb_frame)

                # Analysis variables
                face_detected = False
                surveillance_status = "NO_TARGET"
                threat_level = "SECURE"
                gaze_analysis = "UNKNOWN"

                if detection_results.multi_face_landmarks:
                    for facial_landmarks in detection_results.multi_face_landmarks:
                        face_detected = True

                        # Draw facial mesh for intimidating effect
                        self.mp_drawing.draw_landmarks(
                            raw_frame,
                            facial_landmarks,
                            self.mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=self.mp_drawing.DrawingSpec(
                                color=(0, 255, 0), thickness=1, circle_radius=1
                            )
                        )

                        # Comprehensive analysis
                        eyes_open = self.detect_eye_state(facial_landmarks)
                        head_moved, movement_intensity = self.monitor_head_movement(facial_landmarks)

                        left_gaze, left_alert = self.analyze_gaze_direction(facial_landmarks, "left")
                        right_gaze, right_alert = self.analyze_gaze_direction(facial_landmarks, "right")

                        # Determine system status
                        if not eyes_open:
                            surveillance_status = "EYES_CLOSED"
                            threat_level = "ALERT"
                            self.trigger_alert("warning")
                        elif head_moved:
                            surveillance_status = "EXCESSIVE_MOVEMENT"
                            threat_level = "WARNING"
                            self.trigger_alert("standard")
                        elif left_alert or right_alert:
                            surveillance_status = "SUSPICIOUS_GAZE"
                            threat_level = "CAUTION"
                            self.trigger_alert("standard")
                        else:
                            surveillance_status = "TARGET_LOCKED"
                            threat_level = "SECURE"

                        gaze_analysis = f"L:{left_gaze} | R:{right_gaze}"

                # Store frame for UI update
                with self.ui_update_lock:
                    self.frame_buffer = rgb_frame.copy()

                # Update UI (less frequently to prevent flicker)
                if self.fps_counter % 3 == 0:  # Update UI every 3rd frame
                    self.root.after(0, self.update_surveillance_display, 
                                  face_detected, surveillance_status, threat_level, gaze_analysis)

                # FPS calculation
                if current_time - getattr(self, 'fps_start_time', current_time) >= 1.0:
                    self.fps_display = self.fps_counter
                    self.fps_counter = 0
                    self.fps_start_time = current_time

    def update_surveillance_display(self, face_detected, status, threat_level, gaze_info):
        """Update the UI with current surveillance data"""
        try:
            with self.ui_update_lock:
                if self.frame_buffer is not None:
                    # Process frame for display
                    display_frame = self.frame_buffer.copy()

                    # Add surveillance overlay
                    height, width = display_frame.shape[:2]

                    # Create intimidating overlay effect
                    if face_detected:
                        # Green tint for active tracking
                        overlay = np.zeros_like(display_frame)
                        overlay[:, :] = (0, 50, 0)  # Dark green
                        display_frame = cv2.addWeighted(display_frame, 0.85, overlay, 0.15, 0)

                    # Convert and resize for display
                    pil_image = Image.fromarray(display_frame)
                    pil_image = pil_image.resize((640, 480), Image.Resampling.LANCZOS)

                    # Apply visual effects based on threat level
                    if threat_level == "ALERT":
                        enhancer = ImageEnhance.Color(pil_image)
                        pil_image = enhancer.enhance(1.5)  # Increase saturation

                    photo = ImageTk.PhotoImage(image=pil_image)
                    self.camera_display.configure(image=photo)
                    self.camera_display.image = photo  # Keep reference

            # Update status displays
            status_colors = {
                "SECURE": "#00FF41",
                "CAUTION": "#FFD700", 
                "WARNING": "#FF8C00",
                "ALERT": "#FF0000"
            }

            self.status_var.set(f"STATUS: {status}")
            self.threat_var.set(f"THREAT LEVEL: {threat_level}")
            self.gaze_var.set(f"GAZE: {gaze_info}")
            self.fps_var.set(f"FPS: {self.fps_display}")

            # Update status colors
            color = status_colors.get(threat_level, "#FFFFFF")
            self.status_display.configure(fg=color)
            self.threat_display.configure(fg=color)

        except Exception as e:
            print(f"Display update error: {e}")

    def initiate_monitoring(self):
        """Start the surveillance system"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.capture_device = cv2.VideoCapture(self.camera_index)
            if self.capture_device.isOpened():
                self.monitoring_thread = threading.Thread(target=self.process_video_frame, daemon=True)
                self.monitoring_thread.start()
                self.control_status.set("SYSTEM ACTIVE")
                self.start_btn.configure(state='disabled')
                self.stop_btn.configure(state='normal')
            else:
                self.is_monitoring = False
                self.control_status.set("CAMERA ERROR")

    def terminate_monitoring(self):
        """Stop the surveillance system"""
        self.is_monitoring = False
        if self.capture_device:
            self.capture_device.release()
            self.capture_device = None
        self.control_status.set("SYSTEM OFFLINE")
        self.start_btn.configure(state='normal')
        self.stop_btn.configure(state='disabled')

    def cycle_camera_input(self):
        """Switch between available camera inputs"""
        was_monitoring = self.is_monitoring
        if was_monitoring:
            self.terminate_monitoring()

        # Test next camera
        for test_index in range(5):  # Test cameras 0-4
            test_camera = (self.camera_index + 1 + test_index) % 5
            test_cap = cv2.VideoCapture(test_camera)
            if test_cap.isOpened():
                test_cap.release()
                self.camera_index = test_camera
                self.camera_var.set(f"CAMERA: {self.camera_index}")
                break
            test_cap.release()

        if was_monitoring:
            self.initiate_monitoring()

    def setup_surveillance_interface(self):
        """Create the intimidating surveillance UI"""
        self.root = tk.Tk()
        self.root.title("◉ SURVEILLANCE SYSTEM v2.1 ◉")
        self.root.geometry("1200x800")
        self.root.configure(bg="#0a0a0a")
        self.root.resizable(False, False)

        # Make window look more professional
        self.root.attributes('-alpha', 0.95)  # Slight transparency

        # Variables for dynamic updates
        self.status_var = StringVar(value="STATUS: STANDBY")
        self.threat_var = StringVar(value="THREAT LEVEL: UNKNOWN")
        self.gaze_var = StringVar(value="GAZE: NOT DETECTED")
        self.fps_var = StringVar(value="FPS: 0")
        self.camera_var = StringVar(value="CAMERA: 0")
        self.control_status = StringVar(value="SYSTEM OFFLINE")

        # Header section
        header_frame = Frame(self.root, bg="#0a0a0a", height=80)
        header_frame.pack(fill="x", padx=10, pady=5)
        header_frame.pack_propagate(False)

        title_label = Label(
            header_frame, 
            text="◉ ADVANCED EYE TRACKING SURVEILLANCE ◉",
            font=("Courier New", 16, "bold"),
            fg="#00FF41",
            bg="#0a0a0a"
        )
        title_label.pack(pady=10)

        subtitle_label = Label(
            header_frame,
            text="UNAUTHORIZED ACCESS WILL BE DETECTED AND REPORTED",
            font=("Courier New", 10),
            fg="#FF4444",
            bg="#0a0a0a"
        )
        subtitle_label.pack()

        # Main content area
        main_frame = Frame(self.root, bg="#0a0a0a")
        main_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Left panel - Camera feed
        camera_frame = Frame(main_frame, bg="#1a1a1a", relief="ridge", bd=2)
        camera_frame.pack(side="left", fill="both", expand=True, padx=(0,5))

        feed_label = Label(
            camera_frame,
            text="◉ LIVE SURVEILLANCE FEED ◉",
            font=("Courier New", 12, "bold"),
            fg="#00FF41",
            bg="#1a1a1a"
        )
        feed_label.pack(pady=5)

        self.camera_display = Label(
            camera_frame,
            text="CAMERA OFFLINE\nCLICK START TO INITIATE",
            font=("Courier New", 14),
            fg="#666666",
            bg="#000000",
            width=50,
            height=20,
            relief="sunken",
            bd=3
        )
        self.camera_display.pack(padx=10, pady=10, fill="both", expand=True)

        # Right panel - Control and status
        control_frame = Frame(main_frame, bg="#1a1a1a", relief="ridge", bd=2, width=300)
        control_frame.pack(side="right", fill="y", padx=(5,0))
        control_frame.pack_propagate(False)

        # Status section
        status_label = Label(
            control_frame,
            text="◉ SYSTEM STATUS ◉",
            font=("Courier New", 12, "bold"),
            fg="#00FF41",
            bg="#1a1a1a"
        )
        status_label.pack(pady=(10,5))

        # Status displays
        self.status_display = Label(
            control_frame,
            textvariable=self.status_var,
            font=("Courier New", 11, "bold"),
            fg="#FFFFFF",
            bg="#1a1a1a",
            relief="ridge",
            bd=1,
            height=2
        )
        self.status_display.pack(fill="x", padx=10, pady=2)

        self.threat_display = Label(
            control_frame,
            textvariable=self.threat_var,
            font=("Courier New", 11, "bold"),
            fg="#FFFFFF", 
            bg="#1a1a1a",
            relief="ridge",
            bd=1,
            height=2
        )
        self.threat_display.pack(fill="x", padx=10, pady=2)

        gaze_display = Label(
            control_frame,
            textvariable=self.gaze_var,
            font=("Courier New", 9),
            fg="#CCCCCC",
            bg="#1a1a1a",
            relief="ridge", 
            bd=1,
            height=2
        )
        gaze_display.pack(fill="x", padx=10, pady=2)

        # System info
        fps_display = Label(
            control_frame,
            textvariable=self.fps_var,
            font=("Courier New", 9),
            fg="#888888",
            bg="#1a1a1a"
        )
        fps_display.pack(fill="x", padx=10, pady=2)

        camera_display = Label(
            control_frame,
            textvariable=self.camera_var,
            font=("Courier New", 9),
            fg="#888888",
            bg="#1a1a1a"
        )
        camera_display.pack(fill="x", padx=10, pady=2)

        # Control section
        control_label = Label(
            control_frame,
            text="◉ CONTROLS ◉",
            font=("Courier New", 12, "bold"),
            fg="#00FF41",
            bg="#1a1a1a"
        )
        control_label.pack(pady=(20,5))

        # Control buttons with intimidating styling
        button_style = {
            "font": ("Courier New", 10, "bold"),
            "width": 20,
            "height": 2,
            "relief": "raised",
            "bd": 3,
            "activebackground": "#333333"
        }

        self.start_btn = Button(
            control_frame,
            text="▶ INITIATE SURVEILLANCE",
            command=self.initiate_monitoring,
            bg="#2d5a2d",
            fg="#00FF41",
            **button_style
        )
        self.start_btn.pack(pady=5, padx=10, fill="x")

        self.stop_btn = Button(
            control_frame,
            text="⏹ TERMINATE SURVEILLANCE", 
            command=self.terminate_monitoring,
            bg="#5a2d2d",
            fg="#FF4444",
            state="disabled",
            **button_style
        )
        self.stop_btn.pack(pady=5, padx=10, fill="x")

        switch_btn = Button(
            control_frame,
            text="⚙ CYCLE CAMERA INPUT",
            command=self.cycle_camera_input,
            bg="#2d2d5a", 
            fg="#4444FF",
            **button_style
        )
        switch_btn.pack(pady=5, padx=10, fill="x")

        # System status
        status_system = Label(
            control_frame,
            textvariable=self.control_status,
            font=("Courier New", 10, "bold"),
            fg="#FF4444",
            bg="#1a1a1a",
            relief="sunken",
            bd=2,
            height=2
        )
        status_system.pack(fill="x", padx=10, pady=(20,10))

        # Footer
        footer_frame = Frame(self.root, bg="#0a0a0a", height=40)
        footer_frame.pack(fill="x", side="bottom")
        footer_frame.pack_propagate(False)

        footer_label = Label(
            footer_frame,
            text="⚠ WARNING: This system monitors and records all activity ⚠",
            font=("Courier New", 8),
            fg="#FF8800",
            bg="#0a0a0a"
        )
        footer_label.pack(pady=10)

    def execute_surveillance_protocol(self):
        """Launch the surveillance system"""
        try:
            self.root.protocol("WM_DELETE_WINDOW", self.shutdown_system)
            self.root.mainloop()
        except KeyboardInterrupt:
            self.shutdown_system()

    def shutdown_system(self):
        """Clean system shutdown"""
        self.terminate_monitoring()
        self.root.quit()
        self.root.destroy()

# System execution
if __name__ == "__main__":
    surveillance_system = SurveillanceSystem()
    surveillance_system.execute_surveillance_protocol()
