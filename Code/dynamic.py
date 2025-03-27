import cv2
import mediapipe as mp
import numpy as np
import time
import tensorflow as tf
import os
from deepface import DeepFace
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel('ERROR')

# Initialize MediaPipe for Face & Hands
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Audio control setup
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

# MediaPipe Facial & Hand Landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_EAR = [234, 132]  
RIGHT_EAR = [454, 361]  
PALM_BASE = [0, 1, 5, 9, 13, 17]  # Palm base keypoints

# Brightness & Volume Variables
brightness_factor = 1.0
volume_level = volume.GetMasterVolumeLevelScalar()
eye_closed_start = None
hands_ear_start = None
current_emotion = "Neutral"  

def eye_aspect_ratio(eye_points):
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    return (A + B) / (2.0 * C)

def adjust_brightness(image, factor):
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)

def get_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def detect_features():
    global brightness_factor, volume_level, eye_closed_start, hands_ear_start, current_emotion

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return

    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Couldn't read frame from webcam.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_face = face_mesh.process(rgb_frame)
            results_hands = hands.process(rgb_frame)

            h, w, _ = frame.shape
            landmarks = None

            if results_face.multi_face_landmarks:
                for face_landmarks in results_face.multi_face_landmarks:
                    landmarks = np.array([(int(p.x * w), int(p.y * h)) for p in face_landmarks.landmark])
                    left_ear_pos = landmarks[LEFT_EAR]
                    right_ear_pos = landmarks[RIGHT_EAR]

                    # Eye Blink Detection
                    left_eye = landmarks[LEFT_EYE]
                    right_eye = landmarks[RIGHT_EYE]
                    left_ear = eye_aspect_ratio(left_eye)
                    right_ear = eye_aspect_ratio(right_eye)
                    ear = (left_ear + right_ear) / 2.0

                    if ear < 0.25:  # Eyes are closed
                        if eye_closed_start is None:
                            eye_closed_start = time.time()
                        elif time.time() - eye_closed_start >= 2:
                            brightness_factor -= 0.1
                            brightness_factor = max(0.2, brightness_factor)  # Limit minimum brightness
                            print(f"[INFO] Eyes closed! 🌙 Reducing brightness to {brightness_factor:.1f}")
                            eye_closed_start = time.time()
                    else:
                        eye_closed_start = None  # Reset timer if eyes open

            # Palm Covering Ears Detection for Volume Control
            if results_hands.multi_hand_landmarks and landmarks is not None:
                hands_near_ears = 0
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    palm_x = np.mean([int(hand_landmarks.landmark[p].x * w) for p in PALM_BASE])
                    palm_y = np.mean([int(hand_landmarks.landmark[p].y * h) for p in PALM_BASE])

                    # Check if palm is near both ears
                    if get_distance((palm_x, palm_y), left_ear_pos) < 80 or \
                       get_distance((palm_x, palm_y), right_ear_pos) < 80:
                        hands_near_ears += 1

                if hands_near_ears >= 1:  # If any palm is detected near an ear
                    if hands_ear_start is None:
                        hands_ear_start = time.time()
                    elif time.time() - hands_ear_start >= 2:
                        if volume_level > 0.2:
                            volume_level -= 0.1
                            volume_level = max(0.1, volume_level)  # Clamp volume
                            volume.SetMasterVolumeLevelScalar(volume_level, None)
                            print(f"[INFO] Hands covering ears! 🔇 Reducing volume to {volume_level:.1f}")
                            hands_ear_start = time.time()
                else:
                    hands_ear_start = None  # Reset if hands move away

            # Real-time Emotion Detection (Every Frame)
            try:
                emotion = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                current_emotion = emotion[0]['dominant_emotion']
            except:
                pass  # Ignore errors & keep the last detected emotion

            # Apply brightness effect
            frame = adjust_brightness(frame, brightness_factor)

            # Display emotion on screen
            cv2.putText(frame, f"Emotion: {current_emotion}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            frame_count += 1
            cv2.imshow("Real-Time Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting...")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Manually interrupted. Closing webcam...")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Webcam closed.")

detect_features()
