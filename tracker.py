import cv2
import mediapipe as mp
import numpy as np
import time
import socket # Added for communication

# --- Model & Global Setup ---
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
from mediapipe.framework.formats import landmark_pb2 

# --- Emotion Interpretation Function (Facial) ---
def interpret_emotion(blendshape_scores):
    """
    Interprets emotion based on MediaPipe blendshape scores using
    a simplified logic (Happy, Smile, Surprise, Neutral).
    """    
    smile_val = (blendshape_scores.get('mouthSmileLeft', 0) + blendshape_scores.get('mouthSmileRight', 0)) / 2
    jaw_open_val = blendshape_scores.get('jawOpen', 0)
    teeth_show_val = (blendshape_scores.get('mouthUpperUpLeft', 0) + blendshape_scores.get('mouthUpperUpRight', 0)) / 2

    SMILE_THRESH = 0.3
    JAW_OPEN_FOR_SMILE = 0.2
    TEETH_SHOW_THRESH = 0.2
    SURPRISE_JAW_THRESH = 0.4

    if smile_val > SMILE_THRESH:
        if jaw_open_val > JAW_OPEN_FOR_SMILE or teeth_show_val > TEETH_SHOW_THRESH:
            return "Smile"
        else:
            return "Happy"

    if jaw_open_val > SURPRISE_JAW_THRESH:
        return "Surprise"

    return "Neutral"

# --- Gesture Detection Helper Functions ---
# MediaPipe Hand Landmark indices
# Tip indices: 4 (Thumb), 8 (Index), 12 (Middle), 16 (Ring), 20 (Pinky)
# PIP (knuckle) indices: 3 (Thumb), 6 (Index), 10 (Middle), 14 (Ring), 18 (Pinky)
#
# MediaPipe Face Landmark indices
# Mouth: 13 (Upper Lip), 14 (Lower Lip), 61 (Left Corner), 291 (Right Corner)
# Nose/Eyes: 6 (Top of nose bridge, between eyes)

def is_finger_open(landmarks, tip_idx, pip_idx):
    """Checks if a finger is open by comparing Y-coordinates."""
    # Screen coordinates: smaller Y is higher
    return landmarks[tip_idx].y < landmarks[pip_idx].y

def is_thumb_open(landmarks):
    """Checks if the thumb is open."""
    return landmarks[4].y < landmarks[3].y

def calc_distance(lm1, lm2):
    """Calculates the 2D normalized distance between two landmarks."""
    return ((lm1.x - lm2.x)**2 + (lm1.y - lm2.y)**2)**0.5

def check_middle_finger(hand_landmarks):
    """Checks for the middle finger gesture. (Relaxed)"""
    middle_open = is_finger_open(hand_landmarks, 12, 10)
    index_closed = not is_finger_open(hand_landmarks, 8, 6)
    ring_closed = not is_finger_open(hand_landmarks, 16, 14)
    
    return middle_open and index_closed and ring_closed

def check_hand_uwu(hand_landmarks):
    """Checks for the 'L' shape (thumb + index) for UwU. (Relaxed)"""
    index_open = is_finger_open(hand_landmarks, 8, 6)
    thumb_open = is_thumb_open(hand_landmarks)
    middle_closed = not is_finger_open(hand_landmarks, 12, 10)
    
    return index_open and thumb_open and middle_closed

def check_v_shape(hand_landmarks):
    """Checks for the 'V' (peace) sign."""
    index_open = is_finger_open(hand_landmarks, 8, 6)
    middle_open = is_finger_open(hand_landmarks, 12, 10)
    ring_closed = not is_finger_open(hand_landmarks, 16, 14)
    pinky_closed = not is_finger_open(hand_landmarks, 20, 18)
    thumb_closed = not is_thumb_open(hand_landmarks)
    
    return index_open and middle_open and ring_closed and pinky_closed and thumb_closed

def check_like_shape(hand_landmarks):
    """Checks for the 'Like' (thumbs up) sign."""
    thumb_open = is_thumb_open(hand_landmarks)
    index_closed = not is_finger_open(hand_landmarks, 8, 6)
    middle_closed = not is_finger_open(hand_landmarks, 12, 10)
    ring_closed = not is_finger_open(hand_landmarks, 16, 14)
    pinky_closed = not is_finger_open(hand_landmarks, 20, 18)
    
    return thumb_open and index_closed and middle_closed and ring_closed and pinky_closed

def check_pointing_finger(hand_landmarks):
    """Checks if only the index finger is pointing up."""
    index_open = is_finger_open(hand_landmarks, 8, 6)
    middle_closed = not is_finger_open(hand_landmarks, 12, 10)
    ring_closed = not is_finger_open(hand_landmarks, 16, 14)
    pinky_closed = not is_finger_open(hand_landmarks, 20, 18)
    
    return index_open and middle_closed and ring_closed and pinky_closed

def check_hand_open(hand_landmarks):
    """Checks if all fingers on a hand are open."""
    return (is_finger_open(hand_landmarks, 8, 6) and
            is_finger_open(hand_landmarks, 12, 10) and
            is_finger_open(hand_landmarks, 16, 14) and
            is_finger_open(hand_landmarks, 20, 18) and
            is_thumb_open(hand_landmarks))

def check_hand_at_face(hand_landmarks, face_box_norm):
    """Checks if the hand (wrist) is within the Y-bounds of the face."""
    wrist_y = hand_landmarks[0].y # Normalized Y of the wrist
    face_min_y = face_box_norm[1]
    face_max_y = face_box_norm[3]
    
    return face_min_y < wrist_y < face_max_y

#
# --- *** UPDATED GESTURE DETECTION LOGIC *** ---
#
def detect_priority_gestures(hand_result, face_emotion, face_landmarks, face_box_norm):
    """
    Checks for priority gestures. Returns the gesture name or None.
    'face_landmarks' is the full list of 478 points.
    'face_box_norm' is the (min_x, min_y, max_x, max_y) tuple.
    """
    if not hand_result.hand_landmarks:
        return None # No hands detected

    hands = hand_result.hand_landmarks
    num_hands = len(hands)

    # --- 1. Check for 1-Hand Gestures ---
    if num_hands == 1:
        hand = hands[0]
        
        # --- "Silent" gesture ---
        if check_pointing_finger(hand) and face_landmarks:
            tip = hand[8] # Index finger tip
            
            # Create a box from between the eyes down to the mouth
            y_min = face_landmarks[6].y  # Point between eyes
            y_max = face_landmarks[14].y # Point on lower lip
            x_min = face_landmarks[61].x # Left mouth corner
            x_max = face_landmarks[291].x # Right mouth corner
            
            if (x_min < tip.x < x_max) and (y_min < tip.y < y_max):
                return "Silent"
        # --- End of Silent check ---

        # Check other 1-hand gestures
        if check_v_shape(hand):
            return "V"
        if check_like_shape(hand):
            return "Like"
        if check_middle_finger(hand):
            return "Middle Finger"

    # --- 2. Check for 2-Hand Gestures ---
    if num_hands == 2:
        
        # --- CHANGED: Removed "Pray" check ---

        # Check for UwU
        hand1_uwu = check_hand_uwu(hands[0])
        hand2_uwu = check_hand_uwu(hands[1])
        if hand1_uwu and hand2_uwu:
            tip1 = hands[0][8] # Index finger tip
            tip2 = hands[1][8] # Index finger tip
            
            distance = calc_distance(tip1, tip2)
            vertically_aligned = abs(tip1.y - tip2.y) < 0.07
            
            if distance < 0.15 and vertically_aligned:
                return "UwU"
        
        # Check for Absolute Cinema
        if face_emotion == "Neutral" and face_box_norm:
            hand1_open = check_hand_open(hands[0])
            hand2_open = check_hand_open(hands[1])
            hand1_at_face = check_hand_at_face(hands[0], face_box_norm)
            hand2_at_face = check_hand_at_face(hands[1], face_box_norm)
            
            if hand1_open and hand2_open and hand1_at_face and hand2_at_face:
                return "Absolute Cinema"

        # Check for middle finger on EITHER hand (even in a 2-hand pose)
        for hand in hands:
            if check_middle_finger(hand):
                return "Middle Finger"

    # --- 3. No priority gesture found ---
    return None

# --- Main Program ---
def run_tracker():
    
    # --- 0. Connect to Display Server ---
    HOST = '127.0.0.1'
    PORT = 65432
    
    print("Attempting to connect to display server...")
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((HOST, PORT))
        print("Connected!")
    except Exception as e:
        print(f"Failed to connect to display server: {e}")
        print("Please run 'python display_emotion.py' in another terminal first.")
        return

    # --- 1. Initialize Landmarkers ---
    face_model_path = 'face_landmarker.task'
    hand_model_path = 'hand_landmarker.task'

    face_options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=face_model_path),
        running_mode=VisionRunningMode.IMAGE,
        output_face_blendshapes=True,
        num_faces=1)

    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=hand_model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2)

    with FaceLandmarker.create_from_options(face_options) as face_detector, \
         HandLandmarker.create_from_options(hand_options) as hand_detector:
        
        # --- 2. Start Webcam ---
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            s.close()
            return

        last_sent_emotion = ""
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            frame_height, frame_width, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # --- 4. Detect Face (for fallback emotion and face box) ---
            face_result = face_detector.detect(mp_image)
            face_emotion = "Neutral" # Default
            face_landmarks = None # Full list of face landmarks
            face_box_norm = None # Normalized face box for gesture logic
            blendshape_scores = {} # Dictionary of blendshape scores

            if face_result.face_blendshapes:
                blendshapes_list = face_result.face_blendshapes[0]
                # Create dictionary for easy access
                blendshape_scores = {c.category_name: c.score for c in blendshapes_list}
                face_emotion = interpret_emotion(blendshape_scores)
            
            if face_result.face_landmarks:
                face_landmarks = face_result.face_landmarks[0]
                # Get normalized coords for gesture logic
                x_coords = [lm.x for lm in face_landmarks]
                y_coords = [lm.y for lm in face_landmarks]
                face_box_norm = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
                
                # Get pixel coords for drawing
                min_x = int(face_box_norm[0] * frame_width) - 10
                max_x = int(face_box_norm[2] * frame_width) + 10
                min_y = int(face_box_norm[1] * frame_height) - 10
                max_y = int(face_box_norm[3] * frame_height) + 10
                cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)
            
            # --- 5. Detect Hands and Final Emotion ---
            hand_result = hand_detector.detect(mp_image)
            
            # Check for priority hand gestures
            priority_emotion = detect_priority_gestures(hand_result, face_emotion, face_landmarks, face_box_norm)
            
            if priority_emotion:
                final_emotion = priority_emotion
            else:
                # No hand gestures, fallback to face emotion
                final_emotion = face_emotion 

            # --- 5b. SEND EMOTION TO DISPLAY ---
            if final_emotion != last_sent_emotion:
                try:
                    s.sendall(final_emotion.encode('utf-8'))
                    last_sent_emotion = final_emotion
                except Exception as e:
                    print(f"Error sending data: {e}. Display window may be closed.")
                    break
            
            # Draw final emotion on tracker window
            cv2.putText(frame, 
                        f"Emotion: {final_emotion}", 
                        (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 0, 255), 2, cv2.LINE_AA)

            # --- 6. Draw Hand Landmarks ---
            if hand_result.hand_landmarks:
                for hand_landmarks_list in hand_result.hand_landmarks:
                    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                    for landmark in hand_landmarks_list:
                        hand_landmarks_proto.landmark.add(x=landmark.x, y=landmark.y, z=landmark.z)

                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks_proto,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))
            
            # --- 7. Show the Result ---
            cv2.imshow('Emotion and Hand Tracker', frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        # --- 8. Cleanup ---
        cap.release()
        cv2.destroyAllWindows()
        s.close() # Close the connection
        print("Tracker shut down.")

if __name__ == "__main__":
    run_tracker()