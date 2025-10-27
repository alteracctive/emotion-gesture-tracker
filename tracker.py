import cv2
import mediapipe as mp
import numpy as np
import time
import socket
from path_helper import resource_path # <-- IMPORT THE HELPER

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

#
# --- *** THIS FUNCTION IS FIXED *** ---
#
def interpret_emotion(blendshape_scores):
    """
    Interprets emotion based on MediaPipe blendshape scores using
    a simplified logic (Happy, Smile, Surprise, Neutral).
    """    
    # The 'blendshape_scores' variable is ALREADY the dictionary we need.
    # The redundant line that caused the error has been removed.
    
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
def is_finger_open(landmarks, tip_idx, pip_idx):
    return landmarks[tip_idx].y < landmarks[pip_idx].y
def is_thumb_open(landmarks):
    return landmarks[4].y < landmarks[3].y
def calc_distance(lm1, lm2):
    return ((lm1.x - lm2.x)**2 + (lm1.y - lm2.y)**2)**0.5
def check_middle_finger(hand_landmarks):
    middle_open = is_finger_open(hand_landmarks, 12, 10)
    index_closed = not is_finger_open(hand_landmarks, 8, 6)
    ring_closed = not is_finger_open(hand_landmarks, 16, 14)
    return middle_open and index_closed and ring_closed
def check_hand_uwu(hand_landmarks):
    index_open = is_finger_open(hand_landmarks, 8, 6)
    thumb_open = is_thumb_open(hand_landmarks)
    middle_closed = not is_finger_open(hand_landmarks, 12, 10)
    return index_open and thumb_open and middle_closed
def check_v_shape(hand_landmarks):
    index_open = is_finger_open(hand_landmarks, 8, 6)
    middle_open = is_finger_open(hand_landmarks, 12, 10)
    ring_closed = not is_finger_open(hand_landmarks, 16, 14)
    pinky_closed = not is_finger_open(hand_landmarks, 20, 18)
    thumb_closed = not is_thumb_open(hand_landmarks)
    return index_open and middle_open and ring_closed and pinky_closed and thumb_closed
def check_like_shape(hand_landmarks):
    thumb_open = is_thumb_open(hand_landmarks)
    index_closed = not is_finger_open(hand_landmarks, 8, 6)
    middle_closed = not is_finger_open(hand_landmarks, 12, 10)
    ring_closed = not is_finger_open(hand_landmarks, 16, 14)
    pinky_closed = not is_finger_open(hand_landmarks, 20, 18)
    return thumb_open and index_closed and middle_closed and ring_closed and pinky_closed
def check_pointing_finger(hand_landmarks):
    index_open = is_finger_open(hand_landmarks, 8, 6)
    middle_closed = not is_finger_open(hand_landmarks, 12, 10)
    ring_closed = not is_finger_open(hand_landmarks, 16, 14)
    pinky_closed = not is_finger_open(hand_landmarks, 20, 18)
    return index_open and middle_closed and ring_closed and pinky_closed
def check_hand_open(hand_landmarks):
    return (is_finger_open(hand_landmarks, 8, 6) and
            is_finger_open(hand_landmarks, 12, 10) and
            is_finger_open(hand_landmarks, 16, 14) and
            is_finger_open(hand_landmarks, 20, 18) and
            is_thumb_open(hand_landmarks))
def check_hand_at_face(hand_landmarks, face_box_norm):
    wrist_y = hand_landmarks[0].y
    face_min_y = face_box_norm[1]
    face_max_y = face_box_norm[3]
    return face_min_y < wrist_y < face_max_y
def detect_priority_gestures(hand_result, face_emotion, face_landmarks, face_box_norm):
    if not hand_result.hand_landmarks:
        return None
    hands = hand_result.hand_landmarks
    num_hands = len(hands)
    if num_hands == 1:
        hand = hands[0]
        if check_pointing_finger(hand) and face_landmarks:
            tip = hand[8]
            y_min = face_landmarks[6].y
            y_max = face_landmarks[14].y
            x_min = face_landmarks[61].x
            x_max = face_landmarks[291].x
            if (x_min < tip.x < x_max) and (y_min < tip.y < y_max):
                return "Silent"
        if check_v_shape(hand):
            return "V"
        if check_like_shape(hand):
            return "Like"
        if check_middle_finger(hand):
            return "Middle Finger"
    if num_hands == 2:
        hand1_uwu = check_hand_uwu(hands[0])
        hand2_uwu = check_hand_uwu(hands[1])
        if hand1_uwu and hand2_uwu:
            tip1 = hands[0][8]
            tip2 = hands[1][8]
            distance = calc_distance(tip1, tip2)
            vertically_aligned = abs(tip1.y - tip2.y) < 0.07
            if distance < 0.15 and vertically_aligned:
                return "UwU"
        if face_emotion == "Neutral" and face_box_norm:
            hand1_open = check_hand_open(hands[0])
            hand2_open = check_hand_open(hands[1])
            hand1_at_face = check_hand_at_face(hands[0], face_box_norm)
            hand2_at_face = check_hand_at_face(hands[1], face_box_norm)
            if hand1_open and hand2_open and hand1_at_face and hand2_at_face:
                return "Absolute Cinema"
        for hand in hands:
            if check_middle_finger(hand):
                return "Middle Finger"
    return None

# --- RENAMED TO main() ---
def main():
    
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
        print("Display server may not be running yet.")
        return

    # --- 1. Initialize Landmarkers ---
    # --- USE resource_path() HERE ---
    face_model_path = resource_path('face_landmarker.task')
    hand_model_path = resource_path('hand_landmarker.task')

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

            # --- 4. Detect Face ---
            face_result = face_detector.detect(mp_image)
            face_emotion = "Neutral"
            face_landmarks = None
            face_box_norm = None
            blendshape_scores = {}

            if face_result.face_blendshapes:
                blendshapes_list = face_result.face_blendshapes[0]
                blendshape_scores = {c.category_name: c.score for c in blendshapes_list}
                face_emotion = interpret_emotion(blendshape_scores)
            
            if face_result.face_landmarks:
                face_landmarks = face_result.face_landmarks[0]
                x_coords = [lm.x for lm in face_landmarks]
                y_coords = [lm.y for lm in face_landmarks]
                face_box_norm = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
                
                min_x = int(face_box_norm[0] * frame_width) - 10
                max_x = int(face_box_norm[2] * frame_width) + 10
                min_y = int(face_box_norm[1] * frame_height) - 10
                max_y = int(face_box_norm[3] * frame_height) + 10
                cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)
            
            # --- 5. Detect Hands and Final Emotion ---
            hand_result = hand_detector.detect(mp_image)
            priority_emotion = detect_priority_gestures(hand_result, face_emotion, face_landmarks, face_box_norm)
            
            if priority_emotion:
                final_emotion = priority_emotion
            else:
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
        s.close()
        print("Tracker shut down.")

# This block lets the script be runnable OR importable
if __name__ == "__main__":
    main()