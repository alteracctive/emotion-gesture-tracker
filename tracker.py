import cv2
import mediapipe as mp
import numpy as np
import time
import socket # Added for communication
from path_helper import resource_path # Use helper for .exe compatibility

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

# --- Global Toggles ---
draw_landmarks = True 
show_emotion_text = False # Start with text hidden
button_area_drawing = None 
button_area_emotion = None

# --- Mouse Callback Function ---
def handle_click(event, x, y, flags, param):
    global draw_landmarks, show_emotion_text, button_area_drawing, button_area_emotion
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check drawing button
        if button_area_drawing and button_area_drawing[0] < x < button_area_drawing[2] and button_area_drawing[1] < y < button_area_drawing[3]:
            draw_landmarks = not draw_landmarks
            print(f"Drawing landmarks toggled: {draw_landmarks}") 

        # Check emotion text button
        elif button_area_emotion and button_area_emotion[0] < x < button_area_emotion[2] and button_area_emotion[1] < y < button_area_emotion[3]:
            show_emotion_text = not show_emotion_text
            print(f"Show emotion text toggled: {show_emotion_text}")

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
def calc_distance(lm1, lm2):
    """Calculates the 2D normalized distance between two landmarks."""
    return ((lm1.x - lm2.x)**2 + (lm1.y - lm2.y)**2)**0.5

def is_finger_open(landmarks, tip_idx, pip_idx):
    """Checks if a finger is open by comparing distance from wrist."""
    wrist = landmarks[0]
    tip = landmarks[tip_idx]
    pip = landmarks[pip_idx]

    dist_wrist_tip = calc_distance(wrist, tip)
    dist_wrist_pip = calc_distance(wrist, pip)
    return dist_wrist_tip > (dist_wrist_pip * 1.1)

def is_thumb_open(landmarks):
    """Checks if the thumb is distinctly pointing upwards and extended outwards."""
    tip = landmarks[4]
    mcp = landmarks[2] 
    palm_center_approx = landmarks[9] 

    is_pointing_up = (mcp.y - tip.y) > 0.03 
    dist_palm_tip = calc_distance(palm_center_approx, tip)
    dist_palm_mcp = calc_distance(palm_center_approx, mcp)
    is_extended = dist_palm_tip > dist_palm_mcp

    return is_pointing_up and is_extended
    
def is_thumb_down(landmarks):
    """Checks if the thumb is distinctly pointing downwards."""
    tip = landmarks[4]
    mcp = landmarks[2] 
    is_pointing_down = (tip.y - mcp.y) > 0.03 
    return is_pointing_down

def check_palm_facing_camera(hand_landmarks, handedness):
    """Checks if the palm is oriented towards the camera, accounting for flip."""
    index_base_x = hand_landmarks[5].x
    pinky_base_x = hand_landmarks[17].x
    if handedness == 'Right': # Physical Right hand is on the LEFT side of the screen
        return index_base_x > pinky_base_x
    elif handedness == 'Left': # Physical Left hand is on the RIGHT side of the screen
        return index_base_x < pinky_base_x
    return False

def check_palm_facing_away(hand_landmarks, handedness):
    """Checks if the back of the hand is oriented towards the camera (palm away)."""
    return not check_palm_facing_camera(hand_landmarks, handedness)


def check_fingers_pointing_up(hand_landmarks):
    """Checks if index and middle fingers are generally pointing upwards."""
    wrist_y = hand_landmarks[0].y
    index_tip_y = hand_landmarks[8].y
    middle_tip_y = hand_landmarks[12].y
    threshold = 0.05
    return (wrist_y - index_tip_y > threshold) and (wrist_y - middle_tip_y > threshold)

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
    """Checks for the 'V' (peace) sign based only on finger open/closed state."""
    index_open = is_finger_open(hand_landmarks, 8, 6)
    middle_open = is_finger_open(hand_landmarks, 12, 10)
    ring_closed = not is_finger_open(hand_landmarks, 16, 14)
    pinky_closed = not is_finger_open(hand_landmarks, 20, 18)
    thumb_closed = not is_thumb_open(hand_landmarks) and not is_thumb_down(hand_landmarks)
    return index_open and middle_open and ring_closed and pinky_closed and thumb_closed

def check_like_shape(hand_landmarks):
    """Checks for the 'Like' (thumbs up) sign."""
    thumb_open = is_thumb_open(hand_landmarks)
    index_closed = not is_finger_open(hand_landmarks, 8, 6)
    middle_closed = not is_finger_open(hand_landmarks, 12, 10)
    ring_closed = not is_finger_open(hand_landmarks, 16, 14)
    pinky_closed = not is_finger_open(hand_landmarks, 20, 18)
    return thumb_open and index_closed and middle_closed and ring_closed and pinky_closed
    
def check_dislike_shape(hand_landmarks):
    """Checks for the 'Dislike' (thumbs down) sign."""
    thumb_down = is_thumb_down(hand_landmarks)
    index_closed = not is_finger_open(hand_landmarks, 8, 6)
    middle_closed = not is_finger_open(hand_landmarks, 12, 10)
    ring_closed = not is_finger_open(hand_landmarks, 16, 14)
    pinky_closed = not is_finger_open(hand_landmarks, 20, 18)
    return thumb_down and index_closed and middle_closed and ring_closed and pinky_closed

def check_pointing_finger(hand_landmarks):
    """Checks if only the index finger is pointing up."""
    index_open = is_finger_open(hand_landmarks, 8, 6)
    middle_closed = not is_finger_open(hand_landmarks, 12, 10)
    ring_closed = not is_finger_open(hand_landmarks, 16, 14)
    pinky_closed = not is_finger_open(hand_landmarks, 20, 18)
    return index_open and middle_closed and ring_closed and pinky_closed

def check_hand_open_fingers_only(hand_landmarks):
    """Checks if index, middle, ring, pinky fingers are open (thumb ignored)."""
    return (is_finger_open(hand_landmarks, 8, 6) and
            is_finger_open(hand_landmarks, 12, 10) and
            is_finger_open(hand_landmarks, 16, 14) and
            is_finger_open(hand_landmarks, 20, 18))

def detect_priority_gestures(hand_result, face_emotion, face_landmarks, face_box_norm, mouth_box_norm):
    """
    Checks for priority gestures. Returns the gesture name or None.
    'face_landmarks' is the full list of 478 points.
    'face_box_norm' is the (min_x, min_y, max_x, max_y) tuple for the whole face.
    'mouth_box_norm' is the (min_x, min_y, max_x, max_y) tuple for the mouth area.
    """
    if not hand_result.hand_landmarks:
        return None # No hands detected

    hands = hand_result.hand_landmarks
    handedness_list = hand_result.handedness # Get handedness info
    num_hands = len(hands)

    # --- 1. Check for 1-Hand Gestures ---
    if num_hands == 1:
        hand = hands[0]
        if check_pointing_finger(hand) and face_landmarks:
            tip = hand[8]
            if mouth_box_norm and \
               (mouth_box_norm[0] < tip.x < mouth_box_norm[2]) and \
               (mouth_box_norm[1] < tip.y < mouth_box_norm[3]):
                return "Silent"
        if check_v_shape(hand):
            return "V"
        if check_like_shape(hand): 
            return "Like"
        if check_dislike_shape(hand):
            return "Dislike"
        if check_middle_finger(hand): 
             if not is_finger_open(hand, 8, 6):
                 return "Middle Finger"

    # --- 2. Check for 2-Hand Gestures ---
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
        if face_box_norm: # Absolute Cinema requires face box
            hand1 = hands[0]
            hand2 = hands[1]
            if len(handedness_list) >= 2:
                handedness1 = handedness_list[0][0].category_name
                handedness2 = handedness_list[1][0].category_name
                hand1_ok = (check_hand_open_fingers_only(hand1) and 
                            check_palm_facing_camera(hand1, handedness1) and 
                            check_fingers_pointing_up(hand1))
                hand2_ok = (check_hand_open_fingers_only(hand2) and 
                            check_palm_facing_camera(hand2, handedness2) and 
                            check_fingers_pointing_up(hand2))
                face_y_min = face_box_norm[1]
                face_y_max = face_box_norm[3]
                wrist1_at_face_level = face_y_min < hand1[0].y < face_y_max
                wrist2_at_face_level = face_y_min < hand2[0].y < face_y_max
                if hand1_ok and hand2_ok and wrist1_at_face_level and wrist2_at_face_level:
                    return "Absolute Cinema"
        for hand in hands:
            if check_middle_finger(hand):
                 if not is_finger_open(hand, 8, 6):
                     return "Middle Finger"

    # --- 3. No priority gesture found ---
    return None

# --- Main Program ---
def main(): # Renamed from run_tracker for .exe build
    global button_area_drawing, button_area_emotion

    HOST = '127.0.0.1'
    PORT = 65432
    print("Attempting to connect to display server...")
    # Use context manager for socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            print("Connected!")

            # --- Initialize Landmarkers ---
            # Use resource_path for model loading in .exe
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

            # Use context manager for landmarkers
            with FaceLandmarker.create_from_options(face_options) as face_detector, \
                 HandLandmarker.create_from_options(hand_options) as hand_detector:
                
                # --- Start Webcam ---
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    print("Error: Could not open webcam.")
                    return # Exit if webcam fails
                    
                # --- Setup Window and Mouse Callback ---
                window_name = 'Emotion and Hand Tracker'
                cv2.namedWindow(window_name)
                cv2.setMouseCallback(window_name, handle_click)

                last_sent_emotion = ""
                
                # --- Main Loop ---
                while cap.isOpened():
                    success, frame = cap.read()
                    if not success:
                        print("Warning: Ignoring empty camera frame.")
                        continue

                    frame = cv2.flip(frame, 1)
                    frame_height, frame_width, _ = frame.shape
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                    # --- Define Button Areas (do this once) ---
                    if button_area_drawing is None:
                        btn_w, btn_h = 120, 30
                        margin = 10
                        draw_btn_x1 = frame_width - btn_w - margin
                        draw_btn_y1 = margin
                        draw_btn_x2 = frame_width - margin
                        draw_btn_y2 = margin + btn_h
                        button_area_drawing = (draw_btn_x1, draw_btn_y1, draw_btn_x2, draw_btn_y2)
                        emo_btn_x1 = draw_btn_x1
                        emo_btn_y1 = draw_btn_y2 + margin // 2
                        emo_btn_x2 = draw_btn_x2
                        emo_btn_y2 = emo_btn_y1 + btn_h
                        button_area_emotion = (emo_btn_x1, emo_btn_y1, emo_btn_x2, emo_btn_y2)

                    # --- Detect Face ---
                    face_result = face_detector.detect(mp_image)
                    face_emotion = "Neutral"
                    face_landmarks = None
                    face_box_norm = None
                    mouth_box_norm = None
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
                        padding = 0.03 # Padding for mouth box
                        mouth_x_min = face_landmarks[61].x - padding # Left corner
                        mouth_x_max = face_landmarks[291].x + padding # Right corner
                        mouth_y_min = face_landmarks[13].y - padding # Upper lip
                        mouth_y_max = face_landmarks[14].y + padding # Lower lip center
                        mouth_box_norm = (mouth_x_min, mouth_y_min, mouth_x_max, mouth_y_max)
                        
                    # --- Detect Hands and Final Emotion ---
                    hand_result = hand_detector.detect(mp_image)
                    priority_emotion = detect_priority_gestures(hand_result, face_emotion, face_landmarks, face_box_norm, mouth_box_norm)
                    if priority_emotion:
                        final_emotion = priority_emotion
                    else:
                        final_emotion = face_emotion
                        
                    # --- Draw Face & Mouth Box (Conditional) ---
                    if draw_landmarks and face_landmarks:
                         # Draw overall face box (Blue)
                         min_fx = int(face_box_norm[0] * frame_width) - 10
                         max_fx = int(face_box_norm[2] * frame_width) + 10
                         min_fy = int(face_box_norm[1] * frame_height) - 10
                         max_fy = int(face_box_norm[3] * frame_height) + 10
                         cv2.rectangle(frame, (min_fx, min_fy), (max_fx, max_fy), (255, 0, 0), 2)

                         # Draw mouth box (Green) 
                         if mouth_box_norm: # Check if mouth box was calculated
                            min_mx = int(mouth_box_norm[0] * frame_width)
                            max_mx = int(mouth_box_norm[2] * frame_width)
                            min_my = int(mouth_box_norm[1] * frame_height)
                            max_my = int(mouth_box_norm[3] * frame_height)
                            cv2.rectangle(frame, (min_mx, min_my), (max_mx, max_my), (0, 255, 0), 1)
                    # --- End Conditional Drawing ---
                        
                    if final_emotion != last_sent_emotion:
                        try:
                            s.sendall(final_emotion.encode('utf-8'))
                            last_sent_emotion = final_emotion
                        except Exception as e:
                            print(f"Error sending data: {e}. Display window may be closed.")
                            break # Exit loop if sending fails
                            
                    if show_emotion_text:
                        cv2.putText(frame,
                                    f"Emotion: {final_emotion}",
                                    (10, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 0, 255), 2, cv2.LINE_AA)
                                    
                    if draw_landmarks and hand_result.hand_landmarks:
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
                                
                    # --- Draw Buttons ---
                    def draw_button(area, text, color, font_scale=0.5):
                         if area:
                            btn_x1, btn_y1, btn_x2, btn_y2 = area
                            cv2.rectangle(frame, (btn_x1, btn_y1), (btn_x2, btn_y2), color, -1)
                            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                            text_x = btn_x1 + (btn_x2 - btn_x1 - text_width) // 2
                            text_y = btn_y1 + (btn_y2 - btn_y1 + text_height) // 2
                            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), 1, cv2.LINE_AA)
                            
                    draw_btn_text = "Drawing ON" if draw_landmarks else "Drawing OFF"
                    draw_btn_color = (0, 150, 0) if draw_landmarks else (0, 0, 150)
                    draw_button(button_area_drawing, draw_btn_text, draw_btn_color)
                    
                    emo_btn_text = "Text ON" if show_emotion_text else "Text OFF"
                    emo_btn_color = (0, 150, 0) if show_emotion_text else (0, 0, 150)
                    draw_button(button_area_emotion, emo_btn_text, emo_btn_color)
                    
                    # --- Show Frame ---
                    cv2.imshow(window_name, frame)
                    
                    if cv2.waitKey(5) & 0xFF == ord('q'):
                        break
                        
                # --- Cleanup ---
                cap.release()
                cv2.destroyAllWindows()
                print("Tracker shut down.")
                
    except ConnectionRefusedError:
        print(f"ERROR: Connection refused. Is the display server running on {HOST}:{PORT}?")
    except Exception as e:
        print(f"An unexpected error occurred in the tracker: {e}")
    finally:
        # Ensure windows are closed even if connection fails early
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main() # Call main function