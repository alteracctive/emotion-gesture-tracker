import cv2
import numpy as np
import socket
import os
from path_helper import resource_path # <-- IMPORT THE HELPER

# --- Configuration ---
HOST = '127.0.0.1'  # Localhost
PORT = 65432        # Port to listen on
IMG_SIZE = (400, 400)

VALID_EMOTIONS = ["Smile", "Happy", "Neutral", "Surprise",
                  "Middle Finger", "UwU", "Absolute Cinema",
                  "V", "Like", "Silent"]

# --- Helper Functions ---

def create_text_placeholder(emotion_name):
    img = np.full((IMG_SIZE[0], IMG_SIZE[1], 3), 255, dtype=np.uint8)
    text = emotion_name
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 3
    color = (0, 0, 0)
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = (IMG_SIZE[1] - text_width) // 2
    y = (IMG_SIZE[0] + text_height) // 2
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness)
    return img

# --- RENAMED TO main() ---
def main():
    print("Starting emotion display server...")
    
    image_cache = {}
    text_placeholder_cache = {}
    current_emotion = "Neutral"
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"Listening on {HOST}:{PORT}...")
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            conn.setblocking(False)
            
            while True:
                # 1. Check for new emotion data
                try:
                    data = conn.recv(1024)
                    if not data:
                        break
                    new_emotion = data.decode('utf-8')
                    if new_emotion in VALID_EMOTIONS:
                        current_emotion = new_emotion
                except BlockingIOError:
                    pass
                except Exception as e:
                    print(f"Connection error: {e}")
                    break

                display_frame = None

                # 2. Try to load image from cache
                if current_emotion in image_cache:
                    display_frame = image_cache[current_emotion]
                
                # 3. If not in cache, try to load from file
                else:
                    filename = current_emotion.lower().replace(" ", "")
                    
                    # --- USE resource_path() HERE ---
                    png_path = resource_path(os.path.join('pictures', f"{filename}.png"))
                    jpg_path = resource_path(os.path.join('pictures', f"{filename}.jpg"))

                    path_to_load = None
                    if os.path.exists(png_path):
                        path_to_load = png_path
                    elif os.path.exists(jpg_path):
                        path_to_load = jpg_path
                    
                    if path_to_load:
                        print(f"Loading image: {path_to_load}")
                        img = cv2.imread(path_to_load)
                        img = cv2.resize(img, IMG_SIZE)
                        image_cache[current_emotion] = img
                        display_frame = img
                    
                # 4. If image failed (None), use text placeholder
                if display_frame is None:
                    if current_emotion in text_placeholder_cache:
                        display_frame = text_placeholder_cache[current_emotion]
                    else:
                        print(f"Image not found for {current_emotion}. Using text placeholder.")
                        text_img = create_text_placeholder(current_emotion)
                        text_placeholder_cache[current_emotion] = text_img
                        display_frame = text_img

                # 5. Show the frame
                cv2.imshow('Emotion', display_frame)
                
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
                    
    cv2.destroyAllWindows()
    print("Display server shut down.")

# This block lets the script be runnable OR importable
if __name__ == "__main__":
    main()