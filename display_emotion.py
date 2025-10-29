import cv2
import numpy as np
import socket
import os
from path_helper import resource_path # Use helper for .exe compatibility

# --- Configuration ---
HOST = '127.0.0.1'  # Localhost
PORT = 65432        # Port to listen on
IMG_SIZE = (400, 400)

# Valid emotions list (make sure this matches tracker.py)
VALID_EMOTIONS = ["Smile", "Happy", "Neutral", "Surprise",
                  "Middle Finger", "UwU", "Absolute Cinema",
                  "V", "Like", "Silent", "Dislike"]

# --- Helper Functions ---

def create_text_placeholder(emotion_name):
    """Creates a white image with the emotion name as text."""
    # Create a blank white image
    img = np.full((IMG_SIZE[0], IMG_SIZE[1], 3), 255, dtype=np.uint8)
    
    # Text properties
    text = emotion_name
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 3
    color = (0, 0, 0) # Black
    
    # Get text size to center it
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = (IMG_SIZE[1] - text_width) // 2
    y = (IMG_SIZE[0] + text_height) // 2
    
    # Put the text on the image
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness)
    
    return img

# --- Main Server Loop ---
def main(): # Renamed from run_display_server for .exe build
    print("Starting emotion display server...")
    
    # Cache for loaded images
    image_cache = {}
    text_placeholder_cache = {} # New cache for text placeholders
    
    current_emotion = "Neutral"
    
    # Use context manager for socket handling
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # Allow reusing the address quickly after closing
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            s.bind((HOST, PORT))
        except OSError as e:
            print(f"ERROR: Could not bind to {HOST}:{PORT}. Port might be in use. {e}")
            return # Exit if binding fails
            
        s.listen()
        print(f"Listening on {HOST}:{PORT}...")
        
        try:
            conn, addr = s.accept()
        except OSError as e:
            print(f"ERROR accepting connection: {e}")
            return
            
        with conn:
            print(f"Connected by {addr}")
            conn.setblocking(False) # Don't wait for data
            
            while True:
                # 1. Check for new emotion data
                try:
                    data = conn.recv(1024)
                    if not data:
                        print("Connection closed by client.")
                        break # Connection closed
                    new_emotion = data.decode('utf-8')
                    if new_emotion in VALID_EMOTIONS:
                        current_emotion = new_emotion
                    else:
                        print(f"Warning: Received unknown emotion '{new_emotion}'")
                except BlockingIOError:
                    pass # No new data, just continue
                except ConnectionResetError:
                    print("Connection reset by client.")
                    break
                except Exception as e:
                    print(f"Connection error: {e}")
                    break

                display_frame = None

                # 2. Try to load image from cache
                if current_emotion in image_cache:
                    display_frame = image_cache[current_emotion]
                
                # 3. If not in cache, try to load from file
                else:
                    # Use .lower() and replace spaces for filenames
                    filename = current_emotion.lower().replace(" ", "")
                    
                    # Use resource_path for file loading in .exe
                    png_path = resource_path(os.path.join('pictures', f"{filename}.png"))
                    jpg_path = resource_path(os.path.join('pictures', f"{filename}.jpg"))

                    path_to_load = None
                    if os.path.exists(png_path):
                        path_to_load = png_path  # PNG takes priority
                    elif os.path.exists(jpg_path):
                        path_to_load = jpg_path
                    
                    if path_to_load:
                        print(f"Loading image: {path_to_load}") # Debug print
                        img = cv2.imread(path_to_load)
                        if img is not None: # Check if image loaded successfully
                           img = cv2.resize(img, IMG_SIZE)
                           image_cache[current_emotion] = img # Save to cache
                           display_frame = img
                        else:
                            print(f"Warning: Failed to load image {path_to_load}")
                    
                # 4. If image failed (None), use text placeholder
                if display_frame is None:
                    # Check cache first
                    if current_emotion in text_placeholder_cache:
                        display_frame = text_placeholder_cache[current_emotion]
                    else:
                        # Create placeholder, cache it, and set it
                        print(f"Image not found for {current_emotion}. Using text placeholder.")
                        text_img = create_text_placeholder(current_emotion)
                        text_placeholder_cache[current_emotion] = text_img
                        display_frame = text_img

                # 5. Show the frame
                cv2.imshow('Emotion', display_frame)
                
                # Exit on 'q'
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
                    
    cv2.destroyAllWindows()
    print("Display server shut down.")

if __name__ == "__main__":
    main() # Call main function