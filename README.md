# Emotion & Gesture Tracker

This project uses your webcam to detect facial emotions and specific hand gestures in real-time. It displays the detected emotion on the camera feed and also shows a corresponding image or text placeholder in a separate window.

## How to Run

This project is designed to be run on a **Windows** machine with **Python 3.10** installed and added to your system's PATH.

1.  **Clone or Download:** Get all the project files (including `.py` scripts, `.bat` file, `.task` models, and the `pictures` folder) into a single directory.
2.  **Run the script:** Simply **double-click the `run.bat` file**.

That's it! The batch script will automatically:
* Check if a virtual environment (`.venv`) exists.
* Create a new one if it doesn't.
* Install all the required libraries from `requirements.txt`.
* Launch both the tracker and the display window in the background.

To stop the program, press **'q'** on both of the OpenCV windows that appear.

## Detected Emotions & Gestures

The tracker will prioritize hand gestures over facial emotions. If no gesture is detected, it will fall back to one of the facial emotions.

### Hand Gestures (Priority)

| Emotion | Trigger |
| :--- | :--- |
| **Silent** | (One Hand) Pointing finger held up in front of the center of the face (mouth to nose area). |
| **V** | (One Hand) A "peace" sign, with the index and middle fingers open and pointing up. |
| **Like** | (One Hand) A "thumbs up" gesture. |
| **Middle Finger** | (One or Two Hands) The middle finger is extended while the index and ring fingers are closed. |
| **UwU** | (Two Hands) Both hands make an "L" shape (thumb and index open, middle closed) and the two index fingertips touch. |
| **Absolute Cinema** | (Two Hands) Both palms are open and held up on either side of a **Neutral** face. |

### Facial Emotions (Fallback)

| Emotion | Trigger |
| :--- | :--- |
| **Smile** | A happy expression where the mouth is open or showing teeth. |
| **Happy** | A closed-mouth smile. |
| **Surprise** | The mouth is wide open (but not smiling). |
| **Neutral** | The default state when no other emotion or gesture is detected. |

## Customizing Images

To show your own images for each emotion, add a `.png` or `.jpg` file to the `pictures` folder. The file name must be **lowercase and have no spaces**.

* **Example:** For the "Absolute Cinema" emotion, create a file named `absolutecinema.png` or `absolutecinema.jpg`.
* PNG files will be prioritized over JPG files if both exist.
* If no image is found, a text placeholder will be shown.