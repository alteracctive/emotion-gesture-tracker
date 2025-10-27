import multiprocessing
import time
import sys

# Import the main functions from your other scripts
import display_emotion
import tracker

def start_server():
    """Wrapper function to run the display server."""
    print("Attempting to start display server...")
    try:
        display_emotion.main()
    except Exception as e:
        print(f"Display Server Process Error: {e}")
        # This print might only be visible in logs or if run from cmd
        
def start_tracker():
    """Wrapper function to run the tracker."""
    print("Attempting to start tracker...")
    try:
        tracker.main()
    except Exception as e:
        print(f"Tracker Process Error: {e}")

if __name__ == "__main__":
    # This is ESSENTIAL for PyInstaller on Windows
    multiprocessing.freeze_support()

    print("--- Main Launcher Started ---")

    # 1. Start the display server in a separate process
    print("Launching server process...")
    server_process = multiprocessing.Process(target=start_server)
    server_process.start()

    # 2. Wait 3 seconds for the server to start up and bind to the port
    print("Waiting 3 seconds for server to initialize...")
    time.sleep(3)

    # 3. Start the tracker in a separate process
    print("Launching tracker process...")
    tracker_process = multiprocessing.Process(target=start_tracker)
    tracker_process.start()

    # 4. Wait for the processes to complete (they won't until user quits)
    server_process.join()
    tracker_process.join()

    print("--- Main Launcher Exiting ---")
    sys.exit()