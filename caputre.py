import requests
import time
from datetime import datetime
import os

# Configuration
MOBILE_CAMERA_URL = "http://192.168.0.116:8080/shot.jpg"  # From IP Webcam app
SERVER_URL = "http://127.0.0.1:5000/api/upload"             # Your Flask server
INTERVAL_SECONDS = 60                                        # 1 minute
SAVE_LOCALLY = True                                          # Backup saves
LOCAL_SAVE_DIR = "./captured_images"

def capture_and_upload():
    try:
        # 1. Capture image from mobile camera
        response = requests.get(MOBILE_CAMERA_URL, timeout=10)
        response.raise_for_status()  # Raise error if HTTP request failed

        # 2. Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}.jpg"

        # 3. Save locally (optional)
        if SAVE_LOCALLY:
            os.makedirs(LOCAL_SAVE_DIR, exist_ok=True)
            with open(os.path.join(LOCAL_SAVE_DIR, filename), 'wb') as f:
                f.write(response.content)
            print(f"Saved locally: {filename}")

        # 4. Send to Flask server
        files = {'image': (filename, response.content, 'image/jpeg')}
        server_response = requests.post(SERVER_URL, files=files)

        if server_response.status_code == 200:
            print(f"Successfully uploaded {filename}. Server response: {server_response.json()}")
        else:
            print(f"Upload failed. Status: {server_response.status_code}, Response: {server_response.text}")

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    print(f"Starting capture every {INTERVAL_SECONDS} seconds...")
    while True:
        capture_and_upload()
        time.sleep(INTERVAL_SECONDS)