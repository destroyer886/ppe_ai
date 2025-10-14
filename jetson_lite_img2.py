import cv2
import torch
from ultralytics import YOLO
import time
import os
import numpy as np
import yaml
import os
import sys
import subprocess
import time
import fcntl
import socket
import requests
import json
# Configure CUDA for OpenCV
cv2.cuda.setDevice(0)

with open("/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Enable CUDA optimization for PyTorch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Load the YOLO models
LOCK_FILE = "/tmp/ai_rtsp.lock"
REPO_URL = "https://github.com/destroyer886/ppe_ai"
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
BRANCH = "main"

def single_instance_lock():
    """Prevents multiple instances of. this script."""
    global lock_fd
    lock_fd = open(LOCK_FILE, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_fd.write(str(os.getpid()))
        lock_fd.flush()
    except BlockingIOError:
        print("⚠️ Another instance is already running. Exiting...")
        sys.exit(0)

import os
import subprocess
import shutil

LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))

import requests
import json

def send_log(server_name, status, message=None):
    """
    Sends a log entry to the Jetson log server API.
    
    Args:
        server_name (str): Name of your server (e.g. "Jetson-Orin").
        status (str): Status string ("starting", "running", "error", "stopped").
        message (str, optional): Additional message or info.
    """

    url = "https://jetson-log.vercel.app/api/logs"
    headers = {"Content-Type": "application/json"}
    data = {
        "serverName": server_name,
        "status": status,
        "message": message
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        if response.status_code == 201:
            print("✅ Log sent successfully.")
        else:
            print(f"⚠️ Server responded with status {response.status_code}: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error sending log: {e}")


def get_commit_hash():
    """Returns the current git commit hash, or None if not a git repo. Installs git if missing."""

    # Check if git exists
    if shutil.which("git") is None:
        print("⚠️ Git not found. Trying to install without update...")
        try:
            subprocess.run(
                ["sudo", "apt-get", "install", "git", "-y"],
                check=True
            )
            print("✅ Git installed successfully.")
        except subprocess.CalledProcessError:
            print("❌ Direct install failed. Trying with apt-get update...")
            try:
                subprocess.run(["sudo", "apt-get", "update", "-y"], check=True)
                subprocess.run(["sudo", "apt-get", "install", "git", "-y"], check=True)
                print("✅ Git installed after updating package list.")
            except subprocess.CalledProcessError:
                print("❌ Failed to install Git even after update.")
                return None

    git_dir = os.path.join(LOCAL_DIR, ".git")
    if not os.path.exists(git_dir):
        return None

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=LOCAL_DIR,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None

    
send_log("AI", "starting", "Booting AI model")
def internet_available(host="8.8.8.8", port=53, timeout=5, retries=3):
    """Check if internet is available (with retries)."""
    for attempt in range(retries):
        try:
            socket.setdefaulttimeout(timeout)
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((host, port))
            s.close()
            return True
        except Exception:
            print(f"🌐 Internet check failed (attempt {attempt + 1}/{retries})...")
            time.sleep(2)
    return False


def update_repo():
    """Pull latest code safely (replace repo files only, not entire folder)."""
    if not internet_available():
        print("🌐 No internet connection. Running existing code...")
        return False

    git_dir = os.path.join(LOCAL_DIR, ".git")

    # If repo exists -> normal pull
    if os.path.exists(git_dir):
        try:
            print("🔄 Checking for updates from GitHub...")
            subprocess.run(["git", "fetch", "origin", BRANCH], cwd=LOCAL_DIR, check=True)

            new_commit = subprocess.run(
                ["git", "rev-parse", f"origin/{BRANCH}"],
                cwd=LOCAL_DIR,
                capture_output=True,
                text=True,
                check=True
            ).stdout.strip()

            old_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=LOCAL_DIR,
                capture_output=True,
                text=True,
                check=True
            ).stdout.strip()

            if new_commit != old_commit:
                print("✅ Update found! Resetting repo to latest commit...")
                subprocess.run(["git", "reset", "--hard", f"origin/{BRANCH}"], cwd=LOCAL_DIR, check=True)
                return True
            else:
                print("✅ Code is already up to date.")
                return False

        except subprocess.CalledProcessError as e:
            print(f"❌ Git update failed ({e}). Running existing code.")
            return False

    # If .git folder doesn’t exist, reinitialize instead of cloning
    else:
        print("⚙️ Initializing git repo in existing directory...")
        try:
            subprocess.run(["git", "init"], cwd=LOCAL_DIR, check=True)
            subprocess.run(["git", "remote", "add", "origin", REPO_URL], cwd=LOCAL_DIR, check=True)
            subprocess.run(["git", "fetch", "origin", BRANCH], cwd=LOCAL_DIR, check=True)
            subprocess.run(["git", "checkout", "-f", BRANCH], cwd=LOCAL_DIR, check=True)
            subprocess.run(["git", "reset", "--hard", f"origin/{BRANCH}"], cwd=LOCAL_DIR, check=True)
            print("✅ Repository initialized and synced.")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to initialize and sync repository: {e}")
            return False



model = YOLO("ppev5.pt").to('cuda')  
personModel = YOLO("yolov5su.pt").to('cuda')
VIDEO_SOURCE = config["video"]["source"]
PER_FRAME = 10
TIME_SLEEP = 1
        # Person tracking dictionary {id: {position, last_seen_time, missing_ppe, alert_sent}}
person_tracker = {}

# Movement thresholds
MOVEMENT_THRESHOLD = 100  # Maximum pixel distance to consider the same person
MAX_TRACKING_TIME = 3  # Maximum time in seconds to track a person who disappeared

# Create a directory to save the cropped images (if it doesn't already exist)
output_dir = 'cropped_images'
os.makedirs(output_dir, exist_ok=True)
single_instance_lock()
updated = update_repo()

if updated:
        print("♻️ Restarting with new code...")
        python = sys.executable
        os.execl(python, python, *sys.argv)

# Save cropped image function
def save_cropped_image(cropped_person, person_id, current_time):
    """Save the cropped image of a person without PPE."""
    filename = os.path.join(output_dir, f"{person_id}.jpg")
    cv2.imwrite(filename, cropped_person)
    print(f"⚠️ Image saved for Person ID {person_id} at {filename}")

# Email cooldown to prevent spam (in seconds)
EMAIL_COOLDOWN = 10  # 5 minutes
def verify_vest_color(crop):
    """Re-verify vest presence using HSV color coding (orange/red/lime ranges)."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Define HSV ranges for safety vests
    # Orange
    lower_orange = np.array([5, 100, 100])
    upper_orange = np.array([25, 255, 255])

    # Red (two ranges)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])

    # Lime / Yellow-Green (fluorescent)
    lower_lime = np.array([25, 100, 100])
    upper_lime = np.array([45, 255, 255])

    # Create masks
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    mask_red1   = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2   = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_lime   = cv2.inRange(hsv, lower_lime, upper_lime)

    # Combine all masks
    mask = mask_orange | mask_red1 | mask_red2 | mask_lime

    # Ratio of vest-colored pixels
    ratio = cv2.countNonZero(mask) / (crop.shape[0] * crop.shape[1] + 1e-6)

    # If enough vest color is found → treat as vest present
    return ratio > 0.02   # threshold: 5% pixels



def calculate_distance(pos1, pos2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def clean_old_tracks(current_time):
    """Remove tracks that haven't been seen for too long."""
    to_remove = []
    for person_id, data in person_tracker.items():
        if current_time - data['last_seen_time'] > MAX_TRACKING_TIME:
            to_remove.append(person_id)
    
    for person_id in to_remove:
        print(f"🚶 Person ID {person_id} hasn't been seen for {MAX_TRACKING_TIME}s. Removing from tracking.")
        del person_tracker[person_id]

def process_frame(frame, current_time):
    """Process a single frame for person and PPE detection with tracking."""
    # Process with GPU-optimized models in batches
    with torch.no_grad():  # Disable gradient tracking for inference
        # Detect persons
        person_results = personModel(frame)
        
        # Extract persons
        persons = []
        for result in person_results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0]) 
                confidence = float(box.conf[0])
                cls = int(box.cls[0])
                label = result.names[cls]
                
                if label == "person" and confidence > 0.5:  # Only track high-confidence detections
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    persons.append({
                        'bbox': (x1, y1, x2, y2),
                        'center': (center_x, center_y),
                        'confidence': confidence
                    })
        
        # Track persons based on position
        current_ids = set()
        for person in persons:
            x1, y1, x2, y2 = person['bbox']
            h, w, _ = frame.shape
            center = person['center']
            pad = 100
            bx1 = max(0, x1 - pad)
            by1 = max(0, y1 - pad)
            bx2 = min(w, x2 + pad)
            by2 = min(h, y2 + pad)
            
            # Try to match with existing tracks
            best_match = None
            min_distance = float('inf')
            
            for person_id, data in person_tracker.items():
                distance = calculate_distance(center, data['position'])
                if distance < MOVEMENT_THRESHOLD and distance < min_distance:
                    min_distance = distance
                    best_match = person_id
            
            # Create new track or update existing
            if best_match is not None:
                person_id = best_match
                # Update position
                person_tracker[person_id]['position'] = center
                person_tracker[person_id]['last_seen_time'] = current_time
                current_ids.add(person_id)
            else:
                # Create new track
                new_id = len(person_tracker) + 1
                person_tracker[new_id] = {
                    'position': center,
                    'last_seen_time': current_time,
                    'missing_ppe': False,
                    'alert_sent': False,
                    'last_email_time': 0,
                    'violation_start_time': 0
                }
                person_id = new_id
                current_ids.add(person_id)
            
            # Check for PPE
            cropped_person = frame[y1:y2, x1:x2]
           

            if cropped_person.size == 0:  # Skip empty crops
                continue
                
            ppe_results = model(cropped_person)
            
            missing_ppe = False
            missing_ppe_label = ""
            missing_ppe_labels = []
            for ppe_result in ppe_results:
                for ppe_box in ppe_result.boxes:
                    px1, py1, px2, py2 = map(int, ppe_box.xyxy[0])
                    ppe_cls = int(ppe_box.cls[0])
                    ppe_label = ppe_result.names[ppe_cls]
                    ppe_conf = float(ppe_box.conf[0])
                    
                    if ppe_label == "NO-Safety Vest" or ppe_label == "NO-Hardhat" or ppe_label == "No-Shoes":
                        if ppe_label == "NO-Safety Vest":
                            vest_crop = cropped_person[max(0, py1):min(cropped_person.shape[0], py2),
                               max(0, px1):min(cropped_person.shape[1], px2)]
                            if vest_crop.size > 0 and not verify_vest_color(vest_crop):
                                missing_ppe = True 
                                missing_ppe_labels.append(ppe_label)
                                print(person_id,"with id,no vest")
                                continue

                      
                        if ppe_label == "NO-Hardhat":
                            missing_ppe_labels.append("No Helmet")
                            missing_ppe = True
                        if ppe_label == "No-Shoes":  
                          missing_ppe_labels.append(ppe_label)
                          missing_ppe = True
                              
                        print(person_id,"with id")
                        continue
                        
            
            # Update PPE status
            missing_ppe_label = ", ".join(missing_ppe_labels)
            if missing_ppe and missing_ppe_labels:
                if not person_tracker[person_id]['missing_ppe']:
                    person_tracker[person_id]['missing_ppe'] = True
                    person_tracker[person_id]['violation_start_time'] = current_time
                
                # Check if violation has lasted more than 5 seconds
                violation_duration = current_time - person_tracker[person_id]['violation_start_time']
                
                if missing_ppe:
                     big_cropped_person = frame[by1:by2, bx1:bx2]
        
                     cv2.rectangle(big_cropped_person, 
                       (x1 - bx1, y1 - by1), 
                       (x2 - bx1, y2 - by1), 
                       (0, 255, 255), 1)
                    
                    # Save the cropped image
                    
                    save_cropped_image(big_cropped_person, f"{missing_ppe_label}_{int(current_time)}", current_time)
                    
                    person_tracker[person_id]['alert_sent'] = True
                    person_tracker[person_id]['last_email_time'] = current_time
                    print(f"⚠️ PPE violation detected for Person ID {person_id} - Image saved")
            else:
                # Reset violation tracking if PPE is now worn
                person_tracker[person_id]['missing_ppe'] = False
                person_tracker[person_id]['violation_start_time'] = 0
                
                # Reset alert status after some time with PPE on
                if current_time - person_tracker[person_id].get('last_email_time', 0) > 60:
                    person_tracker[person_id]['alert_sent'] = False
            
            # Draw bounding box
            # color = (0, 0, 255) if missing_ppe else (0, 255, 0)
            # cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # label_text = f"ID:{person_id} {'NO-PPE' if missing_ppe else 'OK'}"
            
            # cv2.putText(frame, missing_ppe_label, (x1, y1 - 10), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Clean up old tracks
        clean_old_tracks(current_time)
                
    return frame

# Sample rate to reduce processing load - increased to every 20th frame for safety
SAMPLE_RATE = PER_FRAME  # Process only every 20th frame
    

def main():
    # Initialize video capture (RTSP stream)
    #cap = cv2.VideoCapture('rtsp://admin:Admin!123@192.168.1.36:554/live1s1.sdp', cv2.CAP_FFMPEG)
    cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_FFMPEG)
    # For testing with local file:
   #  cap = cv2.VideoCapture('./Demo2e.mp4')
    send_log("AI", "started", "Booting AI model")
    
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        send_log("AI", "Error", "Camera not opened")
        
        
    
    frame_count = 0
    last_processed_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            cap =  cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_FFMPEG)
            continue
        
        frame_count += 1
        
        # Process only every 20th frame to significantly reduce CPU/GPU load
        if frame_count % SAMPLE_RATE != 0:
            continue
        
        current_time = time.time()
        processing_delay = current_time - last_processed_time
        
        # Log processing rate for monitoring
        fps = 1.0 / (processing_delay / SAMPLE_RATE)
        print(f"Processing frame {frame_count} | Effective FPS: {fps:.2f} | Processing every {SAMPLE_RATE} frames")
        
        # Process frame
        frame = cv2.resize(frame,(640,640))
        #frame = frame.astype(np.float16) / 255.0
        #frame = torch.from_numpy(frame).permute(2,0,1).unsqueeze(0).to('cuda').half()
        cap.release()
        processed_frame = process_frame(frame, current_time)
        last_processed_time = time.time()
        print("pausing capturing")
        time.sleep(TIME_SLEEP)
        print("start from fresh")

        cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_FFMPEG)

        
        # Resize for display (optional)
        # display_frame = cv2.resize(processed_frame, (620, 480))
        # cv2.imshow("PPE Detection", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

