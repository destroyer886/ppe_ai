# -*- coding:utf-8 -*-
import os
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import sys
import subprocess
import time
import fcntl
import socket
# ============================================================
# Email Configuration
# ============================================================
sender_email = "hexahrplantppe@gmail.com"
sender_password = "uisq nprg apxv apnn"
subject = "🚨 PPE Violation Alert - Missing PPE Detected,Updated"

output_dir = 'cropped_images'
email_delay = 2  # seconds delay between multiple recipients


LOCK_FILE = "/tmp/send_mail.lock"
REPO_URL = "https://github.com/destroyer886/ppe_ai"
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
BRANCH = "main"

def single_instance_lock():
    """Prevents multiple instances of this script."""
    global lock_fd
    lock_fd = open(LOCK_FILE, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_fd.write(str(os.getpid()))
        lock_fd.flush()
    except BlockingIOError:
        print("⚠️ Another instance is already running. Exiting...")
        sys.exit(0)

def get_commit_hash():
    """Returns the current git commit hash, or None if not a git repo."""
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
    """Pull latest code and return True if updated, else False."""
    if not internet_available():
        print("🌐 No internet connection. Running existing code...")
        return False

    old_commit = get_commit_hash()
    if not old_commit:
        print("📦 Cloning repository...")
        try:
            subprocess.run(["git", "clone", REPO_URL, LOCAL_DIR], check=True)
            return True
        except subprocess.CalledProcessError:
            print("❌ Git clone failed — running existing code.")
            return False

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

        if new_commit != old_commit:
            print("✅ Update found! Pulling latest code...")
            subprocess.run(["git", "reset", "--hard", f"origin/{BRANCH}"], cwd=LOCAL_DIR, check=True)
            return True
        else:
            print("✅ Code is up to date.")
            return False

    except subprocess.CalledProcessError as e:
        print(f"❌ Git update failed ({e}). Running existing code.")
        return False


single_instance_lock()
updated = update_repo()

if updated:
        print("♻️ Restarting with new code...")
        python = sys.executable
        os.execl(python, python, *sys.argv)
# ============================================================
# Email Sending Function
# ============================================================
def send_email(image_data, filename, recipient_email, reason):
    msg = MIMEMultipart("related")
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject

    # HTML Email Body
    html_body = f"""
    <html>
      <head>
        <style>
          body {{
            font-family: Arial, sans-serif;
            background-color: #f4f6f8;
            margin: 0;
            padding: 20px;
          }}
          .container {{
            background: #fff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
          }}
          .header {{
            font-size: 18px;
            font-weight: bold;
            color: #c0392b;
          }}
          .reason {{
            font-size: 16px;
            margin-top: 10px;
            color: #333;
          }}
          .footer {{
            margin-top: 20px;
            font-size: 12px;
            color: #777;
          }}
          img {{
            max-width: 100%;
            height: auto;
            border: 2px solid #ddd;
            border-radius: 6px;
            margin-top: 15px;
            cursor: pointer;
          }}
        </style>
      </head>
      <body>
        <div class="container">
          <div class="header">🚨 PPE Violation Alert</div>
          <div class="reason">
            A person was detected without PPE.<br>
            <b>Reason:</b> {reason}
          </div>
          <div>
            <p>Please find the detected image below (click to view full size):</p>
            <a href="cid:image1">
              <img src="cid:image1" alt="PPE Violation Image">
            </a>
          </div>
          <div class="footer">
            This is an automated message. Please do not reply.
          </div>
        </div>
      </body>
    </html>
    """

    msg.attach(MIMEText(html_body, 'html'))

    # Attach the image inline
    image_attachment = MIMEImage(image_data)
    image_attachment.add_header('Content-ID', '<image1>')
    image_attachment.add_header('Content-Disposition', 'inline', filename=filename)
    msg.attach(image_attachment)

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        print(f"✅ Email sent to {recipient_email}")
        return True
    except Exception as e:
        print(f"❌ Failed to send email to {recipient_email}: {e}")
        return False

# ============================================================
# Directory Monitoring Function
# ============================================================
def check_and_send_emails():
    recipients = [
        "ashish.kamboj@madrecert.com",
        "zishan@madrecert.com",
        # "dhruvchoudhary88649@gmail.com",
        # "rajiv.bana@hexaclimate.com"
    ]

    while True:
        image_files = [f for f in os.listdir(output_dir) if f.endswith(('.jpg', '.png'))]

        for image_file in image_files:
            image_path = os.path.join(output_dir, image_file)
            reason = image_file.split("_")[0]

            with open(image_path, 'rb') as img_file:
                image_data = img_file.read()

            all_sent = True

            for i, recipient in enumerate(recipients):
                sent = send_email(image_data, image_file, recipient, reason)
                if not sent:
                    all_sent = False

                # ⏳ Add delay between multiple recipient emails
                if i < len(recipients) - 1:
                    print(f"⏱️ Waiting {email_delay} seconds before sending next email...")
                    time.sleep(email_delay)

            # Always attempt to delete image after all email attempts
            try:
                os.remove(image_path)
                print(f"🗑️ Image {image_file} removed after email attempts.")
            except Exception as e:
                print(f"⚠️ Error deleting {image_file}: {e}")

            if not all_sent:
                print(f"⚠️ Some emails failed to send for {image_file}.")

        time.sleep(10)

# ============================================================
# Script Entry Point
# ============================================================
if __name__ == "__main__":
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        print("📁 Monitoring existing cropped_images folder...")

    print("🚀 Starting to monitor the cropped_images directory for new images.")
    check_and_send_emails()
