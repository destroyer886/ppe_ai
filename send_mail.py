# -*- coding:utf-8 -*-
import os
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import sys
import subprocess
import fcntl
import socket
import shutil
import requests
import json
from datetime import datetime


# ============================================================
# Email Configuration
# ============================================================
sender_email = "hexahse@gmail.com"
sender_password = "ukqo jmoo pvso snpy"
subject = "🚨 PPE Violation Alert - Missing PPE Detected,updated 5.1"

output_dir = 'cropped_images'
email_delay = 2  # seconds delay between multiple recipients
LOCK_FILE = "/tmp/send_mail.lock"
REPO_URL = "https://github.com/destroyer886/ppe_ai"
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
BRANCH = "main"

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


# ============================================================
# Prevent multiple instances
# ============================================================

send_log("Mail", "starting", "Booting mail system")
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


# ============================================================
# Git / Internet utilities
# ============================================================
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



# ============================================================
# Email Sending Function (inline + downloadable)
# ============================================================
def send_email(image_data, filename, recipient_email, reason):
    msg = MIMEMultipart("related")
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject

    # HTML Email Body
    html_body = f"""
   <!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
</head>
<body style="margin:0;padding:0;background-color:#f4f6f8;font-family:Arial,Helvetica,sans-serif;">

  <table width="100%" cellpadding="0" cellspacing="0" style="background-color:#f4f6f8;padding:20px 0;">
    <tr>
      <td align="center">

        <!-- Card Container -->
        <table width="600" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:12px;overflow:hidden;box-shadow:0 4px 20px rgba(0,0,0,0.08);">

          <!-- Header -->
          <tr>
            <td style="background:linear-gradient(90deg,#ff4b2b,#ff416c);padding:20px;text-align:center;">
              <h2 style="color:#ffffff;margin:0;font-size:22px;letter-spacing:1px;">
                ⚠ PPE SAFETY ALERT
              </h2>
            </td>
          </tr>

          <!-- Body -->
          <tr>
            <td style="padding:30px;">

              <h3 style="color:#ff4b2b;margin-top:0;">
                PPE Violation Detected
              </h3>

              <!-- Reason Box -->
              <div style="background:#fff3f3;border-left:5px solid #ff4b2b;padding:15px;margin-bottom:20px;border-radius:6px;">
                <strong style="color:#d63031;">Reason:</strong>
                <span style="color:#333;"> {reason} </span>
              </div>

              <!-- Info Grid -->
              <table width="100%" cellpadding="8" cellspacing="0" style="border-collapse:collapse;margin-bottom:20px;">
                <tr style="background:#f9fafb;">
                  <td style="font-weight:bold;color:#555;">Location</td>
                  <td style="color:#333;">Sirsa</td>
                </tr>
                <tr>
                  <td style="font-weight:bold;color:#555;">Machine</td>
                  <td style="color:#333;">Sirsa-B01</td>
                </tr>
                <tr style="background:#f9fafb;">
                  <td style="font-weight:bold;color:#555;">Time</td>
                  <td style="color:#333;">{datetime.now().strftime("%c")}</td>
                </tr>
              </table>

              <!-- Image Section -->
              <div style="text-align:center;margin-top:20px;">
                <img src="cid:image1"
                     style="max-width:100%;border-radius:10px;border:1px solid #ddd;box-shadow:0 2px 10px rgba(0,0,0,0.1);" />
              </div>

              <!-- Footer Note -->
              <p style="margin-top:25px;font-size:13px;color:#777;text-align:center;">
                This is an automated safety monitoring alert.
                Please take immediate corrective action.
              </p>

            </td>
          </tr>

          <!-- Footer -->
          <tr>
            <td style="background:#f4f6f8;padding:15px;text-align:center;font-size:12px;color:#999;">
              © {datetime.now().year} PPE Monitoring System | Industrial Safety Division
            </td>
          </tr>

        </table>

      </td>
    </tr>
  </table>

</body>
</html>
    """

    msg.attach(MIMEText(html_body, 'html'))

    # Inline image for preview
    if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg"):
     subtype = "jpeg"
    elif filename.lower().endswith(".png"):
     subtype = "png"
    else:
     print(f"❌ Unsupported image format: {filename}")
     return False

    inline_image = MIMEImage(image_data, _subtype=subtype)
    inline_image.add_header('Content-ID', '<image1>')
    inline_image.add_header('Content-Disposition', 'inline', filename=filename)
    msg.attach(inline_image)

    # Downloadable attachment
    attachment = MIMEImage(image_data, _subtype=subtype)
    attachment.add_header('Content-Disposition', 'attachment', filename=filename)
    msg.attach(attachment)

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
        "Abul.f@sunkonnect.co",
        "mantu.p@sunkonnect.co",
        "Rajiv.bana@hexaclimate.com",
        "surjeet.k@sunkonnect.co"
    ]
    send_log("Mail", "started", "mail system is now running")

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

                # ⏳ Delay between emails
                if i < len(recipients) - 1:
                    print(f"⏱️ Waiting {email_delay} seconds before sending next email...")
                    time.sleep(email_delay)

            # Delete image after attempts
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
    single_instance_lock()
    updated = update_repo()

    if updated:
        print("♻️ Restarting with new code...")
        python = sys.executable
        os.execl(python, python, *sys.argv)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        print("📁 Monitoring existing cropped_images folder...")

    print("🚀 Starting to monitor the cropped_images directory for new images.")
    check_and_send_emails()
