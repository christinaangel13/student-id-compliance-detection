"""
Student ID / Lanyard Compliance Detector
==========================================
Step 3-6: Real-time webcam app that:
  - Detects faces
  - Detects ID cards / lanyards (custom YOLOv8 model)
  - Draws GREEN box if the student is wearing a card/lanyard, RED if not
  - On non-compliance, saves a cropped screenshot + logs it to CSV
  - (Optional) Identifies the student by name using face_recognition

Run:
    python main.py
Press 'q' to quit.
"""

import os
import csv
import cv2
import time
import numpy as np
from datetime import datetime
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
YOLO_WEIGHTS = "id_lanyard_best.pt"     # trained model from train_model.py
CONF_THRESHOLD = 0.15                    # detection confidence cutoff (lowered for debugging)
VIOLATIONS_DIR = "violations"
LOG_FILE = "violation_log.csv"
KNOWN_FACES_DIR = "known_faces"          # subfolders per student OR flat "Name.jpg" files
SCREENSHOT_COOLDOWN_SEC = 60              # avoid spamming screenshots of same student
VIOLATION_CONFIRM_SEC = 2                 # must stay non-compliant this long before it counts
POSITION_MATCH_PIXELS = 60                # how close a face must be between frames to count as "the same person"
USE_FACE_RECOGNITION = True             # set False to skip name identification

# Region below the face where we expect to see a card/lanyard (neck/chest)
NECK_REGION_HEIGHT_RATIO = 1.8           # how far below the face to look, x face height
NECK_REGION_WIDTH_RATIO = 1.3            # how much wider than the face to look

# ---------------------------------------------------------------------------
# OPTIONAL: face recognition setup
# ---------------------------------------------------------------------------
known_encodings = []
known_names = []

if USE_FACE_RECOGNITION:
    try:
        import face_recognition

        for fname in os.listdir(KNOWN_FACES_DIR):
            path = os.path.join(KNOWN_FACES_DIR, fname)
            if not os.path.isfile(path):
                continue
            name = os.path.splitext(fname)[0]

            # Force the image into clean 8-bit RGB (no alpha channel, no
            # palette/CMYK modes) — face_recognition/dlib only accepts
            # 8-bit gray or RGB and errors out on anything else.
            try:
                from PIL import Image
                pil_img = Image.open(path).convert("RGB")
                img = np.ascontiguousarray(np.array(pil_img, dtype=np.uint8))
            except Exception as e:
                print(f"[WARN] Could not load '{fname}': {e}")
                continue

            encs = face_recognition.face_encodings(img)
            if encs:
                known_encodings.append(encs[0])
                known_names.append(name)
            else:
                print(f"[WARN] No face found in '{fname}' — skipped.")
        print(f"[INFO] Loaded {len(known_names)} known student face(s).")
    except ImportError:
        print("[WARN] face_recognition not installed; names will show as 'Unknown'.")
        USE_FACE_RECOGNITION = False


def identify_student(frame_rgb, face_box):
    """Return best-matching student name for a face box, or 'Unknown'."""
    if not USE_FACE_RECOGNITION or not known_encodings:
        return "Unknown"

    import face_recognition
    x, y, w, h = face_box
    # face_recognition wants (top, right, bottom, left)
    loc = [(y, x + w, y + h, x)]
    encs = face_recognition.face_encodings(frame_rgb, loc)
    if not encs:
        return "Unknown"

    matches = face_recognition.compare_faces(known_encodings, encs[0], tolerance=0.5)
    distances = face_recognition.face_distance(known_encodings, encs[0])
    if len(distances) == 0:
        return "Unknown"

    best_idx = int(np.argmin(distances))
    if matches[best_idx]:
        return known_names[best_idx]
    return "Unknown"


# ---------------------------------------------------------------------------
# SETUP
# ---------------------------------------------------------------------------
os.makedirs(VIOLATIONS_DIR, exist_ok=True)

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        csv.writer(f).writerow(["timestamp", "student_name", "status", "screenshot_path"])

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

yolo_model = YOLO(YOLO_WEIGHTS)

last_screenshot_time = {}  # name -> last capture timestamp, for cooldown


def boxes_overlap(box_a, box_b):
    """Simple AABB overlap check. Each box = (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def get_neck_region(face_x, face_y, face_w, face_h, frame_shape):
    """Estimate the chest/neck area below a detected face, where a card
    or lanyard would appear."""
    region_w = int(face_w * NECK_REGION_WIDTH_RATIO)
    region_h = int(face_h * NECK_REGION_HEIGHT_RATIO)
    cx = face_x + face_w // 2

    x1 = max(0, cx - region_w // 2)
    y1 = face_y + face_h  # start right below the chin
    x2 = min(frame_shape[1], cx + region_w // 2)
    y2 = min(frame_shape[0], y1 + region_h)
    return (x1, y1, x2, y2)


def find_or_create_pending_slot(pending_violations, cx, cy, now):
    """Find an existing tracked slot near (cx, cy) — i.e. roughly the same
    person as a previous frame — or create a new one. Returns the slot dict.
    This lets us measure how long a given face has been continuously
    non-compliant without needing full identity tracking."""
    for slot in pending_violations.values():
        if abs(slot["cx"] - cx) < POSITION_MATCH_PIXELS and abs(slot["cy"] - cy) < POSITION_MATCH_PIXELS:
            slot["cx"], slot["cy"] = cx, cy
            slot["last_seen"] = now
            return slot

    new_slot = {"cx": cx, "cy": cy, "first_seen": now, "last_seen": now}
    pending_violations[id(new_slot)] = new_slot
    return new_slot


def clear_matching_slot(pending_violations, cx, cy):
    """Remove a tracked slot if this position is now compliant again."""
    for key, slot in list(pending_violations.items()):
        if abs(slot["cx"] - cx) < POSITION_MATCH_PIXELS and abs(slot["cy"] - cy) < POSITION_MATCH_PIXELS:
            del pending_violations[key]


def prune_stale_slots(pending_violations, now, max_age=10):
    """Drop tracked slots that haven't been seen in a while (person left frame)."""
    for key, slot in list(pending_violations.items()):
        if now - slot["last_seen"] > max_age:
            del pending_violations[key]


# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------
def main():
    # CAP_DSHOW is more reliable on Windows than the default backend, which
    # can occasionally hang waiting for a frame. We also stopped forcing an
    # exact resolution, since not all webcams support every resolution and
    # forcing an unsupported one can cause read() to stall indefinitely.
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)  # fallback to default backend
    cv2.namedWindow("Student ID Compliance Detector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Student ID Compliance Detector", 1000, 700)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    session_violations = []  # tracks ONLY violations from this run, for the email report
    pending_violations = {}  # tracks how long each roughly-tracked face has been non-compliant

    print("[INFO] Starting detection. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        raw_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=12, minSize=(120, 120))

        # Extra filter: ignore anything detected in the top 12% of the frame.
        # Ceiling fans, lights, and wall clutter tend to sit up there; real
        # faces on a webcam almost never do.
        frame_h = frame.shape[0]
        ignore_above_y = int(frame_h * 0.12)
        faces = [f for f in raw_faces if f[1] > ignore_above_y]

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        for (fx, fy, fw, fh) in faces:
            neck_box = get_neck_region(fx, fy, fw, fh, frame.shape)
            nx1, ny1, nx2, ny2 = neck_box

            # Only search for an ID card within THIS face's own neck/chest
            # region, not the whole frame. This prevents distant background
            # objects (posters, door decorations, etc.) from ever being
            # mistaken for a card, since they're never even scanned.
            neck_crop = frame[ny1:ny2, nx1:nx2]
            item_boxes = []
            if neck_crop.size > 0:
                yolo_results = yolo_model.predict(neck_crop, conf=CONF_THRESHOLD, imgsz=320, verbose=False)[0]
                for box in yolo_results.boxes:
                    bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    # translate crop-local coords back to full-frame coords
                    item_boxes.append((bx1 + nx1, by1 + ny1, bx2 + nx1, by2 + ny1, cls))

            # DEBUG: draw every detection found within this neck region in blue
            for (ix1, iy1, ix2, iy2, icls) in item_boxes:
                label = yolo_model.names[icls]
                cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), (255, 150, 0), 2)
                cv2.putText(frame, label, (ix1, iy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 0), 2)

            compliant = len(item_boxes) > 0

            color = (0, 200, 0) if compliant else (0, 0, 255)  # BGR: green / red
            status = "COMPLIANT" if compliant else "NO ID DETECTED"

            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), color, 2)
            cv2.putText(frame, status, (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            now = time.time()
            cx, cy = fx + fw // 2, fy + fh // 2  # rough position used to track "the same person" across frames

            if compliant:
                # Compliant again — clear any in-progress violation timer for this position
                clear_matching_slot(pending_violations, cx, cy)
            else:
                slot = find_or_create_pending_slot(pending_violations, cx, cy, now)
                sustained_for = now - slot["first_seen"]

                # Show a small countdown so it's clear on-screen that a violation
                # is being confirmed, not yet finalized
                remaining = max(0, VIOLATION_CONFIRM_SEC - sustained_for)
                if remaining > 0:
                    cv2.putText(frame, f"confirming... {remaining:.1f}s", (fx, fy + fh + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

                if sustained_for >= VIOLATION_CONFIRM_SEC:
                    name = identify_student(frame_rgb, (fx, fy, fw, fh))
                    last_time = last_screenshot_time.get(name, 0)

                    if now - last_time > SCREENSHOT_COOLDOWN_SEC:
                        last_screenshot_time[name] = now
                        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                        # EVIDENCE CROP: include the face AND the neck/chest region
                        # (where an ID/lanyard would be), plus extra room above for
                        # the "NO ID DETECTED" label, so the saved image is proof,
                        # not just a face photo.
                        pad_top = 40   # room for the red label text above the face
                        pad_side = 40
                        ex1 = max(0, fx - pad_side)
                        ey1 = max(0, fy - pad_top)
                        ex2 = min(frame.shape[1], neck_box[2] + pad_side)
                        ey2 = min(frame.shape[0], neck_box[3] + pad_side // 2)
                        evidence_crop = frame[ey1:ey2, ex1:ex2]  # frame already has box+label drawn on it

                        fname = f"{timestamp}_{name}.jpg"
                        fpath = os.path.join(VIOLATIONS_DIR, fname)
                        cv2.imwrite(fpath, evidence_crop)

                        with open(LOG_FILE, "a", newline="") as f:
                            csv.writer(f).writerow([timestamp, name, "NO_ID", fpath])

                        session_violations.append([timestamp, name, "NO_ID", fpath])

                        print(f"[VIOLATION] {name} flagged at {timestamp} -> {fpath}")

        cv2.imshow("Student ID Compliance Detector", frame)
        prune_stale_slots(pending_violations, time.time())
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Automatically email THIS SESSION's violation report (PDF) when quit —
    # only what happened between running main.py and pressing 'q', so
    # re-running the app doesn't re-send earlier violations.
    try:
        import send_daily_report
        print("[INFO] Session ended — preparing violation report email...")

        if session_violations:
            report_date = datetime.now().strftime("%Y-%m-%d")
            session_label = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            pdf_path = send_daily_report.build_pdf_report(
                header=None, rows=session_violations, today_str=session_label
            )
            send_daily_report.send_email(pdf_path, len(session_violations), report_date)
        else:
            print("[INFO] No violations this session — no email sent.")
    except Exception as e:
        print(f"[WARN] Could not send report email: {e}")


if __name__ == "__main__":
    main()
