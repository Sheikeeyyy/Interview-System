import time
import os
import math
import sys

# friendly imports with guidance
try:
    import cv2
except ModuleNotFoundError:
    print("Missing dependency: opencv-python (cv2). Install with: pip install opencv-python")
    sys.exit(1)

try:
    import dlib
except ModuleNotFoundError:
    print("Missing dependency: dlib. Install with: pip install dlib")
    sys.exit(1)

try:
    import numpy as np
except ModuleNotFoundError:
    print("Missing dependency: numpy. Install with: pip install numpy")
    sys.exit(1)

# ---------------- CONFIG ----------------
GAZE_THRESHOLD_X = 0.35  # fraction of eye width (left/right)
GAZE_THRESHOLD_Y = 0.25  # fraction of eye height (up/down)
WARNING_TIME = 2          # seconds looking away before warning
# Head pose thresholds (degrees)
HEAD_YAW_THRESHOLD = 30    # moderate left/right turn (requires gaze agreement)
HEAD_PITCH_THRESHOLD = 25  # moderate up/down turn
HEAD_YAW_STRONG = 45      # very strong head turn -> immediate trigger
# smoothing
HEAD_SMOOTHING_FRAMES = 5
# ----------------------------------------

def get_bounding_box(landmarks):
    min_x = min(landmarks, key=lambda p: p.x).x
    max_x = max(landmarks, key=lambda p: p.x).x
    min_y = min(landmarks, key=lambda p: p.y).y
    max_y = max(landmarks, key=lambda p: p.y).y
    return min_x, max_x, min_y, max_y

def crop(image, bbox):
    return image[bbox[2]:bbox[3], bbox[0]:bbox[1]]

def filter_for_iris(eye_image):
    gray = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 10, 15, 15)
    gray = cv2.equalizeHist(gray)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return 255 - thresh

def find_iris_location(iris_image):
    contours, _ = cv2.findContours(iris_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    m = cv2.moments(c)
    if m["m00"] == 0:
        return None
    x = int(m["m10"] / m["m00"])
    y = int(m["m01"] / m["m00"])
    return x, y

def is_looking_away(norm_offset):
    x, y = norm_offset
    return abs(x) > GAZE_THRESHOLD_X or abs(y) > GAZE_THRESHOLD_Y


def get_head_pose(landmarks, size):
    # landmarks: list-like of dlib points
    # size: frame.shape -> (h, w, _)
    image_points = np.array([
        (landmarks[30].x, landmarks[30].y),  # Nose tip
        (landmarks[8].x, landmarks[8].y),    # Chin
        (landmarks[36].x, landmarks[36].y),  # Left eye left corner
        (landmarks[45].x, landmarks[45].y),  # Right eye right corner
        (landmarks[48].x, landmarks[48].y),  # Left Mouth corner
        (landmarks[54].x, landmarks[54].y)   # Right mouth corner
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ], dtype="double")

    h, w = size[0], size[1]
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None

    rmat, _ = cv2.Rodrigues(rotation_vector)
    sy = math.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(rmat[2, 1], rmat[2, 2])
        y = math.atan2(-rmat[2, 0], sy)
        z = math.atan2(rmat[1, 0], rmat[0, 0])
    else:
        x = math.atan2(-rmat[1, 2], rmat[1, 1])
        y = math.atan2(-rmat[2, 0], sy)
        z = 0

    # return yaw, pitch, roll in degrees
    return math.degrees(y), math.degrees(x), math.degrees(z)

# ---------------- INITIALIZE ----------------
detector = dlib.get_frontal_face_detector()

# Locate the predictor file
def find_predictor(filename="shape_predictor_68_face_landmarks.dat"):
    for root, _, files in os.walk(os.getcwd()):
        if filename in files:
            return os.path.join(root, filename)
    return None

predictor_path = find_predictor()

def download_and_extract_predictor(dest_path):
    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    try:
        import urllib.request, shutil, bz2
        tmp_bz2 = dest_path + ".bz2"
        print(f"Downloading predictor from {url} ...")
        with urllib.request.urlopen(url) as resp, open(tmp_bz2, "wb") as out:
            shutil.copyfileobj(resp, out)
        print("Extracting predictor...")
        with bz2.open(tmp_bz2, "rb") as f_in, open(dest_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(tmp_bz2)
        print(f"Predictor saved to {dest_path}")
    except Exception as e:
        raise RuntimeError(f"Automatic download failed: {e}")

if predictor_path is None:
    predictor_path = os.path.join(os.getcwd(), "shape_predictor_68_face_landmarks.dat")
    try:
        download_and_extract_predictor(predictor_path)
    except Exception:
        raise FileNotFoundError(
            f"Cannot find {predictor_path}. Place it in project root or download from:\n"
            "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        )

predictor = dlib.shape_predictor(predictor_path)

cap = cv2.VideoCapture(0)
look_away_start = None
warning_active = False
head_yaw_buffer = []
head_pitch_buffer = []
# ------------------------------------------

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detector(frame)

    # No face detected -> treat as looking away after WARNING_TIME
    if len(faces) == 0:
        if look_away_start is None:
            look_away_start = time.time()
        elif time.time() - look_away_start > WARNING_TIME:
            warning_active = True
        else:
            warning_active = False
        if warning_active:
            cv2.putText(frame,
                        "WARNING: PLEASE FACE THE SCREEN",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3)
        cv2.imshow("Interview Eye Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    # use largest detected face
    face = max(faces, key=lambda r: (r.right() - r.left()) * (r.bottom() - r.top()))
    landmarks = predictor(frame, face).parts()

    left_bbox = get_bounding_box(landmarks[36:42])
    right_bbox = get_bounding_box(landmarks[42:48])

    left_eye = crop(frame, left_bbox)
    right_eye = crop(frame, right_bbox)

    left_iris = filter_for_iris(left_eye)
    right_iris = filter_for_iris(right_eye)

    left_pos = find_iris_location(left_iris)
    right_pos = find_iris_location(right_iris)

    if left_pos is None or right_pos is None:
        # fall back to head-pose only
        gaze_away = False
    else:
        # Eye centers
        left_center = ((left_bbox[1]-left_bbox[0])//2, (left_bbox[3]-left_bbox[2])//2)
        right_center = ((right_bbox[1]-right_bbox[0])//2, (right_bbox[3]-right_bbox[2])//2)

        # Normalized offset (-1..1)
        left_offset = ((left_pos[0] - left_center[0]) / max(1, (left_bbox[1]-left_bbox[0])),
                       (left_pos[1] - left_center[1]) / max(1, (left_bbox[3]-left_bbox[2])))
        right_offset = ((right_pos[0] - right_center[0]) / max(1, (right_bbox[1]-right_bbox[0])),
                        (right_pos[1] - right_center[1]) / max(1, (right_bbox[3]-right_bbox[2])))

        avg_offset = ((left_offset[0] + right_offset[0]) / 2,
                      (left_offset[1] + right_offset[1]) / 2)

        gaze_away = is_looking_away(avg_offset)

    # Head pose estimation
    head_pose = None
    try:
        head_pose = get_head_pose(landmarks, frame.shape)
    except Exception:
        head_pose = None

    head_away = False
    head_strong = False
    if head_pose is not None:
        yaw, pitch, roll = head_pose
        # append to smoothing buffers
        head_yaw_buffer.append(yaw)
        head_pitch_buffer.append(pitch)
        if len(head_yaw_buffer) > HEAD_SMOOTHING_FRAMES:
            head_yaw_buffer.pop(0)
            head_pitch_buffer.pop(0)

        # median smoothing to reduce jitter
        try:
            smooth_yaw = float(np.median(np.array(head_yaw_buffer)))
            smooth_pitch = float(np.median(np.array(head_pitch_buffer)))
        except Exception:
            smooth_yaw, smooth_pitch = yaw, pitch

        # very large yaw -> consider strong away
        head_strong = abs(smooth_yaw) > HEAD_YAW_STRONG
        # moderate head-away requires gaze agreement
        head_away = (abs(smooth_yaw) > HEAD_YAW_THRESHOLD and gaze_away) or (abs(smooth_pitch) > HEAD_PITCH_THRESHOLD and gaze_away)
    else:
        # clear buffers if no pose
        head_yaw_buffer.clear()
        head_pitch_buffer.clear()

    # combined decision: require either strong head turn, or gaze+moderate head, or gaze alone
    should_warn_now = False
    if head_strong:
        should_warn_now = True
    elif head_away and gaze_away:
        should_warn_now = True
    elif gaze_away and not head_pose:
        # if gaze indicates away but head pose unavailable, allow gaze to trigger
        should_warn_now = True

    if should_warn_now:
        if look_away_start is None:
            look_away_start = time.time()
        elif time.time() - look_away_start > WARNING_TIME:
            warning_active = True
    else:
        look_away_start = None
        warning_active = False

    # Draw iris points when available
    if left_pos:
        cv2.circle(left_eye, left_pos, 2, (0, 0, 255), -1)
    if right_pos:
        cv2.circle(right_eye, right_pos, 2, (0, 0, 255), -1)

    # Show warning
    if warning_active:
        cv2.putText(frame,
                    "WARNING: PLEASE LOOK AT THE SCREEN",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3)

    cv2.imshow("Interview Eye Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
