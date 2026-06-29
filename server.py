from flask import Flask, Response, jsonify, send_file, request
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import time
import math
import os
import urllib.request
import threading

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────
# Camera check
# ─────────────────────────────────────────────
camera_available = True
try:
    test_cap = cv2.VideoCapture(0)
    if not test_cap.isOpened():
        camera_available = False
    test_cap.release()
except Exception:
    camera_available = False
print(f"Camera available: {camera_available}")

# ─────────────────────────────────────────────
# Download model (required for multi-person)
# ─────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_landmarker_full.task")
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
               "pose_landmarker_full/float16/latest/pose_landmarker_full.task")

if not os.path.exists(MODEL_PATH):
    print(f"Downloading PoseLandmarker model → {MODEL_PATH}")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded OK.")
    except Exception as e:
        print(f"Download failed: {e}")
        MODEL_PATH = None

MAX_PERSONS = 4
use_old_api = True

if MODEL_PATH and os.path.exists(MODEL_PATH):
    try:
        BaseOptions        = mp.tasks.BaseOptions
        PoseLandmarker     = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOpts = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode  = mp.tasks.vision.RunningMode
        pose_options = PoseLandmarkerOpts(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=MAX_PERSONS,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        pose = PoseLandmarker.create_from_options(pose_options)
        use_old_api = False
        print(f"Multi-person PoseLandmarker ready (up to {MAX_PERSONS})")
    except Exception as e:
        print(f"PoseLandmarker failed ({e}), using single-person fallback")

if use_old_api:
    mp_pose_mod = mp.solutions.pose
    pose = mp_pose_mod.Pose(
        static_image_mode=False, model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )
    MAX_PERSONS = 1
    print("Using single-person fallback API")

# ─────────────────────────────────────────────
# Palettes — BGR for OpenCV
# ─────────────────────────────────────────────
PERSON_PALETTES = [
    {"primary":(180,30,30),  "accent":(80,180,255), "shadow":(80,10,10),  "white":(220,255,255)},  # blue
    {"primary":(30,150,30),  "accent":(80,255,80),  "shadow":(10,60,10),  "white":(220,255,220)},  # green
    {"primary":(20,100,200), "accent":(50,200,255), "shadow":(10,40,80),  "white":(220,240,255)},  # orange
    {"primary":(150,30,150), "accent":(255,80,255), "shadow":(60,10,60),  "white":(255,220,255)},  # magenta
]

# ─────────────────────────────────────────────
# Suspicious activity thresholds & ROI Parameters
# ─────────────────────────────────────────────
STANDING_STILL_SECONDS = 5.0
LOOKING_CAMERA_SECONDS = 5.0
FACE_HIDDEN_SECONDS = 2.0
STILL_ANCHOR_RADIUS_PX = 38.0
EVENT_COOLDOWN_SECONDS = 6.0
RED = (0, 0, 255)
YELLOW = (0, 220, 255)
CROUCH_HEIGHT_REDUCTION_THRESHOLD = 0.30
CROUCH_CONSECUTIVE_FRAMES = 3

# Region of Interest (ROI) boundaries (normalized 0.0 to 1.0)
roi_x1 = 0.35
roi_x2 = 0.65
roi_y1 = 0.1
roi_y2 = 0.9

# ─────────────────────────────────────────────
# Stable person tracker (centroid matching)
# ─────────────────────────────────────────────
class SmoothedLandmark:
    def __init__(self, x, y, z, visibility):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility

class PersonTracker:
    """Assigns stable slot indices (0-3) to detected persons across frames with hysteresis and coordinate smoothing."""
    def __init__(self, max_persons=4, max_dist=250, max_lost_frames=60, alpha=0.35):
        self.slots   = {}        # slot_id → last centroid (x,y)
        self.lost_frames = {}    # slot_id → count of frames lost
        self.smoothed_lms = {}   # slot_id → list of SmoothedLandmark
        self.max_p   = max_persons
        self.max_dist= max_dist
        self.max_lost_frames = max_lost_frames
        self.alpha   = alpha

    def _centroid(self, landmarks, w, h):
        # Torso landmarks: 11 (L shoulder), 12 (R shoulder), 23 (L hip), 24 (R hip)
        torso_indices = [11, 12, 23, 24]
        xs = [landmarks[i].x * w for i in torso_indices if i < len(landmarks)]
        ys = [landmarks[i].y * h for i in torso_indices if i < len(landmarks)]
        if len(xs) >= 2:
            return (sum(xs)/len(xs), sum(ys)/len(ys))
        # Fallback to all landmarks
        xs = [l.x * w for l in landmarks]
        ys = [l.y * h for l in landmarks]
        if xs and ys:
            return (sum(xs)/len(xs), sum(ys)/len(ys))
        return (w / 2, h / 2)

    def assign(self, all_landmarks, w, h):
        """Return list of (slot_id, smoothed_landmarks) with stable slot assignment."""
        if not all_landmarks:
            for slot_id in list(self.slots.keys()):
                self.lost_frames[slot_id] = self.lost_frames.get(slot_id, 0) + 1
                if self.lost_frames[slot_id] > self.max_lost_frames:
                    self.slots.pop(slot_id, None)
                    self.lost_frames.pop(slot_id, None)
                    self.smoothed_lms.pop(slot_id, None)
            return []

        new_centroids = [self._centroid(lm, w, h) for lm in all_landmarks]
        used_slots    = set()
        assignment    = [None] * len(all_landmarks)

        # 1. Match existing slots (both active and lost) to nearest new centroids
        all_slots = sorted(self.slots.keys(), key=lambda s: self.lost_frames.get(s, 0))
        for slot_id in all_slots:
            old_c = self.slots[slot_id]
            best_i, best_d = None, self.max_dist
            for i, nc in enumerate(new_centroids):
                if assignment[i] is not None:
                    continue
                d = math.hypot(nc[0]-old_c[0], nc[1]-old_c[1])
                if d < best_d:
                    best_d, best_i = d, i
            if best_i is not None:
                assignment[best_i] = slot_id
                used_slots.add(slot_id)
                self.lost_frames[slot_id] = 0

        # 2. Increment lost frames counter for slots that did not get matched
        for slot_id in list(self.slots.keys()):
            if slot_id not in used_slots:
                self.lost_frames[slot_id] = self.lost_frames.get(slot_id, 0) + 1
                if self.lost_frames[slot_id] > self.max_lost_frames:
                    self.slots.pop(slot_id, None)
                    self.lost_frames.pop(slot_id, None)
                    self.smoothed_lms.pop(slot_id, None)

        # 3. Assign new slots to unmatched detections
        active_or_lost_slots = set(self.slots.keys())
        free_slots = [s for s in range(self.max_p) if s not in active_or_lost_slots]
        for i, slot in enumerate(assignment):
            if slot is None and free_slots:
                assigned_slot = free_slots.pop(0)
                assignment[i] = assigned_slot
                self.lost_frames[assigned_slot] = 0

        # 4. Perform smoothing and update slot centroids
        result = []
        for i, slot in enumerate(assignment):
            if slot is not None:
                raw_lm = all_landmarks[i]
                if slot in self.smoothed_lms and len(self.smoothed_lms[slot]) == len(raw_lm):
                    smoothed_list = []
                    for idx, lm in enumerate(raw_lm):
                        prev = self.smoothed_lms[slot][idx]
                        smoothed_x = self.alpha * lm.x + (1.0 - self.alpha) * prev.x
                        smoothed_y = self.alpha * lm.y + (1.0 - self.alpha) * prev.y
                        smoothed_z = self.alpha * getattr(lm, 'z', 0.0) + (1.0 - self.alpha) * prev.z
                        smoothed_vis = self.alpha * getattr(lm, 'visibility', 1.0) + (1.0 - self.alpha) * prev.visibility
                        smoothed_list.append(SmoothedLandmark(smoothed_x, smoothed_y, smoothed_z, smoothed_vis))
                    self.smoothed_lms[slot] = smoothed_list
                else:
                    self.smoothed_lms[slot] = [
                        SmoothedLandmark(lm.x, lm.y, getattr(lm, 'z', 0.0), getattr(lm, 'visibility', 1.0))
                        for lm in raw_lm
                    ]
                
                self.slots[slot] = self._centroid(self.smoothed_lms[slot], w, h)
                result.append((slot, self.smoothed_lms[slot]))

        return result

tracker = PersonTracker(MAX_PERSONS)

# ─────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────
def draw_robot_head(img, center, pal, scale=1.0):
    x, y = int(center[0]), int(center[1])
    r = int(45 * scale)
    cv2.ellipse(img,(x,y),(r,int(r*.85)),0,0,360,pal["primary"],-1,cv2.LINE_AA)
    vw,vh = int(r*1.4), int(r*.45)
    cv2.ellipse(img,(x,y),(vw//2,vh//2),0,0,360,pal["accent"],-1,cv2.LINE_AA)
    cv2.ellipse(img,(x,y),(vw//2-6,vh//2-4),0,0,360,(120,40,10),2,cv2.LINE_AA)

def draw_robot_torso(img, sL, sR, hL, hR, pal):
    pts = np.array([[sL[0],sL[1]+6],[sR[0],sR[1]+6],[hR[0],hR[1]-10],[hL[0],hL[1]-10]],np.int32)
    cv2.fillPoly(img,[pts],pal["primary"])
    cv2.polylines(img,[pts],True,pal["accent"],3,cv2.LINE_AA)
    c1 = ((sL[0]+sR[0])//2,(sL[1]+sR[1])//2+20)
    for t in [.25,.5,.75]:
        Lp=(int(c1[0]+(hL[0]-c1[0])*t),int(c1[1]+(hL[1]-c1[1])*t))
        Rp=(int(c1[0]+(hR[0]-c1[0])*t),int(c1[1]+(hR[1]-c1[1])*t))
        cv2.line(img,Lp,Rp,(230,130,80),2,cv2.LINE_AA)
    cx=(sL[0]+sR[0]+hL[0]+hR[0])//4; cy=(sL[1]+sR[1]+hL[1]+hR[1])//4
    cv2.circle(img,(cx,cy),18,pal["accent"],-1)
    cv2.circle(img,(cx,cy),10,pal["primary"],-1)

def draw_spine(img, sL, sR, hL, hR, pal):
    top=((sL[0]+sR[0])//2,(sL[1]+sR[1])//2); bot=((hL[0]+hR[0])//2,(hL[1]+hR[1])//2)
    cv2.line(img,top,bot,pal["shadow"],16,cv2.LINE_AA)
    cv2.line(img,top,bot,pal["primary"],10,cv2.LINE_AA)
    for t in np.linspace(.2,.8,4):
        cv2.circle(img,(int(top[0]+(bot[0]-top[0])*t),int(top[1]+(bot[1]-top[1])*t)),8,pal["accent"],-1)

def draw_cylinder_limb(img, p1, p2, pal):
    length = int(math.hypot(p2[0]-p1[0],p2[1]-p1[1]))
    th = max(14, length//6)
    cv2.line(img,p1,p2,pal["shadow"],th+6,cv2.LINE_AA)
    cv2.line(img,p1,p2,pal["primary"],th,cv2.LINE_AA)
    cuff = th//2+4
    for pt in [p1,p2]:
        cv2.circle(img,pt,cuff,pal["accent"],-1)
        cv2.circle(img,pt,cuff-5,pal["primary"],-1)

def draw_joint_ball(img, center, pal):
    cv2.circle(img,center,18,pal["shadow"],-1)
    cv2.circle(img,center,12,pal["accent"],-1)
    cv2.circle(img,center,6,pal["white"],-1)

def draw_person(glow, landmarks, w, h, pal):
    pts = [(int(l.x*w), int(l.y*h)) for l in landmarks]
    def _pt(i): return pts[i] if 0<=i<len(pts) else (w//2,h//2)
    nose=_pt(0); Ls,Rs=_pt(11),_pt(12); Le,Re=_pt(13),_pt(14)
    Lw,Rw=_pt(15),_pt(16); Lh,Rh=_pt(23),_pt(24)
    Lk,Rk=_pt(25),_pt(26); La,Ra=_pt(27),_pt(28)
    draw_robot_head(glow,nose,pal)
    draw_robot_torso(glow,Ls,Rs,Lh,Rh,pal)
    draw_spine(glow,Ls,Rs,Lh,Rh,pal)
    for a,b in [(Ls,Le),(Le,Lw),(Rs,Re),(Re,Rw),(Lh,Lk),(Lk,La),(Rh,Rk),(Rk,Ra)]:
        draw_cylinder_limb(glow,a,b,pal)
    for j in [Ls,Rs,Le,Re,Lw,Rw,Lh,Rh,Lk,Rk,La,Ra]:
        draw_joint_ball(glow,j,pal)
    return pts

# ─────────────────────────────────────────────
# Global state
# ─────────────────────────────────────────────
start_time     = None
prev_pts_map   = {}      # slot_id → pts from previous frame
last_time      = time.time()
cap            = None
session_start  = None
suspicious_state = {}
suspicious_events = []
global_frame_ts_ms = 1

stats = {
    'fps':0,
    'confidence':0,
    'motion':0,
    'keypoints':0,
    'persons_detected':0,
    'fps_history':[],
    'confidence_history':[],
    'motion_history':[],
    'persons_history':[],
    'camera_status':'active'
    if camera_available else 'inactive',
    'start_time':None,
    'multi_person': not use_old_api,
    'max_persons':MAX_PERSONS,
    'session_duration':0,
    'suspicious_current':[],
    'suspicious_summary':{
        'total':0,
        'high':0,
        'medium_high':0,
        'standing_still':0,
        'looking_camera':0,
        'face_hidden':0,
        'frequent_looking':0,
        'repeated_entry_exit':0,
        'crouching_bending':0,
        'sudden_disappearance':0,
    },
    'suspicious_events':[],
}

def default_suspicious_summary():
    return {
        'total':0,
        'high':0,
        'medium_high':0,
        'standing_still':0,
        'looking_camera':0,
        'face_hidden':0,
        'frequent_looking':0,
        'repeated_entry_exit':0,
        'crouching_bending':0,
        'sudden_disappearance':0,
    }

def reset_suspicious_state():
    global suspicious_state, suspicious_events
    suspicious_state = {}
    suspicious_events = []
    stats['suspicious_current'] = []
    stats['suspicious_events'] = []
    stats['suspicious_summary'] = default_suspicious_summary()

def landmark_visibility(landmarks, idx):
    if idx >= len(landmarks):
        return 0.0
    return float(getattr(landmarks[idx], 'visibility', 1.0))

def face_visible_score(landmarks):
    face_ids = list(range(0, 11))
    vals = [landmark_visibility(landmarks, i) for i in face_ids if i < len(landmarks)]
    return sum(vals) / len(vals) if vals else 0.0

def is_looking_camera(landmarks, pts, w):
    if len(pts) <= 12 or face_visible_score(landmarks) < 0.55:
        return False
    nose = pts[0]
    left_eye = pts[2] if len(pts) > 2 else nose
    right_eye = pts[5] if len(pts) > 5 else nose
    left_shoulder = pts[11]
    right_shoulder = pts[12]
    face_width = max(abs(left_eye[0] - right_eye[0]), 1)
    shoulder_width = max(abs(left_shoulder[0] - right_shoulder[0]), 1)
    face_center = (left_eye[0] + right_eye[0]) / 2
    shoulder_center = (left_shoulder[0] + right_shoulder[0]) / 2
    centered_on_body = abs(face_center - shoulder_center) < shoulder_width * 0.22
    eyes_level = abs(left_eye[1] - right_eye[1]) < max(face_width * 0.45, 8)
    nose_centered = abs(nose[0] - face_center) < max(face_width * 0.75, w * 0.03)
    return centered_on_body and eyes_level and nose_centered

def is_face_hidden(landmarks, pts, w, h):
    face_score = face_visible_score(landmarks)
    nose = pts[0] if pts else (w // 2, h // 2)
    face_hidden_by_limb = False
    face_shape_bad = False
    if len(pts) > 16:
        left_eye = pts[2] if len(pts) > 2 else nose
        right_eye = pts[5] if len(pts) > 5 else nose
        left_ear = pts[7] if len(pts) > 7 else left_eye
        right_ear = pts[8] if len(pts) > 8 else right_eye
        left_shoulder = pts[11]
        right_shoulder = pts[12]
        shoulder_width = max(abs(left_shoulder[0] - right_shoulder[0]), 1)
        shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
        face_center = ((left_eye[0] + right_eye[0] + nose[0]) / 3, (left_eye[1] + right_eye[1] + nose[1]) / 3)
        eye_width = abs(left_eye[0] - right_eye[0])
        ear_width = abs(left_ear[0] - right_ear[0])
        face_radius = max(ear_width * 0.55, shoulder_width * 0.28, w * 0.06)
        for limb_idx in (13, 14, 15, 16):
            limb = pts[limb_idx]
            near_face = math.hypot(limb[0] - face_center[0], limb[1] - face_center[1]) < face_radius * 1.45
            raised_to_face = limb[1] < shoulder_y and abs(limb[0] - face_center[0]) < shoulder_width * 0.55
            if near_face or raised_to_face:
                face_hidden_by_limb = True
                break
        face_shape_bad = eye_width < shoulder_width * 0.04 or face_center[1] > shoulder_y
    return face_score < 0.62 or face_hidden_by_limb or face_shape_bad

def body_center(pts):
    if len(pts) > 24:
        body_ids = [11, 12, 23, 24]
    elif len(pts) > 12:
        body_ids = [11, 12]
    else:
        body_ids = list(range(len(pts)))
    xs = [pts[i][0] for i in body_ids]
    ys = [pts[i][1] for i in body_ids]
    return (sum(xs) / len(xs), sum(ys) / len(ys)) if xs else None

def estimate_head_direction(landmarks, pts):
    """Estimate head direction based on nose and ear/eye horizontal distances.
    Returns asymmetry ratio (negative is looking left/camera right, positive is looking right/camera left).
    """
    if not pts or len(pts) <= 8:
        return 0.0
    nose = pts[0]
    
def estimate_head_direction(landmarks, pts):
    """Estimate head direction based on nose and eye horizontal distances.
    Returns asymmetry ratio (negative is looking right/camera left, positive is looking left/camera right).
    """
    if not pts or len(pts) <= 5:
        return 0.0
    nose_x = pts[0][0]
    left_eye_x = pts[2][0]
    right_eye_x = pts[5][0]
    
    eye_distance = max(abs(left_eye_x - right_eye_x), 1)
    eye_center_x = (left_eye_x + right_eye_x) / 2.0
    
    return (nose_x - eye_center_x) / eye_distance

def update_suspicious_activity(slot, landmarks, pts, motion, now, elapsed, w, h):
    state = suspicious_state.setdefault(slot, {
        'still_since':None,
        'smoothed_torso_motion':0.0,
        'last_center':None,
        'looking_since':None,
        'last_seen_looking':0.0,
        'hidden_since':None,
        'last_seen_hidden':0.0,
        'head_dir':'CENTER',
        'yaw_smoothed':0.0,
        'head_switches':[],
        'roi_events':[],
        'in_roi':False,
        'active':set(),
        'last_event':{},
        'crouch_frames':0,
        'height_history':[],
        'visible_frames_count':0,
        'was_lost':False,
    })
    
    if state.get('was_lost', False):
        state['visible_frames_count'] = 1
        state['was_lost'] = False
    else:
        state['visible_frames_count'] = state.get('visible_frames_count', 0) + 1

    checks = []

    center = body_center(pts)
    
    # 1) Standing Still (torso speed-based checks with low-pass EMA)
    torso_motion = 0.0
    if center and state.get('last_center'):
        lc = state['last_center']
        torso_motion = math.hypot(center[0] - lc[0], center[1] - lc[1])
    if center:
        state['last_center'] = center
        
    smoothed_torso_motion = state.get('smoothed_torso_motion', 0.0)
    smoothed_torso_motion = 0.2 * torso_motion + 0.8 * smoothed_torso_motion
    state['smoothed_torso_motion'] = smoothed_torso_motion
    
    if smoothed_torso_motion < 2.5:
        if state['still_since'] is None:
            state['still_since'] = now
    else:
        state['still_since'] = None
        
    if state['still_since'] and now - state['still_since'] >= STANDING_STILL_SECONDS:
        checks.append(('standing_still', 'Standing too long', 'high', RED))

    # 2) Looking at Camera (checks with a 1.2s grace period to absorb blinks/flicker)
    looking = is_looking_camera(landmarks, pts, w)
    if looking:
        if state['looking_since'] is None:
            state['looking_since'] = now
        state['last_seen_looking'] = now
    else:
        if state['looking_since'] is not None and now - state.get('last_seen_looking', 0) > 1.2:
            state['looking_since'] = None
            
    if state['looking_since'] and now - state['looking_since'] >= LOOKING_CAMERA_SECONDS:
        checks.append(('looking_camera', 'Looking at camera', 'medium_high', YELLOW))

    # 3) Face Hidden (checks with a 1.2s grace period to filter momentary hands occlusion)
    hidden = is_face_hidden(landmarks, pts, w, h)
    if hidden:
        if state['hidden_since'] is None:
            state['hidden_since'] = now
        state['last_seen_hidden'] = now
    else:
        if state['hidden_since'] is not None and now - state.get('last_seen_hidden', 0) > 1.2:
            state['hidden_since'] = None
            
    if state['hidden_since'] and now - state['hidden_since'] >= FACE_HIDDEN_SECONDS:
        checks.append(('face_hidden', 'Face hidden/not visible', 'medium_high', YELLOW))

    # 4) Frequent Looking Around
    asym = estimate_head_direction(landmarks, pts)
    smoothed_yaw = state.get('yaw_smoothed', 0.0)
    smoothed_yaw = 0.25 * asym + 0.75 * smoothed_yaw
    state['yaw_smoothed'] = smoothed_yaw
    
    current_head_dir = state.get('head_dir', 'CENTER')
    next_head_dir = current_head_dir
    
    if current_head_dir == 'CENTER':
        if smoothed_yaw > 0.24:
            next_head_dir = 'LEFT'
        elif smoothed_yaw < -0.24:
            next_head_dir = 'RIGHT'
    elif current_head_dir == 'LEFT':
        if smoothed_yaw < 0.12:
            next_head_dir = 'CENTER'
            if smoothed_yaw < -0.24:
                next_head_dir = 'RIGHT'
    elif current_head_dir == 'RIGHT':
        if smoothed_yaw > -0.12:
            next_head_dir = 'CENTER'
            if smoothed_yaw > 0.24:
                next_head_dir = 'LEFT'
                
    if next_head_dir != current_head_dir:
        state['head_dir'] = next_head_dir
        switches = state.setdefault('head_switches', [])
        switches.append(now)
        
    switches = state.setdefault('head_switches', [])
    switches = [t for t in switches if now - t <= 8.0]
    state['head_switches'] = switches
    
    if len(switches) >= 3:
        checks.append(('frequent_looking', 'Frequent looking around', 'medium_high', YELLOW))

    # 5) Repeated Entry / Exit Check
    if center:
        cx, cy = center
        rx1, rx2 = int(roi_x1 * w), int(roi_x2 * w)
        ry1, ry2 = int(roi_y1 * h), int(roi_y2 * h)
        
        inside = (rx1 <= cx <= rx2) and (ry1 <= cy <= ry2)
        
        if 'in_roi' not in state:
            state['in_roi'] = inside
        else:
            was_inside = state.get('in_roi', False)
            if inside != was_inside:
                state['in_roi'] = inside
                roi_events = state.setdefault('roi_events', [])
                roi_events.append(now)
                
                roi_events = [t for t in roi_events if now - t <= 45.0]
                state['roi_events'] = roi_events
                
                if len(roi_events) >= 4:
                    checks.append(('repeated_entry_exit', 'Repeated entry/exit (ROI)', 'high', RED))

    # 6) Crouching / Bending Down
    h_val = None
    if len(pts) > 24:
        vis_s11 = landmark_visibility(landmarks, 11)
        vis_s12 = landmark_visibility(landmarks, 12)
        vis_h23 = landmark_visibility(landmarks, 23)
        vis_h24 = landmark_visibility(landmarks, 24)
        if vis_s11 > 0.5 and vis_s12 > 0.5 and vis_h23 > 0.5 and vis_h24 > 0.5:
            d_left = math.hypot(pts[11][0] - pts[23][0], pts[11][1] - pts[23][1])
            d_right = math.hypot(pts[12][0] - pts[24][0], pts[12][1] - pts[24][1])
            h_val = (d_left + d_right) / 2.0

    was_crouching = 'crouching_bending' in state.get('active', set())
    if h_val is not None:
        history = state.setdefault('height_history', [])
        
        # Only update the height history if we weren't already crouching in the last frame
        if not was_crouching:
            history.append(h_val)
            if len(history) > 45:
                history.pop(0)
                
        if len(history) >= 10:
            baseline_h = max(history)
            # Check if height is reduced by CROUCH_HEIGHT_REDUCTION_THRESHOLD
            if h_val < (1.0 - CROUCH_HEIGHT_REDUCTION_THRESHOLD) * baseline_h:
                state['crouch_frames'] = state.get('crouch_frames', 0) + 1
            else:
                # If we were crouching, we need a clear signal of standing up to reset
                # (height returns to >= 85% of baseline)
                if was_crouching and h_val < 0.85 * baseline_h:
                    state['crouch_frames'] = max(state.get('crouch_frames', 0), CROUCH_CONSECUTIVE_FRAMES)
                else:
                    state['crouch_frames'] = 0
            
            if state.get('crouch_frames', 0) >= CROUCH_CONSECUTIVE_FRAMES:
                checks.append(('crouching_bending', 'Crouching / Bending Down', 'medium_high', YELLOW))

    # 7) Sudden Disappearance (confidence/visibility drop while still detected)
    vis_vals = [landmark_visibility(landmarks, i) for i in range(len(landmarks))]
    avg_vis = sum(vis_vals) / len(vis_vals) if vis_vals else 0.0
    
    prev_vis = state.get('last_avg_visibility', 1.0)
    state['last_avg_visibility'] = avg_vis
    
    if prev_vis > 0.70 and avg_vis < 0.35:
        if center:
            cx, cy = center
            border_x = 0.05 * w
            if border_x <= cx <= w - border_x:
                checks.append(('sudden_disappearance', 'Sudden Disappearance', 'high', RED))

    active_keys = {c[0] for c in checks}
    state['active'] = active_keys
    for key, label, risk, color in checks:
        last = state['last_event'].get(key, 0)
        if now - last >= EVENT_COOLDOWN_SECONDS:
            suspicious_events.append({
                'time':round(elapsed, 1),
                'person':slot + 1,
                'type':key,
                'label':label,
                'risk':risk,
            })
            state['last_event'][key] = now
    return checks

def build_suspicious_summary():
    summary = default_suspicious_summary()
    summary['total'] = len(suspicious_events)
    for event in suspicious_events:
        if event['risk'] == 'high':
            summary['high'] += 1
        else:
            summary['medium_high'] += 1
        if event['type'] in summary:
            summary[event['type']] += 1
    return summary

def draw_suspicious_alerts(frame, pts, alerts, slot, w, h):
    if not alerts or not pts:
        return
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x1 = max(min(xs) - 20, 0)
    y1 = max(min(ys) - 50, 0)
    x2 = min(max(xs) + 20, w - 1)
    y2 = min(max(ys) + 20, h - 1)
    top_color = RED if any(a[2] == 'high' for a in alerts) else YELLOW
    cv2.rectangle(frame, (x1, y1), (x2, y2), top_color, 3)
    for i, (_, label, risk, color) in enumerate(alerts[:3]):
        text = f"P{slot + 1}: {label} ({'HIGH' if risk == 'high' else 'MED-HIGH'})"
        ty = max(y1 - 10 - i * 25, 24)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
        cv2.rectangle(frame, (x1, ty - th - 8), (min(x1 + tw + 12, w - 1), ty + 6), (15, 15, 15), -1)
        cv2.putText(frame, text, (x1 + 6, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)

# ─────────────────────────────────────────────
# Threaded Camera to resolve latency/buffering
# ─────────────────────────────────────────────
class ThreadedCamera:
    def __init__(self, src=0, width=1280, height=720):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.grabbed, self.frame = self.cap.read()
        self.frame_id = 0
        self.started = False
        self.read_lock = threading.Lock()
        self.thread = None

    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            if grabbed:
                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = frame
                    self.frame_id += 1
            else:
                time.sleep(0.01)

    def read(self):
        with self.read_lock:
            frame_copy = self.frame.copy() if self.frame is not None else None
            return self.grabbed, frame_copy, self.frame_id

    def release(self):
        self.started = False
        if self.thread:
            try:
                self.thread.join(timeout=1.0)
            except:
                pass
        self.cap.release()

    def isOpened(self):
        return self.cap.isOpened()

# ─────────────────────────────────────────────
# Frame generator
# ─────────────────────────────────────────────
def generate_frames():
    global prev_pts_map, last_time, stats, cap, start_time, global_frame_ts_ms

    if start_time is None:
        start_time = time.time()
        stats['start_time'] = start_time

    if cap is None or not cap.isOpened():
        cap = ThreadedCamera(0).start()

    if not cap.isOpened():
        err = np.zeros((480,640,3),dtype=np.uint8)
        cv2.putText(err,"CAMERA NOT AVAILABLE",(80,240),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        _,buf = cv2.imencode('.jpg',err)
        fb = buf.tobytes()
        while True:
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + fb + b'\r\n'
            time.sleep(0.1)
        return

    print("Camera opened successfully.")
    last_frame_id = -1

    while True:
        try:
            ret, frame, frame_id = cap.read()
            if not ret:
                break

            if frame_id == last_frame_id:
                time.sleep(0.005)
                continue
            last_frame_id = frame_id

            now = time.time()
            dt  = max(now - last_time, 1e-6)
            last_time = now
            fps = 1.0 / dt
            elapsed = now - start_time

            h, w = frame.shape[:2]
            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_small = cv2.resize(rgb, (640, 360), interpolation=cv2.INTER_AREA)

            # ── Detect ──────────────────────────────
            raw_landmarks = []
            if use_old_api:
                res = pose.process(rgb_small)
                if res.pose_landmarks:
                    raw_landmarks.append(res.pose_landmarks.landmark)
            else:
                global_frame_ts_ms += max(int(dt*1000), 1)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_small)
                res = pose.detect_for_video(mp_img, global_frame_ts_ms)
                for plm in res.pose_landmarks:
                    raw_landmarks.append(plm)

            # ── Stable slot assignment ───────────────
            assigned = tracker.assign(raw_landmarks, w, h)

            # ── Draw ────────────────────────────────
            glow = np.zeros_like(frame)
            total_conf=0.; total_motion=0.; total_kp=0
            new_prev = {}
            current_alerts = []
            alerts_by_slot = {}
            active_slots = {slot for slot, _ in assigned}

            # ── Sudden Disappearance Check ──────────
            for slot_id in list(tracker.slots.keys()):
                if slot_id not in active_slots:
                    state = suspicious_state.get(slot_id)
                    if state:
                        state['was_lost'] = True
                        if tracker.lost_frames.get(slot_id, 0) == 1:
                            if state.get('visible_frames_count', 0) >= 3:
                                last_c = state.get('last_center')
                                if last_c:
                                    cx, cy = last_c
                                    border_x = 0.05 * w
                                    if border_x <= cx <= w - border_x:
                                        last_event_time = state.setdefault('last_event', {}).get('sudden_disappearance', 0)
                                        if now - last_event_time >= EVENT_COOLDOWN_SECONDS:
                                            suspicious_events.append({
                                                'time': round(elapsed, 1),
                                                'person': slot_id + 1,
                                                'type': 'sudden_disappearance',
                                                'label': 'Sudden Disappearance',
                                                'risk': 'high',
                                            })
                                            state['last_event']['sudden_disappearance'] = now

            # Clean up state for slots deleted from the tracker, triggering final ROI exit events
            for stale_slot in list(suspicious_state.keys()):
                if stale_slot not in tracker.slots:
                    st_val = suspicious_state.get(stale_slot)
                    if st_val and st_val.get('in_roi'):
                        roi_events = st_val.setdefault('roi_events', [])
                        roi_events.append(now)
                        st_val['in_roi'] = False
                        
                        roi_events = [t for t in roi_events if now - t <= 45.0]
                        st_val['roi_events'] = roi_events
                        
                        if len(roi_events) >= 4:
                            suspicious_events.append({
                                'time':round(elapsed, 1),
                                'person':stale_slot + 1,
                                'type':'repeated_entry_exit',
                                'label':'Repeated entry/exit (ROI)',
                                'risk':'high',
                            })
                    suspicious_state.pop(stale_slot, None)

            for slot, landmarks in assigned:
                pal = PERSON_PALETTES[slot % len(PERSON_PALETTES)]
                pts = draw_person(glow, landmarks, w, h, pal)
                new_prev[slot] = pts
                conf = float(getattr(landmarks[0],'visibility',0.)) if landmarks else 0.
                total_conf += conf; total_kp += len(landmarks)
                motion = 999.0
                if slot in prev_pts_map and len(prev_pts_map[slot])==len(pts):
                    motion = float(np.mean([
                        np.linalg.norm(np.array(pts[i])-np.array(prev_pts_map[slot][i]))
                        for i in range(len(pts))
                    ]))
                    total_motion += motion
                alerts = update_suspicious_activity(slot, landmarks, pts, motion, now, elapsed, w, h)
                alerts_by_slot[slot] = (pts, alerts)
                for key, label, risk, _ in alerts:
                    current_alerts.append({
                        'person':slot + 1,
                        'type':key,
                        'label':label,
                        'risk':risk,
                    })

            prev_pts_map = new_prev
            num = len(assigned)

            if num > 0:
                frame = cv2.addWeighted(frame, 0.55, glow, 0.9, 0)
                for slot, (pts, alerts) in alerts_by_slot.items():
                    draw_suspicious_alerts(frame, pts, alerts, slot, w, h)

            # ── Draw ROI boundaries (always visible) ─
            rx1 = int(roi_x1 * w)
            rx2 = int(roi_x2 * w)
            ry1 = int(roi_y1 * h)
            ry2 = int(roi_y2 * h)
            
            # Create semi-transparent overlay for ROI zone
            overlay = frame.copy()
            cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (0, 140, 255), -1)  # soft orange fill
            cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)
            
            # Draw ROI boundary line
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 140, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "ROI: ENTRY/EXIT ZONE", (rx1 + 10, ry1 + 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2, cv2.LINE_AA)

            # ── Person count badge ───────────────────
            badge_col = (30,200,30) if num>0 else (60,60,200)
            cv2.rectangle(frame,(w-170,8),(w-8,52),(255,255,255),-1)
            cv2.rectangle(frame,(w-170,8),(w-8,52),badge_col,2)
            cv2.putText(frame,f"Persons: {num}",(w-162,38),
                        cv2.FONT_HERSHEY_SIMPLEX,0.75,badge_col,2,cv2.LINE_AA)

            # ── Stats ────────────────────────────────
            avg_conf = total_conf/num if num else 0.
            summary = build_suspicious_summary()
            stats.update({
                'fps':round(fps,1),'confidence':round(avg_conf,3),
                'motion':round(total_motion,1),'keypoints':total_kp,
                'persons_detected':num,'camera_status':'active',
                'session_duration':round(elapsed,1),
                'suspicious_current':current_alerts,
                'suspicious_summary':summary,
                'suspicious_events':suspicious_events[-100:],
            })
            stats['fps_history'].append({'time':round(elapsed,2),'value':round(fps,1)})
            stats['confidence_history'].append({'time':round(elapsed,2),'value':round(avg_conf,3)})
            stats['motion_history'].append({'time':round(elapsed,2),'value':round(total_motion,1)})
            stats['persons_history'].append({'time':round(elapsed,2),'value':num})

            _,buf = cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,80])
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'

        except Exception as e:
            import traceback; traceback.print_exc()
            break

# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route('/')
def index():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'index.html')
    return send_file(p) if os.path.exists(p) else "<h1>Backend running</h1>"

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def get_stats():
    return jsonify(stats)

@app.route('/set_roi')
def set_roi():
    global roi_x1, roi_x2
    try:
        left = float(request.args.get('left', 35)) / 100.0
        right = float(request.args.get('right', 65)) / 100.0
        if 0.0 <= left < right <= 1.0:
            roi_x1 = left
            roi_x2 = right
            return jsonify({'status': 'ok', 'left': roi_x1, 'right': roi_x2})
        else:
            return jsonify({'status': 'error', 'message': 'Invalid boundaries'}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/test')
def test():
    return jsonify({
        'status':'ok','camera_available':camera_available,
        'multi_person': not use_old_api,'max_persons':MAX_PERSONS,
    })

@app.route('/reset')
def reset():
    global start_time, prev_pts_map, cap
    start_time = None; prev_pts_map = {}
    tracker.slots = {}
    tracker.smoothed_lms = {}
    tracker.lost_frames = {}
    stats['fps_history']=[]; stats['confidence_history']=[]
    stats['motion_history']=[]; stats['persons_history']=[]
    stats['session_duration']=0
    reset_suspicious_state()
    return jsonify({'status':'reset'})

if __name__ == '__main__':
    print("="*55)
    print("  Pose Tracker Backend — http://localhost:5001")
    print(f"  Multi-person: {'Yes (' + str(MAX_PERSONS) + ')' if not use_old_api else 'No (fallback)'}")
    print("="*55)
    try:
        app.run(debug=False, host='0.0.0.0', port=5001, threaded=True)
    except KeyboardInterrupt:
        if cap: cap.release()
        try: pose.close()
        except: pass
