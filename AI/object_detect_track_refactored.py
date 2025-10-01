import cv2
import numpy as np
import os
import time
import threading
from datetime import datetime
import config as cf
from vispy import scene, app
from types import SimpleNamespace

import helpers.ai_helpers as aih

import AI.background_reconstruction as br
import AI.motion_detection as md
import config as cf
from vispy import scene, app
import AI.object_detect_track as ot
import AI.pose_estimation as pe
import AI.face_detection as fd

import algorithms.histogram as hist
import algorithms.histogram_one_chanel as hist_oc

import objects.image_secgment_object as iso
import objects.motion_object as motion_obj
import ui.object_traking_window as otw
import servises.image_seg_service as iss

# ─── CONFIGURATION ────────────────────────────────────────────────────
MAX_BUFFER      = 50
FPS             = 15
RTSP_DEVICE     = "/dev/video0"
is_webcam       = True
rtsp_url = cf.RTSP_URL_PATH
VIDEO_PATH      = rtsp_url
POINT_SIZE      = 3
HIST_BINS       = 16
HIST_THRESH     = 5
HIST_RANGE = [0,256] * 3

_RES_OPTIONS = {
    "180": (320, 180),
    "360": (640, 360),
    "720": (1280, 720),
    "1080": (1920, 1080)
}
RES_TO_DISPLAY = ["360"]  # Change as needed
kernel = np.ones((3, 3), np.uint8)
object_traking_window = otw.TableDisplay()

render_lock = threading.Lock()

histogram_service = hist.HistogramSevice()
histogram_service_one_channel = hist_oc.OneChannel_HistogramSevice()

object_detector = ot.ObjDetectTrack(cf.OBJECT_DETECTION_MODEL)
pose_estimator = pe.PoseEstimation(cf.POSE_ESTIMATION_MODEL)
face_detector = fd.FaceDetectorMTCNN()

segmentation_obj = iso.segmentation_object()

frame_buffer = []
buffer_lock = threading.Lock()
frame_od = None

motion_obj_list = []

latest = {
    'processed': None,
    'background': None,
    'mask': None,
    'points_bg': np.zeros((0,3), dtype=np.float32),
    'size_bg':   np.zeros((0,), dtype=np.float32),
    'color_bg':  np.zeros((0,4), dtype=np.float32),
    'points_motion': np.zeros((0,3), dtype=np.float32),
    'size_motion':   np.zeros((0,), dtype=np.float32),
    'color_motion':  np.zeros((0,4), dtype=np.float32),
    'profiling': {}
}
state_lock = threading.Lock()

countur_img = None
roi_coords_left = None
roi_coords_right = None
LEFT_CLICK_RADIUS  = 25
RIGHT_CLICK_RADIUS = 25
update_hist_dist_flag = False

def on_mouse(event, x, y, flags, param):
    global roi_coords_left, roi_coords_right, update_hist_dist_flag
    if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
        return

    res = RES_TO_DISPLAY[0]
    frame_mouse = frame_to_display_list.get(res, {}).get("current frame", None)
    if frame_mouse is None:
        return

    radius = LEFT_CLICK_RADIUS if event == cv2.EVENT_LBUTTONDOWN else RIGHT_CLICK_RADIUS
    update_hist_dist_flag = True

    if event == cv2.EVENT_LBUTTONDOWN:
        roi_coords_left = (x, y, radius)
    else:
        roi_coords_right = (x, y, radius)

win_name = f"{RES_TO_DISPLAY[0]}: current frame"
cv2.namedWindow(win_name)
cv2.setMouseCallback(win_name, on_mouse)

# ─── PER-RESOLUTION INSTANCES ────────────────────────────────────────
background_builder_list = {}
detector_list = {}
frame_to_display_list = {}
histogram_data = {}
for resolution in RES_TO_DISPLAY:
    histogram_data[resolution] = None
    background_builder_list[resolution] = br.BackgroundReconstruction()
    detector_list[resolution] = md.MotionDetection()
    detector_list[resolution].motion_start_threshold = 10
    detector_list[resolution].motion_mask_threshold = 10

def resize_with_aspect_ratio(image, target_width, target_height):
    h, w = image.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    result = np.zeros((target_height, target_width, 3), dtype=resized.dtype)
    y_offset = (target_height - new_h) // 2
    x_offset = (target_width - new_w) // 2
    result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return result

countur_img = None

# ─── NEW MOTION TRACKER CLASS ─────────────────────────────────────────
class MotionTracker:
    def __init__(self, scale_factor: float = 0.05, min_size: int = 20, overlap_thres: float = 0.1, 
                 correlation_filter_thres: float = 0.9, max_frames_inactive: int = 10):
        self.scale_factor = scale_factor  # 1/20 = 0.05
        self.min_size = min_size
        self.overlap_thres = overlap_thres  # Minimum overlap to match
        self.correlation_filter_thres = correlation_filter_thres  # Filter out if correlation > this (noise)
        self.max_frames_inactive = max_frames_inactive
        self.tracked_objects = []  # List of motion_obj.motion_object
        self.object_id_counter = 0

    def track_motion_blobs(self, motion_blobs: list[np.ndarray], frame: np.ndarray, bg: np.ndarray) -> list[motion_obj.motion_object]:
        """
        Track motion blobs across frames using overlap matching and correlation filtering.
        Returns updated list of tracked objects.
        """
        new_objects = []
        current_ids = set()  # Updated this frame
        for blob in motion_blobs:
            h, w = blob.shape[:2]
            scaled = cv2.resize(blob, (int(w * self.scale_factor), int(h * self.scale_factor)), cv2.INTER_MAX)
            non_zero_coords = np.nonzero(scaled)
            if len(non_zero_coords[0]) == 0:
                continue

            nonzero_full = np.nonzero(blob)
            min_x, min_y = min(nonzero_full[1]), min(nonzero_full[0])
            max_x, max_y = max(nonzero_full[1]), max(nonzero_full[0])
            max_dim = max(max_x - min_x, max_y - min_y)
            if max_dim <= self.min_size:
                continue

            subimage = frame[min_y:max_y, min_x:max_x]
            sub_bg = bg[min_y:max_y, min_x:max_x]
            _, sub_mask = cv2.threshold(blob[min_y:max_y, min_x:max_x], 50, 255, cv2.THRESH_BINARY)

            subimage_lab = cv2.cvtColor(subimage, cv2.COLOR_BGR2LAB)
            sub_bg_lab = cv2.cvtColor(sub_bg, cv2.COLOR_BGR2LAB)

            l1 = cv2.normalize(cv2.calcHist([subimage_lab], [0], sub_mask, [64], [0,256]), None, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
            a1 = cv2.normalize(cv2.calcHist([subimage_lab], [1], sub_mask, [64], [0,256]), None, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
            b1 = cv2.normalize(cv2.calcHist([subimage_lab], [2], sub_mask, [64], [0,256]), None, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)

            bl = cv2.normalize(cv2.calcHist([sub_bg_lab], [0], sub_mask, [64], [0,256]), None, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
            ba = cv2.normalize(cv2.calcHist([sub_bg_lab], [1], sub_mask, [64], [0,256]), None, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
            bb = cv2.normalize(cv2.calcHist([sub_bg_lab], [2], sub_mask, [64], [0,256]), None, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)

            lc = cv2.compareHist(l1, bl, cv2.HISTCMP_CORREL)
            ac = cv2.compareHist(a1, ba, cv2.HISTCMP_CORREL)
            bc = cv2.compareHist(b1, bb, cv2.HISTCMP_CORREL)
            avg_correlation = (lc + ac + bc) / 3

            if avg_correlation > self.correlation_filter_thres:
                continue  # Filter as background noise

            motion_grid = list(zip(non_zero_coords[1], non_zero_coords[0]))  # Scaled (x,y)

            # Find best match by max overlap
            best_overlap = 0
            best_obj = None
            for track in self.tracked_objects:
                overlap = motion_obj.get_overlaping(track.screen_position, motion_grid)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_obj = track

            if best_obj and best_overlap > self.overlap_thres:
                # Update best match
                best_obj.screen_position = motion_grid
                best_obj.mask = scaled
                best_obj.type = avg_correlation
                best_obj.corelation = avg_correlation
                best_obj.image_to_display = cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)
                best_obj.sqear = ((min_x, min_y), (max_x, max_y))
                best_obj.creation_dt = datetime.now()
                best_obj.frame_counter += 1
                best_obj.frames_inactive = 0  # Reset
                current_ids.add(id(best_obj))  # Use Python id or add custom ID
            else:
                # New track
                new_track = motion_obj.motion_object(scaled, motion_grid)
                new_track.type = avg_correlation
                new_track.corelation = avg_correlation
                new_track.sqear = ((min_x, min_y), (max_x, max_y))
                new_track.frames_inactive = 0
                new_track.id = self.object_id_counter  # Add ID if not in class
                self.object_id_counter += 1
                new_objects.append(new_track)

        # Add new objects
        self.tracked_objects.extend(new_objects)

        # Update inactive and expire
        to_remove = []
        for track in self.tracked_objects:
            if id(track) in current_ids:
                track.frames_inactive = 0
            else:
                track.frames_inactive += 1
                if track.frames_inactive > self.max_frames_inactive:
                    to_remove.append(track)

        for track in to_remove:
            self.tracked_objects.remove(track)

        return self.tracked_objects

# ─── PER-RESOLUTION MOTION TRACKERS ───────────────────────────────────
motion_tracker_list = {}
for resolution in RES_TO_DISPLAY:
    motion_tracker_list[resolution] = MotionTracker()

# ─── WORKER THREAD ───────────────────────────────────────────────────
def ai_service():
    global background_builder_list, segmentation_obj, update_hist_dist_flag, histogram_service_one_channel
    global frame_pe
    global frame_fd
    global countur_img
    global motion_obj_list
    frame_pe = None
    frame_fd = None
    while True:
        with buffer_lock:
            if not frame_buffer:
                time.sleep(0.01)
                continue
            timestamp, frame = frame_buffer[-1]
        profiling = {}

        t0 = time.time()
        for resolution in RES_TO_DISPLAY:
            w, h = _RES_OPTIONS[resolution]
            frame_local = resize_with_aspect_ratio(frame, w, h)
            background_builder = background_builder_list[resolution]
            # 1) Background reconstruction
            bg_rgba = background_builder.background_builder(frame_local, [])
            thresh = None
            if len(background_builder.backgroud_histoy) > 4:
                diff = cv2.absdiff(background_builder.backgroud_histoy[-5], background_builder.backgroud_histoy[-1])
                diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)

            if bg_rgba.ndim == 2:
                bg = cv2.cvtColor(bg_rgba, cv2.COLOR_GRAY2BGR)
            elif bg_rgba.shape[2] == 4:
                bg = cv2.cvtColor(bg_rgba, cv2.COLOR_BGRA2BGR)
            else:
                bg = bg_rgba.astype(np.uint8)
            profiling['bg'] = time.time() - t0

            # 2) Motion detection
            t1 = time.time()
            frame_motion = frame_local.copy()
            proc, mask, flag, color, countur_img, motion_object_list = detector_list[resolution].detect_motion(frame_motion, bg, [])
            profiling['detect'] = time.time() - t1

            # 3) Motion tracking
            t_track = time.time()
            motion_tracker = motion_tracker_list[resolution]
            mol_updated = motion_tracker.track_motion_blobs(motion_object_list, frame_local, bg)
            profiling['track'] = time.time() - t_track

            # 4) Histograms for background vs motion
            t2 = time.time()
            histogram_data["frame"] = histogram_service.get_hist(frame_local)
            subframe_roi_l = None
            subframe_roi_r = None
            img_hsl_l = None
            img_hsl_r = None 
            if roi_coords_left is not None: 
                x, y, r = roi_coords_left
                ih, iw = frame_local.shape[:2]
                sy = max(0, y - r)
                ey = min(ih - 1, y + r)
                sx = max(0, x - r)
                ex = min(iw - 1, x + r)
                subframe_roi_l = frame_local[sy:ey, sx:ex]
                img_hsl_l = cv2.cvtColor(subframe_roi_l, cv2.COLOR_BGR2HLS)
                bg_sub = bg[sy:ey, sx:ex]
                img_hls_b = cv2.cvtColor(bg_sub, cv2.COLOR_BGR2HLS)

            if roi_coords_right is not None: 
                x, y, r = roi_coords_right
                subframe_roi_r = frame_local[y-r:y+r, x-r:x+r]
                img_hsl_r = cv2.cvtColor(subframe_roi_r, cv2.COLOR_BGR2HLS)

            if update_hist_dist_flag and roi_coords_left:
                histogram_service_one_channel.calc_hist([img_hsl_l, img_hls_b])
                histogram_service_one_channel.find_distances_to_first_element()
                histogram_service_one_channel.draw_all()

            profiling['hist'] = time.time() - t2

            # Visualization: Draw on motion_filtered_image
            motion_filtered_image = frame_local.copy()
            for obj in mol_updated:
                if obj.frame_counter > 5:
                    if 0.80 < obj.corelation <= 0.90:
                        cv2.rectangle(motion_filtered_image, obj.sqear[0], obj.sqear[1], (0, 255, 255), thickness=1)
                    if obj.corelation <= 0.80: 
                        cv2.rectangle(motion_filtered_image, obj.sqear[0], obj.sqear[1], (0, 0, 255), thickness=1)

            with render_lock:
                motion_obj_list = mol_updated
                frame_to_display_list[resolution] = {
                    "motion": frame_motion, 
                    "mask": mask, 
                    "bckground": bg_rgba, 
                    "current frame": frame_local, 
                    "sf1": subframe_roi_l, 
                    "sf2": subframe_roi_r, 
                    "motion filtered": motion_filtered_image
                }

        # 5) Update shared state
        with state_lock:
            latest.update({
                'processed': proc,
                'background': bg,
                'mask': mask,
                'profiling': profiling
            })

threading.Thread(target=ai_service, daemon=True).start()

# ─── DISPLAY FUNCTION ────────────────────────────────────────────────
def render():
    start_time = time.perf_counter()
    object_traking_window.update_display(motion_obj_list)
    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f"Code took {duration:.6f} seconds to execute")

    with render_lock:
        for resolution in RES_TO_DISPLAY:
            if resolution in frame_to_display_list:
                img_list = frame_to_display_list[resolution]
                for img in img_list:
                    if img_list[img] is not None:
                        cv2.imshow(f"{resolution}: {img}", img_list[img])
    histogram_service_one_channel.render()

# ─── MAIN LOOP ───────────────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_PATH, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("Error opening video source")
    exit(1)
ret, frame = cap.read()
iss.init_seg_obg(segmentation_obj, frame)
cv2.imshow("segmentation", segmentation_obj.image_with_counturs)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    timestamp = datetime.now()
    with buffer_lock:
        frame_buffer.append((timestamp, frame))
        if len(frame_buffer) > MAX_BUFFER:
            frame_buffer.pop(0)
    with state_lock:
        proc = latest['processed']
        bg   = latest['background']
        msk  = latest['mask']
        p_bg, s_bg, c_bg = latest['points_bg'], latest['size_bg'], latest['color_bg']
        p_m, s_m, c_m   = latest['points_motion'], latest['size_motion'], latest['color_motion']
        profiling = latest.get('profiling', {})
    render()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()