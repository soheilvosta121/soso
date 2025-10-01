import os
import cv2
import numpy as np
import mediapipe as mp
import math
import pandas as pd
from helpers import ai_helpers as aih
class FaceOvalExtractor:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=True
        )
        self.counter = 0
    def align_and_crop_face_oval(self, img_RGB):
        results = self.face_mesh.process(img_RGB)
        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0]
        left_eye = landmarks.landmark[234]
        right_eye = landmarks.landmark[454]

        left_eye_coord = (int(left_eye.x * img_RGB.shape[1]), int(left_eye.y * img_RGB.shape[0]))
        right_eye_coord = (int(right_eye.x * img_RGB.shape[1]), int(right_eye.y * img_RGB.shape[0]))
        h, w, _ = img_RGB.shape

        # validate eye coordinates and distance
        if not (0 < left_eye_coord[0] < w and 0 < left_eye_coord[1] < h and
                0 < right_eye_coord[0] < w and 0 < right_eye_coord[1] < h):
            return None
        eye_dist = np.linalg.norm(np.array(left_eye_coord) - np.array(right_eye_coord))
        if eye_dist < 15 or eye_dist > 300:
            return None

        # align face by eyes
        angle = math.degrees(math.atan2(right_eye_coord[1] - left_eye_coord[1],
                                        right_eye_coord[0] - left_eye_coord[0]))
        center = ((left_eye_coord[0] + right_eye_coord[0]) // 2,
                (left_eye_coord[1] + right_eye_coord[1]) // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned_face = cv2.warpAffine(img_RGB, M, (w, h))

        # re-detect landmarks on aligned face
        aligned_res = self.face_mesh.process(aligned_face)
        if not aligned_res.multi_face_landmarks:
            return None
        aln_lms = aligned_res.multi_face_landmarks[0]

        # build face oval contour points
        oval_idx = mp.solutions.face_mesh.FACEMESH_FACE_OVAL
        df = pd.DataFrame(oval_idx, columns=["p1","p2"])
        routes = []
        p2 = df.iloc[0]["p2"]
        for _ in range(len(df)):
            row = df[df["p1"] == p2].iloc[0]
            p2 = row["p2"]
            lm = aln_lms.landmark[row["p1"]]
            routes.append((int(w * lm.x), int(h * lm.y)))
        contour = np.array(routes, dtype=np.int32).reshape(-1,1,2)

        # mask and convexity check
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [contour], 255)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        if not cv2.isContourConvex(approx):
            return None

        # extract only the face oval region
        ys, xs = np.where(mask == 255)
        if xs.size == 0 or ys.size == 0:
            return None
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()
        face_rect = aligned_face[y_min:y_max+1, x_min:x_max+1]
        mask_crop = mask[y_min:y_max+1, x_min:x_max+1]
        oval_face = cv2.bitwise_and(face_rect, face_rect, mask=mask_crop)

        return oval_face


    def align_and_crop_face_org(self, img_RGB):
        results = self.face_mesh.process(img_RGB)
        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0]
        left_eye = landmarks.landmark[234]
        right_eye = landmarks.landmark[454]

        left_eye_coord = (int(left_eye.x * img_RGB.shape[1]), int(left_eye.y * img_RGB.shape[0]))
        right_eye_coord = (int(right_eye.x * img_RGB.shape[1]), int(right_eye.y * img_RGB.shape[0]))
        # Check coordinates are within frame bounds
        h, w, _ = img_RGB.shape
        coords_valid = all([
            0 < left_eye_coord[0] < w,
            0 < left_eye_coord[1] < h,
            0 < right_eye_coord[0] < w,
            0 < right_eye_coord[1] < h
        ])

        if not coords_valid:
            return None
        eye_distance = np.linalg.norm(np.array(left_eye_coord) - np.array(right_eye_coord))

        # Empirical range (adjust according to your typical face size)
        if eye_distance < 15 or eye_distance > 300:
            return None
        angle = math.degrees(math.atan2(right_eye_coord[1] - left_eye_coord[1],
                                        right_eye_coord[0] - left_eye_coord[0]))
        center = ((left_eye_coord[0] + right_eye_coord[0]) // 2,
                  (left_eye_coord[1] + right_eye_coord[1]) // 2)

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned_face = cv2.warpAffine(img_RGB, rotation_matrix, (img_RGB.shape[1], img_RGB.shape[0]))

        aligned_results = self.face_mesh.process(aligned_face)
        if not aligned_results.multi_face_landmarks:
            return None

        aligned_landmarks = aligned_results.multi_face_landmarks[0]
        face_oval_idxs = mp.solutions.face_mesh.FACEMESH_FACE_OVAL
        df = pd.DataFrame(face_oval_idxs, columns=["p1", "p2"])

        routes_idx = []
        p2 = df.iloc[0]["p2"]
        for _ in range(len(df)):
            row = df[df["p1"] == p2].iloc[0]
            routes_idx.append([row["p1"], row["p2"]])
            p2 = row["p2"]

        routes = []
        for source_idx, target_idx in routes_idx:
            source = aligned_landmarks.landmark[source_idx]
            target = aligned_landmarks.landmark[target_idx]

            source_px = (int(aligned_face.shape[1] * source.x), int(aligned_face.shape[0] * source.y))
            target_px = (int(aligned_face.shape[1] * target.x), int(aligned_face.shape[0] * target.y))

            routes.extend([source_px, target_px])

        mask = np.zeros(aligned_face.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.array(routes), 255)

        #contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #if not contours or not cv2.isContourConvex(contours[0]):
        #    return None  # Skip if not convex
        cv2.imshow("Raw Face Mask", mask)
        cv2.waitKey(1)
        oval_face = cv2.bitwise_and(aligned_face, aligned_face, mask=mask)
        ys, xs = np.where(mask == 255)
        x_min, x_max, y_min, y_max = xs.min(), xs.max(), ys.min(), ys.max()
        x_min_enlarged, y_min_enlarged, x_max_enlarged, y_max_enlarged = aih.enlarge_bbox((x_min, y_min, x_max, y_max), (oval_face.shape[0], oval_face.shape[1]), enlargement_percentage=0.2)
        oval_face_cropped = oval_face[y_min_enlarged:y_max_enlarged, x_min_enlarged:x_max_enlarged]

        return oval_face_cropped
        
    def extract_from_detected_faces(self, frame_bgr, face_bboxes, save_dir, video_id="unknown", face_id= "unknown"):
        os.makedirs(save_dir, exist_ok=True)
        person_folder = os.path.join(save_dir, video_id, face_id)
        os.makedirs(person_folder, exist_ok=True)
        cropped_faces = []
        img_RGB = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        for i, (x1, y1, x2, y2) in enumerate(face_bboxes):
            face_org = img_RGB[y1:y2, x1:x2]
            cropped_faces.append(face_org)
            
            ## Face augmentation
            face_flipped = aih.face_flipped_img(face_org)
            cropped_faces.append(face_flipped) if face_flipped is not None else None
            face_rotated = aih.face_rotated_imgs(face_org)
            for rotated in face_rotated:
                if rotated is not None:
                    cropped_faces.append(rotated)

            #cropped_faces.append(face_rotated) if face_rotated is not None else None
            #if face_crop.size == 0 or face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
                #continue
            #cv2.imshow("Face Crop", face_crop)
            #cv2.waitKey(1)
            for face_crop in cropped_faces:
                if face_crop.size == 0 or face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
                    continue
                aligned_oval = self.align_and_crop_face_oval(face_crop)
                if aligned_oval is not None:
                    save_path = os.path.join(person_folder, f"{video_id}_{self.counter}_{face_id}.jpg")
                    cv2.imwrite(save_path, cv2.cvtColor(aligned_oval, cv2.COLOR_RGB2BGR))
                    self.counter += 1
