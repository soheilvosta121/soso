import torch
import cv2
import numpy as np
from config import POSE_ESTIMATION_MODEL
from torchvision import transforms
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords, non_max_suppression_kpt
from utils.plots import plot_one_box, output_to_keypoint, plot_skeleton_kpts
import helpers.draw_rectangle as rec
dim_multiplier = 10
dimension_width = 32 * dim_multiplier
dimension_height = 32 * dim_multiplier

# Define keypoint indices for specific body parts 
NOSE_IDX = 0  
LEFT_EYE_IDX = 1  
RIGHT_EYE_IDX = 2 
LEFT_EAR_IDX = 3  
RIGHT_EAR_IDX = 4
LEFT_SHOULDER_IDX = 5
RIGHT_SHOULDER_IDX = 6
LEFT_ELBOW_IDX = 7
RIGHT_ELBOW_IDX = 8 
LEFT_HAND_IDX = 9 
RIGHT_HAND_IDX = 10
LEFT_HIP_IDX = 11
RIGHT_HIP_IDX = 12
LEFT_KNEE_IDX = 13
RIGHT_KNEE_IDX = 14
LEFT_FOOT_IDX = 15
RIGHT_FOOT_IDX = 16 
       

#HEAD_IDX = [NOSE_IDX, LEFT_EYE_IDX, RIGHT_EYE_IDX, LEFT_EAR_IDX, RIGHT_EAR_IDX]
body_parts = {
#"Head": HEAD_IDX,
"Nose": NOSE_IDX,
"Left Eye": LEFT_EYE_IDX,
"Right Eye": RIGHT_EYE_IDX,
"Left Ear": LEFT_EAR_IDX,
"Right Ear": RIGHT_EAR_IDX,
"Left Shoulder": LEFT_SHOULDER_IDX,
"Right Shoulder": RIGHT_SHOULDER_IDX,
"Left Elbow": LEFT_ELBOW_IDX,
"Right Elbow": RIGHT_ELBOW_IDX,
"Left Hand": LEFT_HAND_IDX,
"Right Hand": RIGHT_HAND_IDX,
"Left Hip": LEFT_HIP_IDX,
"Right Hip": RIGHT_HIP_IDX,
"Left Knee": LEFT_KNEE_IDX,
"Right Knee": RIGHT_KNEE_IDX,
"Left Foot": LEFT_FOOT_IDX,
"Right Foot": RIGHT_FOOT_IDX}

class PoseEstimation:
    def __init__(self, model_path=POSE_ESTIMATION_MODEL, conf_threshold=0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        weights_pose = torch.load(model_path, map_location=self.device)
        self.model_pose = weights_pose['model']
        self.model_pose.float().eval()
        self.conf_threshold = conf_threshold

    def preprocess_for_pose(self, frame):
        self.orig_w, self.orig_h = frame.shape[1], frame.shape[0]

        # Resize the frame to match the model's input requirements
        frame_resized = cv2.resize(frame, (dimension_height, dimension_width))
        # Convert to tensor
        image_tensor = transforms.ToTensor()(frame_resized).unsqueeze(0)
        return image_tensor

    def extract_kp(self, keypoints, idx):
        """
        Returns (x, y, conf) for keypoint index `idx`,
        whether keypoints is shape (N,3) or a flat (3N,) array.
        """
        if keypoints.ndim == 2:
            # shape (N,3)
            return keypoints[idx]
        else:
            # flat shape (3N,)
            base = idx * 3
            return (
                keypoints[base],
                keypoints[base + 1],
                keypoints[base + 2],
            )

    def detect_pose(self, frame):
        # Resize frame for pose estimation
        frame_resized_for_pose = cv2.resize(frame, (dimension_height, dimension_width))  # Resize to match model requirements

        # Preprocess the frame
        image_tensor = self.preprocess_for_pose(frame_resized_for_pose)
        if torch.cuda.is_available():
            image_tensor = image_tensor.float().to(self.device)

        # Perform inference
        with torch.no_grad():
            output, _ = self.model_pose(image_tensor)

        # Apply Non-Max Suppression and convert output to keypoints
        output = non_max_suppression_kpt(
            output, conf_thres=0.3, iou_thres=0.5, nc=self.model_pose.yaml['nc'], nkpt=self.model_pose.yaml['nkpt'], kpt_label=True
        )
        output = output_to_keypoint(output)

        # Copy the resized frame for skeleton plotting
        frame_copy = frame_resized_for_pose.copy()

        # Initialize a dictionary to store keypoints for each body part
        keypoints_dict = {part_name: None for part_name in body_parts.keys()}

        for person_i in range(output.shape[0]):
            keypoints = output[person_i, 7:].T
            plot_skeleton_kpts(frame_copy, keypoints, 3)

            for part_name, idx_or_list in body_parts.items():
                x, y, conf = self.extract_kp(keypoints, idx_or_list)
                if conf > self.conf_threshold:
                    # Save the keypoint for the current body part
                    keypoints_dict[part_name] = (x, y, conf)
        frame_copy, head_bbox = rec.draw_head_square(frame_copy, keypoints_dict, expansion=2, thickness=2)
        print("Head BBox:", head_bbox)
        frame_copy, body_boxes = rec.draw_pose_rectangles(frame_copy, keypoints_dict, half_width_percentage=2)

        bboxes = []
        frame_h, frame_w = frame_copy.shape[:2]
        if head_bbox:
            # Clip head_bbox to frame boundaries
            hx0, hy0, hx1, hy1 = head_bbox
            hx0 = max(0, min(hx0, frame_w - 1))
            hy0 = max(0, min(hy0, frame_h - 1))
            hx1 = max(0, min(hx1, frame_w - 1))
            hy1 = max(0, min(hy1, frame_h - 1))
            bboxes.append({'segment': 'head', 'bbox': (hx0, hy0, hx1, hy1)})
        # Append other body parts boxes
        for box in body_boxes:
            points = box['points']
            x_coords, y_coords = zip(*points)
            x0, y0, x1, y1 = min(x_coords), min(y_coords), max(x_coords), max(y_coords)
            # Clip to frame boundaries
            x0 = max(0, min(x0, frame_w - 1))
            y0 = max(0, min(y0, frame_h - 1))
            x1 = max(0, min(x1, frame_w - 1))
            y1 = max(0, min(y1, frame_h - 1))
            bboxes.append({'segment': box['segment'], 'bbox': (x0, y0, x1, y1)})
            #self.object_vs_label_mask(bboxes[box]['bbox']])


        """ 
        has_face_kpts = (
            keypoints_dict["Nose"] is not None and
            keypoints_dict["Left Eye"] is not None and
            keypoints_dict["Right Eye"] is not None
        )
        frame_copy, bboxes = rec.draw_pose_rectangles(frame_copy, keypoints_dict)"""


        frame_copy = cv2.resize(frame_copy, (frame.shape[1], frame.shape[0])) 
        print("Pose input   size:", frame_resized_for_pose.shape[:2])


        return frame_copy, keypoints_dict, bboxes, (dimension_height, dimension_width)#, has_face_kpts

