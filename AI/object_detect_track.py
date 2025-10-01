import sys
import os
import torch
import cv2
import numpy as np
import helpers.ai_helpers as aih

# Add the root directory to the Python path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'AI/'))
sys.path.insert(0, root_dir)

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from config import object_detection_list  

class DetectedObject:
    def __init__(self, name, center, confidence, bbox):
        self.name = name
        self.center = center
        self.confidence = confidence
        self.bbox = bbox  # Bounding box (x1, y1, x2, y2)


class ObjDetectTrack:
    def __init__(self, weights_path, conf_thres=0.3, iou_thres=0.45):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model = torch.load(weights_path, map_location=self.device, weights_only=False)['model'].float().eval()
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.trackers = []  # List to hold trackers for each detected object
        self.object_id_counter = 0  # Counter to assign unique IDs
        self.tracked_objects = {}  # Dictionary to store tracked objects with IDs
        self.object_trails = {}  # new: stores center history for each object

    def detect_from_image(self, image_source, label_list=None, output_image_path=None):
        image = image_source.copy()
        original_h, original_w = image.shape[:2]

        resized_h = (original_h // 32) * 32
        resized_w = (original_w // 32) * 32
        img_resized = cv2.resize(image, (resized_w, resized_h))
        #resized_h, resized_w = 320, 320
        #img_resized = cv2.resize(image, (resized_w, resized_h))
        #img_rc = img_resized.copy()

        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().to(self.device) / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.model(img_tensor)[0]

        pred = non_max_suppression(pred, conf_thres=self.conf_thres, iou_thres=self.iou_thres)

        detected_objects = []
        bboxes = []

        for det in pred:
            if len(det):
                det = det.cpu()
                det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], (original_h, original_w)).round()


                for *xyxy, conf, cls in det:
                    name = self.model.names[int(cls)]
                    
                    if name not in object_detection_list:
                        continue

                    x1, y1, x2, y2 = map(int, xyxy)
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

                    # Assign an ID to the detected object
                    object_id = self._get_object_id((x1, y1, x2, y2))
                    # Update trail history
                    if object_id not in self.object_trails:
                        self.object_trails[object_id] = []
                    self.object_trails[object_id].append((center_x, center_y))

                    # Optional: Limit trail length
                    #self.object_trails[object_id] = self.object_trails[object_id][-200:]
                    # Draw bounding box, label, and ID
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    #cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    

                    label_display = f'{name} {object_id}: {conf:.2f}'
                    cv2.putText(image, label_display, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    # Draw trail line
                    trail = self.object_trails[object_id]
                    for i in range(1, len(trail)):
                        cv2.line(image, trail[i - 1], trail[i], (0, 255, 255), 2)
                    person_bbox = aih.enlarge_bbox((x1, y1, x2, y2), (original_h, original_w), enlargement_percentage=0.1)

                    # Store detected object
                    detected_objects.append(DetectedObject(name, (center_x, center_y), float(conf), person_bbox))
                    bboxes.append((x1, y1, x2, y2))  # Add bbox to the list

                    # Update tracked objects
                    self.tracked_objects[object_id] = ((x1, y1, x2, y2), (center_x, center_y))
        #cv2.imshow("ai view", img_rc)
        

        return self.tracked_objects, detected_objects, bboxes, image

    def _get_object_id(self, bbox):
        """
        Assign a unique ID to a detected object or match it with an existing tracked object.
        """
        x1, y1, x2, y2 = bbox
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        # Match with existing tracked objects based on proximity
        for object_id, (tracked_bbox, tracked_center) in self.tracked_objects.items():
            tracked_x1, tracked_y1, tracked_x2, tracked_y2 = tracked_bbox
            tracked_center_x, tracked_center_y = tracked_center

            # Calculate distance between centers
            distance = ((center_x - tracked_center_x) ** 2 + (center_y - tracked_center_y) ** 2) ** 0.5
            if distance < 50:  # Threshold for matching 
                return object_id

        # If no match, assign a new ID
        self.object_id_counter += 1
        return self.object_id_counter
    
    