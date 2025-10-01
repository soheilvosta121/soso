import os
import cv2
import pickle
import numpy as np
from sklearn.preprocessing import Normalizer
import mtcnn
from architecture import InceptionResNetV2
from config import SAVE_ENCODINGS_PATH, SAVED_FACES_PATH

class FaceEncoder:
    def __init__(self, model_weights_path: str):
        """
        Initializes the Keras-based face encoder using InceptionResNetV2.
        Args:
            model_weights_path: Path to the .h5 weights file for InceptionResNetV2.
        """
        self.required_size = (160, 160)
        # Load the custom architecture
        self.model = InceptionResNetV2()
        self.model.load_weights(model_weights_path)

        # MTCNN for face detection
        self.face_detector = mtcnn.MTCNN()

        # L2 normalizer
        self.l2_normalizer = Normalizer('l2')

    def normalize(self, img: np.ndarray) -> np.ndarray:
        """Zero-center and unit-scale the image."""
        mean, std = img.mean(), img.std()
        return (img - mean) / std

    def encode_face(self, face_img: np.ndarray) -> np.ndarray:
        """Process a single face crop and return its embedding."""
        face = self.normalize(face_img)
        face = cv2.resize(face, self.required_size)
        face_batch = np.expand_dims(face, axis=0)
        embedding = self.model.predict(face_batch)[0]
        return embedding

    def create_encodings(self, oval_faces_dir: str) -> str:
        """
        Iterate through saved face ovals directory, encode each face per person,
        average and normalize, then save embeddings to disk.
        Args:
            oval_faces_dir: Directory containing subfolders for each person with face images.
        Returns:
            Path to the last written encoding file.
        """
        encoding_save_file = ''
        for person_name in os.listdir(oval_faces_dir):
            person_dir = os.path.join(oval_faces_dir, person_name)
            encodes = []
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                img_BGR = cv2.imread(image_path)
                if img_BGR is None:
                    continue

                img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
                detections = self.face_detector.detect_faces(img_RGB)
                if not detections:
                    continue

                x, y, w, h = detections[0]['box']
                x, y = abs(x), abs(y)
                face = img_RGB[y:y+h, x:x+w]
                embedding = self.encode_face(face)
                encodes.append(embedding)

            if not encodes:
                continue
            
            #emb = np.median(encodes, axis=0)
            #emb = np.mean(encodes, axis=0)
            emb = np.sum(encodes, axis=0)
            normalized = self.l2_normalizer.transform(np.expand_dims(emb, axis=0))[0]
            encoding_dict = {person_name: normalized}

            # Save to individual file
            encoding_save_file = os.path.join(SAVE_ENCODINGS_PATH, f"{person_name}.pkl")
            os.makedirs(SAVE_ENCODINGS_PATH, exist_ok=True)
            with open(encoding_save_file, 'wb') as file:
                pickle.dump(encoding_dict, file)

        return encoding_save_file
