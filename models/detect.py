import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import PIL.Image as Image
from utils_img import draw_bounding_boxes_on_image
from models.base_model import BaseModel


class Detection(BaseModel):
    def __init__(self, hub_handle, confidence=0.5):
        self._hub_handle = hub_handle
        self._model_loaded = False
        self._confidence = confidence

    def _load_model(self):
        print("Initializing model...")
        self.model = hub.load(self._hub_handle, tags=[])
        print("Model initialized")
        self._model_loaded = True

    def __call__(self, img):
        if not self._model_loaded:
            self._load_model()
        return self._detect(img)

    def _detect(self, img):
            image = tf.image.convert_image_dtype(img, dtype=tf.float32)

            image = tf.expand_dims(image, 0)

            detector_output = self.model.signatures["default"](image)
            boxes = detector_output["detection_boxes"]
            scores = detector_output["detection_scores"]
            cis = detector_output["detection_class_entities"]

            #print(cis)
            #print(scores)

            filt_boxes = []
            filt_labels = []
            for b, s, l in zip(boxes, scores, cis):
                if s >= self._confidence:
                    filt_boxes.append(b)
                    filt_labels.append([l.numpy().decode('UTF-8')])

            if not filt_boxes:
                print(img_path)
                raise ValueError("No boxes found")


            print("Inference done, writing image...")
            im = Image.open(img_path)
            draw_bounding_boxes_on_image(im,
                                         np.array(filt_boxes),
                                         display_str_list_list=filt_labels)

            return np.array(im)
