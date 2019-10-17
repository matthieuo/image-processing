import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import PIL.Image as Image
from pathlib import Path
from utils_img import draw_bounding_boxes_on_image
from base_model import BaseModel


class Detection(BaseModel):
    def __init__(self, hub_handle, output_dir, confidence=0.5):
        print("Initializing model...")
        self.model = hub.load(hub_handle, tags=[])
        print("Model initialized")
 
        self._confidence = confidence
        self._output_dir = output_dir / 'detect'
        os.makedirs(self._output_dir, exist_ok=True)

    def __call__(self, img_path_l):
        self._detect(img_path_l)

    def _detect(self, img_path_l: list) -> None:
        for img_path in img_path_l:
            try:
                image = self._load_tf_image(img_path)
            except ValueError as v:
                print(img_path)
                print("Image error, skipping...", v)
                continue

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
                print("No boxes found, skiping image")
                continue
        
            print("Inference done, writing image...")
            im = Image.open(img_path)
            draw_bounding_boxes_on_image(im,
                                         np.array(filt_boxes),
                                         display_str_list_list=filt_labels)


            im.save(self._output_dir / f"detec-{Path(img_path).parts[-1]}")



    
