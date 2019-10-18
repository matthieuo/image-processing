import tensorflow as tf
import numpy as np
from base_model import BaseModel
from PIL import Image
from pathlib import Path
import os
from utils_img import decode_labels

class Segmentation(BaseModel):
    def __init__(self, lite_file, output_dir):
        # Load TFLite model and allocate tensors.
        self._model = tf.lite.Interpreter(model_path=lite_file)
        self._model.allocate_tensors()
        
        # Get input and output tensors.
        self._input_details = self._model.get_input_details()
        self._output_details = self._model.get_output_details()

        print(self._output_details)
        self._output_dir = output_dir / 'segment'
        os.makedirs(self._output_dir, exist_ok=True)
        

    def __call__(self, img_path_l):
        self._segment(img_path_l)

    def _segment(self, img_path_l: list) -> None:
        for img_path in img_path_l:
            try:
                image = self._load_tf_image(img_path)
            except ValueError as v:
                print(img_path)
                print("Image error, skipping...", v)
                continue

            image = tf.expand_dims(image, 0)
            image = tf.image.resize(image, (self._input_details[0]['shape'][1],
                                            self._input_details[0]['shape'][2]))

            self._model.set_tensor(self._input_details[0]['index'], image)

            self._model.invoke()
            
            output_data = self._model.get_tensor(self._output_details[0]['index'])
            
            output_data = tf.argmax(output_data, axis=3)
            pred = tf.expand_dims(output_data, axis=3)

            print("Writing mask")

            out = decode_labels(pred)
            im = Image.fromarray(out[0])
            im.save(self._output_dir / f"segment-{Path(img_path).parts[-1]}")
