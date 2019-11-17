import tensorflow as tf
from models.base_model import BaseModel
from models.utils_img import decode_labels


class Segmentation(BaseModel):
    def __init__(self, lite_file):
        self._lite_file = lite_file
        self._model_loaded = False

    def _load_model(self):
        # Load TFLite model and allocate tensors.
        self._model = tf.lite.Interpreter(model_path=self._lite_file)
        self._model.allocate_tensors()

        # Get input and output tensors.
        self._input_details = self._model.get_input_details()
        self._output_details = self._model.get_output_details()
        self._model_loaded = True
        print(self._output_details)
        print("model loaded")
        
    def __call__(self, img):
        if not self._model_loaded:
            self._load_model()
        
        return self._segment(img)

    def _segment(self, img):
            image = tf.image.convert_image_dtype(img, dtype=tf.float32)
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

            return out
