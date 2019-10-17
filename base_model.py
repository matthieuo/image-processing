import tensorflow as tf



class BaseModel:
    def __call__(self, img_path_l):
        raise NotImplementedError()

    def _load_tf_image(self, image_path):
        image = tf.io.read_file(image_path)
        image = tf.io.decode_image(image,
                                   channels=3,
                                   expand_animations=False)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        return image
       
