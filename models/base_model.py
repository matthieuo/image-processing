import tensorflow as tf



class BaseModel:
    def __call__(self, img_path_l):
        raise NotImplementedError()

    def _load_tf_image(self, image_path, convert=True):
        image = tf.io.read_file(image_path)
        image = tf.io.decode_image(image,
                                   channels=3,
                                   expand_animations=False)

        if convert:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        return image


    def wrap_frozen_graph(graph_def, inputs, outputs, name=""):
        def _imports_graph_def():
            tf.compat.v1.import_graph_def(graph_def, name=name)
        wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
        import_graph = wrapped_import.graph
        return wrapped_import.prune(
            tf.nest.map_structure(import_graph.as_graph_element, inputs),
            tf.nest.map_structure(import_graph.as_graph_element, outputs))

       
