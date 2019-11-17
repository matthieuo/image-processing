import tensorflow as tf
import numpy as np
from models.base_model import BaseModel
from PIL import Image
from pathlib import Path
import os
import cv2


class SegmentationCityScape(BaseModel):
    def __init__(self, tf1_pbfile):

        self._tf1_pbfile = tf1_pbfile
        self._model_loaded = False


    def _load_model(self):
        print("loading graph")
        with open(self._tf1_pbfile, 'rb') as pb_f:
            graph_def = tf.compat.v1.GraphDef()
            loaded = graph_def.ParseFromString(pb_f.read())

            self._inf_func = BaseModel.wrap_frozen_graph(
                graph_def,
                inputs='prefix/ImageTensor:0',
                outputs='prefix/SemanticPredictions:0',
                name='prefix')
            
        self._model_loaded = True
        print("model loaded")
        
    def __call__(self, image):
        if not self._model_loaded:
            self._load_model()
            
        print(image.dtype)
       
        image = tf.expand_dims(image, 0)
        output_data = self._inf_func(image)

        output_data = tf.squeeze(output_data, axis=0)
        print(output_data.shape)
        segmented_image = SegmentationCityScape._parse_pred(output_data.numpy(), 19)
        return segmented_image
        

    def _segment(self, img_path_l: list, video=False) -> None:

        if video:

            cap = cv2.VideoCapture(img_path_l)
 
            # Check if camera opened successfully
            if not cap.isOpened():
                print("Unable to read camera feed")
                return
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))


            scale_percent = 50 # percent of original size
            width = int(frame_width * scale_percent / 100)
            height = int(frame_height * scale_percent / 100)
            dim = (width, height)
            dimrot = (height, width)     
            out = cv2.VideoWriter(str(self._output_dir / f"segment-{Path(img_path_l).parts[-1]}"),
                                  cv2.VideoWriter_fourcc('M','J','P','G'),
                                  10,
                                  dimrot)
            far=0
            while True:
                ret, frame = cap.read()
                far = far + 1
                #if far > 10:
                #    break
                if ret:
                    print(frame.shape)
                    print(frame.dtype)

                    image = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    
                    # resize image
                    image = cv2.resize(image, dimrot, interpolation = cv2.INTER_AREA)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    #image = np.array(image)
                    print(image.shape)
                    print(image.dtype)
       

                    image = tf.expand_dims(image, 0)
                    output_data = self._inf_func(image)
                    #output_data = image
                    output_data = tf.squeeze(output_data, axis=0)
                    #print(output_data.shape)
                    print(f"Writing mask to {str(self._output_dir)}")
                    segmented_image = SegmentationCityScape._parse_pred(output_data.numpy(), 19)
                    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
                    out.write(segmented_image)
                    #output_data = cv2.cvtColor(output_data.numpy(), cv2.COLOR_RGB2BGR)
                    #out.write(output_data)
                else:
                    break
                
 
            cap.release()
            out.release()
        else:
            for img_path in img_path_l:
                try:
                    image = self._load_tf_image(img_path, convert=False)
                    print(image.dtype)
                except ValueError as v:
                    print(img_path)
                    print("Image error, skipping...", v)
                    continue

                image = tf.expand_dims(image, 0)
                output_data = self._inf_func(image)

                output_data = tf.squeeze(output_data, axis=0)
                print(output_data.shape)
                print(f"Writing mask to {str(self._output_dir)}")
                segmented_image = SegmentationCityScape._parse_pred(output_data.numpy(), 19)

                im = Image.fromarray(segmented_image)
                im.save(self._output_dir / f"segment-{Path(img_path).parts[-1]}")


    def _get_n_rgb_colors(n):
        """
        Get n evenly spaced RGB colors.
        Returns:
        rgb_colors (list): List of RGB colors.
        """
        max_value = 16581375 #255**3
        interval = int(max_value / n)
        colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]

        rgb_colors = [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]

        return rgb_colors

    def _parse_pred(pred, n_classes):
        """
        Parses a prediction and returns the prediction as a PIL.Image.
        Args:
        pred (np.array)
        Returns:
        parsed_pred (PIL.Image): Parsed prediction that we can view as an image.
        """
        uni = np.unique(pred)
        print(uni)
        empty = np.empty((pred.shape[0], pred.shape[1], 3))
        print(empty.shape)
        colors = SegmentationCityScape._get_n_rgb_colors(n_classes)

        for i, u in enumerate(uni):
            idx = np.transpose((pred == u).nonzero())
            c = colors[u]
            empty[idx[:,0], idx[:,1]] = [c[0],c[1],c[2]]

        parsed_pred = np.array(empty, dtype=np.uint8)
        #parsed_pred = Image.fromarray(parsed_pred)

        return parsed_pred
