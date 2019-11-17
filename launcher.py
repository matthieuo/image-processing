import tensorflow as tf
import argparse
from models.models_fact import ModelF
from pathlib import Path
from PIL import Image
import os
import cv2


def main():
    os.environ["TFHUB_CACHE_DIR"] = '/tmp/tfhub'
    parser = argparse.ArgumentParser()

    group_genera = parser.add_argument_group('General options')

    group_genera.add_argument("-t",
                              "--model_type",
                              help="Detection or segmentation",
                              required=True)

    group_genera.add_argument("-vf",
                              "--video_file",
                              help="Video file to load",
                              required=False)

    group_genera.add_argument("-di",
                              "--in_directory",
                              help="Directory to load",
                              required=False)

    group_genera.add_argument("-do",
                              "--out_directory",
                              help="Directory to save",
                              required=True)
    args = parser.parse_args()

    model = ModelF.get(args.model_type.lower())
    
    out_dir = Path(args.out_directory)
    print("out dir", out_dir)
    
    if args.video_file is not None:
        cap = cv2.VideoCapture(args.video_file)
 
        # Check if camera opened successfully
        if not cap.isOpened():
            raise ValueError("Unable to read video")

        os.makedirs(out_dir, exist_ok=True)
        
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))


        scale_percent = 50 # percent of original size
        width = int(frame_width * scale_percent / 100)
        height = int(frame_height * scale_percent / 100)
        dim = (width, height)
        dimrot = (height, width)     
        out = cv2.VideoWriter(str(out_dir / f"process-{Path(args.video_file).parts[-1]}"),
                              cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                              10,
                              dimrot)

        while True:
            ret, frame = cap.read()
            if ret:
                print(frame.shape)
                print(frame.dtype)

                image = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    
                # resize image
                image = cv2.resize(image,
                                   dimrot,
                                   interpolation=cv2.INTER_AREA)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                print(image.shape)
                print(image.dtype)

                out_img = model(image)

                cimg = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
                out.write(cimg)
                
            else:
                break
            
        cap.release()
        out.release()

    elif args.in_directory is not None:
        im_files = Path(args.in_directory).glob("*.jpg")
        if not im_files:
            raise ValueError("No images files found in directory", args.in_directory)
        
        os.makedirs(out_dir, exist_ok=True)
        
        for img_path in [str(im) for im in im_files]:
            try:
                image = tf.io.read_file(img_path)
                image = tf.io.decode_image(image,
                                           channels=3,
                                           expand_animations=False)
                out_img = model(image.numpy())
            except ValueError as v:
                print(img_path)
                print("Image error, skipping...", v)
                continue
            
            im = Image.fromarray(out_img)
            im.save(out_dir / f"process-{Path(img_path).parts[-1]}")
    else:
        raise ValueError('Please specify directory or file')





if __name__ == '__main__':
    main()
