import argparse
from detect import Detection
from segment import Segmentation
from pathlib import Path
import os


def main():
    os.environ["TFHUB_CACHE_DIR"] = '/tmp/tfhub'
    parser = argparse.ArgumentParser()

    group_genera = parser.add_argument_group('General options')

    group_genera.add_argument("-t",
                              "--model_type",
                              help="Detection or segmentation",
                              required=True)

    group_genera.add_argument("-i",
                              "--image_file",
                              help="Image file to load",
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

    if args.model_type.lower() == "detection":
        model = Detection("https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1",
                          Path(args.out_directory))
    elif args.model_type.lower() == "segmentation":
        model = Segmentation("deeplabv3_257_mv_gpu.tflite",
                             Path(args.out_directory))
    else:
        ValueError(f"Model type not supported: {args.model_type}")

    if args.image_file is not None:
        model([args.image_file])
    elif args.in_directory is not None:
        im_files = Path(args.in_directory).glob("*.jpg")
        model([str(im) for im in im_files])
    else:
        raise ValueError('Please specify directory or file')


if __name__ == '__main__':
    main()
