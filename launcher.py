import argparse
from detect import Detection
from segment import Segmentation
from segment_cityscape import SegmentationCityScape
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

    if args.model_type.lower() == "detection":
        model = Detection("https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1",
                          Path(args.out_directory))
    elif args.model_type.lower() == "segmentation":
        model = Segmentation("deeplabv3_257_mv_gpu.tflite",
                             Path(args.out_directory))
    elif args.model_type.lower() == "segmentationcs":
        file_pb = 'trainval_fine/frozen_inference_graph.pb'
        # "deeplabv3_mnv2_cityscapes_train/frozen_inference_graph.pb",
        model = SegmentationCityScape(file_pb,
                                      Path(args.out_directory))
    else:
        raise ValueError(f"Model type not supported: {args.model_type}")

    if args.video_file is not None:
        model(args.video_file, video=True)
    elif args.in_directory is not None:
        im_files = Path(args.in_directory).glob("*.jpg")
        model([str(im) for im in im_files])
    else:
        raise ValueError('Please specify directory or file')


if __name__ == '__main__':
    main()
