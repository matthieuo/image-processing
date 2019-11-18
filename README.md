# Image processing

Segment or detect objects on images using tensorflow. Models are from different sources

## Usage
`launcher.py [-h] -t MODEL_TYPE [-vf VIDEO_FILE] [-di IN_DIRECTORY] -do OUT_DIRECTORY`
          
### Videos support
This tool supports video using the `-vf` flag. Input videos can be in any format supported by OpenCV. The output video is written using the MJPEG codec.

## Examples

### Segmentation using deeplabv3 from tf research repo.
`python3 launcher.py -di /dir/with/images -do /output/directory -t segmentationcs`

#### Note: 
To use this model, you need to download one pretrained model from [https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md].

Tested only with `deeplabv3_mnv2_cityscapes_train_2018_02_05.tar.gz`



### Segmentation
`python3 launcher.py -di /dir/with/images -do /output/directory -t segmentation`

### Detection
`python3 launcher.py -di /dir/with/images -do /output/directory -t detection`
