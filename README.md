# Image processing

Segment or detect objects on images using tensorflow. Models are from different sources

## Usage

`launcher.py [-h] -t MODEL_TYPE [-i IMAGE_FILE] [-di IN_DIRECTORY] -do OUT_DIRECTORY`
                   
## Examples

### Segmentation using deeplabv3 from tf research repo.
`python3 launcher.py -di /dir/with/images -do /output/directory -t segmentationcs`

#### Note: 
To use this model, you need to download one model from [https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md].

Tested only with `deeplabv3\_mnv2\_cityscapes\_train\_2018\_02\_05.tar.gz`



### Segmentation
`python3 launcher.py -di /dir/with/images -do /output/directory -t segmentation`

### Detection
`python3 launcher.py -di /dir/with/images -do /output/directory -t detection`
