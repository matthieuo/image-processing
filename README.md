# Image processing

Segment or detect objects on images using tensorflow. Models are from tf.hub and tf.lite

## Usage

`launcher.py [-h] -t MODEL_TYPE [-i IMAGE_FILE] [-di IN_DIRECTORY] -do OUT_DIRECTORY`
                   
## Examples

### Segmentation
`python3 launcher.py -di /dir/with/images -do /output/directory -t segmentation`

### Detection
`python3 launcher.py -di /dir/with/images -do /output/directory -t detection`
