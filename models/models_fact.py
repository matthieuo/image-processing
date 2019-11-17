from models.detect import Detection
from models.segment_cityscape import SegmentationCityScape
from models.segment import Segmentation


class ModelFactory:
    def __init__(self):
        self._creators = {}

    def register_format(self, format, creator):
        self._creators[format] = creator

    def get(self, format):
        creator = self._creators.get(format)
        if not creator:
            raise ValueError("This model is not available ", format)
        return creator


ModelF = ModelFactory()
ModelF.register_format('detection',
                       Detection("https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"))
ModelF.register_format('segmentation',
                       Segmentation("deeplabv3_257_mv_gpu.tflite"))
ModelF.register_format('segmentationcs',
                       SegmentationCityScape('trainval_fine/frozen_inference_graph.pb'))

