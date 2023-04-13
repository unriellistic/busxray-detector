from detectron2.config import CfgNode as CN
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

from tridentnet import add_tridentnet_config
from utils import parse_inference

class TridentNetPredictor(DefaultPredictor):
    '''
    Predictor class for TridentNet object detection.
    Callable; takes an OpenCV-format image as input, and outputs COCO-format predictions.
    See the parent class, DefaultPredictor, for more details.
    '''
    def __init__(self, config_file: str, opts: list[str], confidence_threshold: float = 0.5):
        # get default detectron2 config
        cfg = get_cfg()
        # configure tridentnet (it's not inside by default)
        add_tridentnet_config(cfg)
        # add configs from config file and options
        cfg.merge_from_file(config_file)
        cfg.merge_from_list(opts)
        # Set score_threshold for builtin models
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
        cfg.freeze()

        super().__init__(cfg)

    def __call__(self, original_image):
        raw_predictions = super().__call__(original_image)
        # process the predictions into a JSON-compatible format
        predictions = parse_inference(raw_predictions["instances"])
        return predictions