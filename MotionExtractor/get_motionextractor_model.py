from MotionExtractor.slowfast.utils.parser import load_config, parse_args
import MotionExtractor.slowfast.utils.checkpoint as cu
from MotionExtractor.slowfast.utils import logging
from MotionExtractor.slowfast.models import build_model

logger = logging.get_logger(__name__)

def make_motion_model():

    args = parse_args()
    cfg = load_config(args)

    model = build_model(cfg, gpu_id=None)
    cu.load_test_checkpoint(cfg, model)

    return model

if __name__ == "__main__":
    make_motion_model()