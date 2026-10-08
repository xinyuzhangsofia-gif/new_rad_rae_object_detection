"""Model and detection-head presets for the current Cartesian model zoo."""

from .model7_centerpoint_64 import MODEL_CONFIG as MODEL7_CENTERPOINT_64
from .model7_centerpoint_128 import MODEL_CONFIG as MODEL7_CENTERPOINT_128
from .model7_radenet_64 import MODEL_CONFIG as MODEL7_RADENET_64
from .model8_centerpoint import MODEL_CONFIG as MODEL8_CENTERPOINT
from .model8_radenet import MODEL_CONFIG as MODEL8_RADENET
from .model12_centerpoint import MODEL_CONFIG as MODEL12_CENTERPOINT
from .model12_radenet import MODEL_CONFIG as MODEL12_RADENET
from .model13_centerpoint import MODEL_CONFIG as MODEL13_CENTERPOINT
from .model13_radenet import MODEL_CONFIG as MODEL13_RADENET
from .model14_yolox import MODEL_CONFIG as MODEL14_YOLOX
from .model15_centerpoint import MODEL_CONFIG as MODEL15_CENTERPOINT
from .model15_radenet import MODEL_CONFIG as MODEL15_RADENET
from .model16_radenet import MODEL_CONFIG as MODEL16_RADENET


MODEL_PRESETS = {
    "model7-centerpoint-64": MODEL7_CENTERPOINT_64,
    "model7-centerpoint-128": MODEL7_CENTERPOINT_128,
    "model7-radenet-64": MODEL7_RADENET_64,
    "model8-centerpoint": MODEL8_CENTERPOINT,
    "model8-radenet": MODEL8_RADENET,
    "model12-centerpoint": MODEL12_CENTERPOINT,
    "model12-radenet": MODEL12_RADENET,
    "model13-centerpoint": MODEL13_CENTERPOINT,
    "model13-radenet": MODEL13_RADENET,
    "model14-yolox": MODEL14_YOLOX,
    "model15-centerpoint": MODEL15_CENTERPOINT,
    "model15-radenet": MODEL15_RADENET,
    "model16-radenet": MODEL16_RADENET,
}


__all__ = [
    "MODEL_PRESETS",
    "MODEL7_CENTERPOINT_64",
    "MODEL7_CENTERPOINT_128",
    "MODEL7_RADENET_64",
    "MODEL8_CENTERPOINT",
    "MODEL8_RADENET",
    "MODEL12_CENTERPOINT",
    "MODEL12_RADENET",
    "MODEL13_CENTERPOINT",
    "MODEL13_RADENET",
    "MODEL14_YOLOX",
    "MODEL15_CENTERPOINT",
    "MODEL15_RADENET",
    "MODEL16_RADENET",
]
