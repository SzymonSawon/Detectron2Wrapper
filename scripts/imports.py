"""
Detectron2 dependencies:
    torch
    torchvision
Detectron2 optional dependencies:
    cv2 (testing)
Detectron2 notes:
    - coco annotation format


Other  libraries:
json - detectron2 reads json annotation files
detectron2.utils.logger - not required, but useful
yaml - saving config
os - required for temporary error fix, venv might fix this error
    temporary fix: os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    error:
        OMP: Error #15: Initializing libiomp5md.dll, but found libomp140.x86_64.dll already initialized.
        OMP: Hint This means that multiple copies of the OpenMP runtime have been linked
        into the program. That is dangerous, since it can degrade performance or cause
        incorrect results. The best thing to do is to ensure that only a single OpenMP
        runtime is linked into the process, e.g. by avoiding static linking of the
        OpenMP runtime in any library. As an unsafe, unsupported, undocumented
        workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to
        allow the program to continue to execute, but that may cause crashes or silently
        produce incorrect results. For more information, please see
        http://www.intel.com/software/products/support/.
    Possible fix:
        I was facing the similar problem for python, I fixed it by deleting the
        libiomp5md.dll duplicate file from Anaconda environment folder
        C:\\Users\\user_name\\Anaconda3\\envs\\your_env_name\\Library\\bin\\libiomp5md.dll
        In my case the file libiomp5md.dll was already in the base Anaconda bin
        folder C:\\Users\\aqadir\\Anaconda3\\Library\\bin"
        https://stackoverflow.com/questions/20554074/sklearn-omp-error-15-initializing-libiomp5md-dll-but-found-mk2iomp5md-dll-a
    After fix:
        os should be removed, it was swapped with Pathlib
"""

import torch, detectron2, json, cv2, random, yaml
from enum import Enum
from pathlib import Path
import numpy as np
import os
import copy

from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
import detectron2.data.transforms as T
from detectron2.data import DatasetMapper
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger
from detectron2.evaluation import (
    COCOEvaluator,
    RotatedCOCOEvaluator,
    CityscapesInstanceEvaluator,
    LVISEvaluator,
)
from models import Models
from detectron2.data.build import build_detection_train_loader, build_detection_test_loader
