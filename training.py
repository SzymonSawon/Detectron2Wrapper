from imports import *


class Config:
    """
    atributes:
        pre-configured models:
            list of available models:
                https://github.com/facebookresearch/detectron2/blob/main/detectron2/model_zoo/model_zoo.py
            performance of all models:
                https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md
        device - options:
            'gpu' - default
            'cpu' - debugging alternative
        all changable config parameters:
            https://detectron2.readthedocs.io/en/latest/modules/config.html

    """

    num_workers: int
    model: Models
    batch_size: int
    learning_rate: float
    max_iter: int
    roi_head_batch_size: int
    num_classes: int
    device: str
    testing_treshold: float

    def __init__(
        self,
        model_output_folder: str,
        num_workers: int,
        model: Models,
        batch_size: int,
        learning_rate,
        max_iter: int,
        roi_head_batch_size: int,
        num_classes: int,
        device: str,
        testing_treshold: float,
        evaluation_interval: int,
        config_yaml_file_path: str = None,
    ) -> None:
        self.model_output_folder = model_output_folder
        self.num_workers = num_workers
        self.model = model
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.roi_head_batch_size = roi_head_batch_size
        self.num_classes = num_classes
        self.device = device
        self.testing_treshold = testing_treshold
        self.evaluation_interval = evaluation_interval
        self.config_yaml_file_path = config_yaml_file_path
        self.cfg = get_cfg()

    def set_config(self) -> None:
        """
        sets all variables
        """
        if self.config_yaml_file_path != None:
            self.cfg.merge_from_file(self.config_yaml_file_path)
        else:
            self.cfg.merge_from_file(
                model_zoo.get_config_file(f"{self.model.value}.yaml")
            )
        self.cfg.DATASETS.TRAIN = ("my_dataset_train",)
        self.cfg.DATASETS.TEST = ("my_dataset_test",)
        self.cfg.TEST.EVAL_PERIOD = self.evaluation_interval
        self.cfg.DATALOADER.NUM_WORKERS = self.num_workers
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            f"{self.model.value}.yaml"
        )
        self.cfg.SOLVER.IMS_PER_BATCH = self.batch_size
        self.cfg.SOLVER.BASE_LR = self.learning_rate
        self.cfg.SOLVER.MAX_ITER = self.max_iter
        self.cfg.SOLVER.STEPS = []
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = self.roi_head_batch_size
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes
        self.cfg.MODEL.DEVICE = self.device
        self.cfg.OUTPUT_DIR = self.model_output_folder
        self.cfg.AUGMENTATIONS = []
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.testing_treshold

    def save_config(self) -> None:
        """
        save config to yaml file
        full example config file:
            https://detectron2.readthedocs.io/en/latest/modules/config.html
        """
        with open("output/detectron2_config.yaml", "w") as file:
            yaml.dump(self.cfg, file)


class Model_Trainer:
    """
    train():
        trainer.test(evaluators=COCOEvaluator("my_dataset_test", config, False, output_dir="./output/"))
            attributes:
                - registered dataset
                - model config
                - distributed (defaulet = True): if True, will collect results from all ranks and run evaluation
                    in the main process. (source: https://github.com/facebookresearch/detectron2/blob/main/detectron2/evaluation/coco_evaluation.py)
                - output directory
    """

    def __init__(self, cfg, trainer_type: DefaultTrainer = DefaultTrainer):
        self.cfg = cfg
        self.trainer = trainer_type(cfg)

    def train(self, config, model_name: str) -> None:
        """
        CustomTrainer is custom trainer that enables testing after training
        DefaultTrainer uses only training dataset
        to use DefaultTrainer and then test
            trainer = DefaultTrainer(config)
            trainer.test(config, trainer.model, evaluators=COCOEvaluator("my_dataset_test", config, False, output_dir="./output/"))
        Alternative evaluators:
            https://detectron2.readthedocs.io/en/latest/modules/evaluation.html
        """
        Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        self.trainer.resume_or_load(resume=False)
        self.trainer.train()
        checkpointer = DetectionCheckpointer(self.trainer.model, save_dir="output")
        checkpointer.save(model_name)


"""
https://detectron2.readthedocs.io/en/latest/modules/evaluation.html
Different evaluators:
"""


class COCOEvaluatorTrainer(DefaultTrainer):
    """
    Evaluate AR for object proposals, AP for instance detection/segmentation, AP
    for keypoint detection outputs using COCO’s metrics. See
    http://cocodataset.org/#detection-eval and
    http://cocodataset.org/#keypoints-eval to understand its metrics. The
    metrics range from 0 to 100 (instead of 0 to 1), where a -1 or NaN means the
    metric cannot be computed (e.g. due to no predictions made).
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder="/output"):
        if output_folder is None:
            Path(output_folder).mkdir(parents=True, exist_ok=True)
            output_folder = "coco_eval"
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(cfg, is_train=True, augmentations=cfg.AUGMENTATIONS)
        return build_detection_train_loader(cfg, mapper=mapper)


class RotatedCOCOEvaluatorTrainer(DefaultTrainer):
    """Evaluate object proposal/instance detection outputs using COCO-like
    metrics and APIs, with rotated boxes support. Note: this uses IOU only and
    does not consider angle differences."""

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder="/output"):
        if output_folder is None:
            Path(output_folder).mkdir(parents=True, exist_ok=True)
            output_folder = "rotated_coco_eval"
        return RotatedCOCOEvaluator(dataset_name, cfg, False, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(cfg, is_train=True, augmentations=cfg.AUGMENTATIONS)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        mapper = DatasetMapper(cfg, is_train=False, augmentations=cfg.AUGMENTATIONS)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)
