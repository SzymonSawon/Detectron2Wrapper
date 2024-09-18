from imports import *


class Augment:
    """
    available augmentations:
        https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/transforms/augmentation_impl.py#L253
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def add_resize(self, shape: tuple) -> None:
        """
        Args:
            shape: (h, w) tuple or a int
        """
        self.cfg.AUGMENTATIONS.append(T.Resize(shape))

    def add_random_brightness(self, intensity_min: float, intensity_max: float) -> None:
        """
        Args:
            intensity_min (float): Minimum augmentation
            intensity_max (float): Maximum augmentation
        """
        self.cfg.AUGMENTATIONS.append(T.RandomBrightness(intensity_min, intensity_max))

    def add_random_flip(
        self, prob: float = 0.5, horizontal: bool = True, vertical: bool = False
    ) -> None:
        """
        Args:
            prob (float): probability of flip.
            horizontal (boolean): whether to apply horizontal flipping
            vertical (boolean): whether to apply vertical flipping
        """
        self.cfg.AUGMENTATIONS.append(T.RandomFlip(prob, horizontal, vertical))

    def add_random_rotation(
        self,
        angle,
        epand: bool = True,
        center=None,
        sample_style: str = "range",
        interp=None,
    ) -> None:
        """
        Args:
            angle (list[float]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the angle (in degrees).
                If ``sample_style=="choice"``, a list of angles to sample from
            expand (bool): choose if the image should be resized to fit the whole
                rotated image (default), or simply cropped
            center (list[[float, float]]):  If ``sample_style=="range"``,
                a [[minx, miny], [maxx, maxy]] relative interval from which to sample the center,
                [0, 0] being the top left of the image and [1, 1] the bottom right.
                If ``sample_style=="choice"``, a list of centers to sample from
                Default: None, which means that the center of rotation is the center of the image
                center has no effect if expand=True because it only affects shifting
            sample_style, interp - no informations in source code
        """
        self.cfg.AUGMENTATIONS.append(
            T.RandomRotation(angle, epand, center, sample_style, interp)
        )

    def preview_augmentations(self):
        train_data_loader = build_detection_train_loader(
            self.cfg,
            mapper=DatasetMapper(
                self.cfg, is_train=True, augmentations=self.cfg.AUGMENTATIONS
            ),
        )
        metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
        scale = 1.0
        for batch in train_data_loader:
            for per_image in batch:
                img = per_image["image"].permute(1, 2, 0).cpu().detach().numpy()
                img = utils.convert_image_to_rgb(img, self.cfg.INPUT.FORMAT)

                visualizer = Visualizer(img, metadata=metadata, scale=scale)
                target_fields = per_image["instances"].get_fields()
                labels = [
                    metadata.thing_classes[i] for i in target_fields["gt_classes"]
                ]
                vis = visualizer.overlay_instances(
                    labels=labels,
                    boxes=target_fields.get("gt_boxes", None),
                    masks=target_fields.get("gt_masks", None),
                    keypoints=target_fields.get("gt_keypoints", None),
                )
                cv2.imshow("window", vis.get_image()[:, :, ::-1])
                cv2.waitKey()
