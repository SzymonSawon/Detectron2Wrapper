from imports import *

class Data_Handler:
    img_annotations = {}

    def __init__(self, images_path, annotations_json_file):
        self.images_path = images_path
        self.annotations_json_file = annotations_json_file
        pass

    def read_coco_format(self) -> None:
        register_coco_instances(
            "my_dataset_train",
            {},
            f"{self.images_path}/train/{self.annotations_json_file}",
            f"{self.images_path}/train/",
        )
        register_coco_instances(
            "my_dataset_test",
            {},
            f"{self.images_path}/test/{self.annotations_json_file}",
            f"{self.images_path}/test",
        )

    def test_images(self) -> None:
        """
        check if detectron2 reads annotations correctly
        """
        dataset_dicts = DatasetCatalog.get("my_dataset_train")
        for i in range(1):
            img = cv2.imread(dataset_dicts[i]["file_name"])
            visualizer = Visualizer(
                img[:, :, ::-1],
                metadata=MetadataCatalog.get("my_dataset_train"),
                scale=0.5,
            )
            vis = visualizer.draw_dataset_dict(dataset_dicts[i])
            cv2.imshow("Annotations Tester", vis.get_image()[:, :, ::-1])
            cv2.waitKey()