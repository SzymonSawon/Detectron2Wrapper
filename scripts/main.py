from imports import *
from data_handler import Data_Handler
from visualization import Image_Visualizer, Video_Visualizer
from training import Config, Model_Trainer, COCOEvaluatorTrainer,RotatedCOCOEvaluatorTrainer
from augmentation import Augment

setup_logger()
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#model_output_folder: str,
#num_workers: int,
#model: Models,
#batch_size: int,
#learning_rate,
#max_iter: int,
#roi_head_batch_size: int,
#num_classes: int,
#device: str,
#testing_treshold: float,
#evaluation_interval: int,
#config_yaml_file_path: str = None,
def main():
    test = Data_Handler("dir_path", "annotations.json")
    test.read_coco_format()
    #test.test_images()

    test_config = Config(
        "output/",
        1,
        Models.COCO_DETECTION_FASTER_RCNN_R_101_FPN_3X,
        3,
        0.0002,
        50,
        128,
        1,
        "cuda",
        0.2,
        50,
    )
    test_config.set_config()
    #test_aug = Augment(test_config.cfg)
    #test_aug.add_resize((400,400))
    #test_aug.add_random_brightness(0.9,1.1)
    #test_aug.preview_augmentations()
    #test_trainer = Model_Trainer(test_config.cfg, COCOEvaluatorTrainer)
    #test_trainer.train(test_config.cfg, "test_model")
    model_user = Image_Visualizer("test_model.pth", test_config.cfg)
    #model_user.visualize_image("dir_path")
    vid_v = Video_Visualizer(test_config.cfg, "model_path.pth")
    vid_v.visualize_webcam()
    #vid_v.visualize_video_input('vid_path.mp4', 'output/')

if __name__ == "__main__":
    main()
