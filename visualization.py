from imports import *


class Image_Visualizer:
    """
    Image_Visualizer - uses any given folder of images
    """

    def __init__(self, model_filename, cfg):
        self.model_filename = model_filename
        self.cfg = cfg

    def visualize_image(self, images_path: str) -> None:
        """
        Visualizer Class(line32):
            img_rgb: a numpy array of shape (H, W, C), where H and W correspond to
                    the height and width of the image respectively. C is the number of
                    color channels. The image is required to be in RGB format since that
                    is a requirement of the Matplotlib library. The image is also expected
                    to be in the range [0, 255].
            metadata (Metadata): dataset metadata (e.g. class names and colors)
            instance_mode (ColorMode): defines one of the pre-defined style for drawing
                instances on an image.
            scale (float): scale the input image
        """
        if Path("output").is_dir():
            self.cfg.MODEL.WEIGHTS = f"{self.cfg.OUTPUT_DIR}/{self.model_filename}"
            root = Path(images_path)
            predictor = DefaultPredictor(self.cfg)
            for f in root.glob("*.jpg"):
                im = cv2.imread(f"{f}")
                outputs = predictor(im)
                v = Visualizer(
                    im[:, :, ::-1],
                    metadata=None,
                    scale=0.75,
                )
                v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                cv2.imshow("User", v.get_image()[:, :, ::-1])
                cv2.waitKey()


class Video_Visualizer:
    def __init__(self, cfg, model_path, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.instance_mode = instance_mode
        self.predictor = DefaultPredictor(cfg)
        cfg.MODEL.WEIGHTS = model_path

    def _frame_from_video(self, video):
        """
        get each video frame
        """
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def run_on_video(self, video):
        """
        Visualizes predictions on frames of the input video.

        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.

        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            predictions = predictions["instances"]
            vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            return vis_frame

        frame_gen = self._frame_from_video(video)
        for frame in frame_gen:
            yield process_predictions(frame, self.predictor(frame))

    def visualize_webcam(self):
        cam = cv2.VideoCapture(0)
        for vis in self.run_on_video(cam):
            cv2.namedWindow("Real Time Detection", cv2.WINDOW_NORMAL)
            cv2.imshow("Real Time Detectrion", vis)
            if cv2.waitKey(1) == 27:
                break
        cam.release()
        cv2.destroyAllWindows()

    def visualize_video_input(self, video_path: str, output_path: str):
        video = cv2.VideoCapture(video_path)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        codec, file_ext = ("mp4v", ".mp4")
        if codec == ".mp4v":
            warnings.warn("x264 codec not available, switching to mp4v")
        output_file = cv2.VideoWriter(
            filename=output_path,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(frames_per_second),
            frameSize=(width, height),
            isColor=True,
        )
        for vis_frame in self.run_on_video(video):
            output_file.write(vis_frame)
            cv2.namedWindow("Video Visualization", cv2.WINDOW_NORMAL)
            cv2.imshow("Video Visualization", vis_frame)
            if cv2.waitKey(1) == 27:
                break
        video.release()

