import json
import yaml
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from dg_coco_utils import *
from tqdm import tqdm


class ObjectDetectionModelEvaluator:
    def __init__(
        self,
        dg_model,
        image_folder_path,
        ground_truth_annotations_path,
        classmap,
        pred_path,
        output_conf_threshold=0.001,
        output_nms_threshold=0.6,
        max_detections=100,
        max_detections_per_class=20,
        max_classes_per_detection=1,
        use_regular_nms=True,
        input_resize_method="bilinear",
        input_pad_method="letterbox",
        image_backend="opencv",
        input_img_fmt="JPEG",
        print_frequency=0,
    ):
        """
        Constructor.
            This class evaluates the mAP for Object Detection models.

            Args:
                dg_model (Detection model): Detection model from the Degirum model zoo.
                image_folder_path (str): Path to the image dataset.
                ground_truth_annotations_path (str): Path to the groundtruth json annotations.
                class_map (json): A json file that contains classes with its class ids, each category would have a list of class ids.
                pred_path (str): Path to save the predictions as a json file.
                output_conf_threshold (float): Output Confidence threshold.
                output_nms_threshold (float): Output Non-Max Suppression threshold.
                max_detections (int): Maximum Detections.
                max_detections_per_class (int): Maximum Detections Per Class.
                max_classes_per_detection (int): Maximum Classes Per Detection.
                use_regular_nms (boolean): Whether to use Regular Non-Max Suppression.
                input_resize_method (str): Input Resize Method.
                input_pad_method (str): Input Pad Method.
                image_backend (str): Image Backend.
                input_img_fmt (str): InputImgFmt.
                print_frequency (int): Number of image batches to be evaluated at a time.

        """

        self.dg_model = dg_model
        self.output_conf_threshold = output_conf_threshold
        self.output_nms_threshold = output_nms_threshold
        self.max_detections = max_detections
        self.max_detections_per_class = max_detections_per_class
        self.max_classes_per_detection = max_classes_per_detection
        self.use_regular_nms = use_regular_nms
        self.input_resize_method = input_resize_method
        self.input_pad_method = input_pad_method
        self.image_backend = image_backend
        self.input_img_fmt = input_img_fmt
        self.print_frequency = print_frequency
        self.image_folder_path = image_folder_path
        self.ground_truth_annotations_path = ground_truth_annotations_path
        self.classmap = classmap
        self.pred_path = pred_path
        self.print_frequency = print_frequency

        if (
            self.dg_model.output_postprocess_type == "Detection"
            or "DetectionYolo"
            or "DetectionYoloV8"
        ):
            self.dg_model.output_conf_threshold = self.output_conf_threshold
            self.dg_model.output_nms_threshold = self.output_nms_threshold
            self.dg_model.max_detections = self.max_detections
            self.dg_model.max_detections_per_class = self.max_detections_per_class
            self.dg_model.max_classes_per_detection = self.max_classes_per_detection
            self.dg_model.use_regular_nms = self.use_regular_nms
            self.dg_model.input_resize_method = self.input_resize_method
            self.dg_model.input_pad_method = self.input_pad_method
            self.dg_model.image_backend = self.image_backend
            self.dg_model.input_image_format = self.input_img_fmt
        else:
            raise Exception("Model loaded for evaluation is not a Detection Model")

    @classmethod
    def init_from_yaml(cls, dg_model, config_yaml):
        """
        args_yaml (str) : Path of the yaml file that contains all the arguments.

        """
        with open(config_yaml) as f:
            load_yaml = yaml.load(f, Loader=yaml.FullLoader)

        image_folder_path = load_yaml["ImageFolderPath"]
        ground_truth_annotations_path = load_yaml["GroundTruthAnnotationsPath"]
        classmap = load_yaml["ClassMap"]
        pred_path = load_yaml["PredictionJsonPath"]
        print_frequency = load_yaml["PrintFrequency"]
        output_conf_threshold = load_yaml["OutputConfThreshold"]
        output_nms_threshold = load_yaml["OutputNMSThreshold"]
        max_detections = load_yaml["MaxDetections"]
        max_detections_per_class = load_yaml["MaxDetectionsPerClass"]
        max_classes_per_detection = load_yaml["MaxClassesPerDetection"]
        use_regular_nms = load_yaml["UseRegularNMS"]
        input_resize_method = load_yaml["InputResizeMethod"]
        input_pad_method = load_yaml["InputPadMethod"]
        image_backend = load_yaml["ImageBackend"]
        input_image_format = load_yaml["InputImgFmt"]

        return cls(
            dg_model,
            image_folder_path,
            ground_truth_annotations_path,
            classmap,
            pred_path,
            output_conf_threshold,
            output_nms_threshold,
            max_detections,
            max_detections_per_class,
            max_classes_per_detection,
            use_regular_nms,
            input_resize_method,
            input_pad_method,
            image_backend,
            input_image_format,
            print_frequency,
        )

    def evaluate(self):
        """Evaluation for the Detection model.

        Returns the mAP statistics.
        """
        jdict = []
        anno = COCO(self.ground_truth_annotations_path)
        num_images = len(anno.dataset["images"])
        files_dict = anno.dataset["images"][0:num_images]
        path_list = []
        for image_number in tqdm(range(0, num_images)):
            image_id = files_dict[image_number]["id"]
            path = self.image_folder_path + files_dict[image_number]["file_name"]
            path_list.append(path)
        with self.dg_model:
            for image_number, predictions in enumerate(
                self.dg_model.predict_batch(path_list)
            ):
                if self.print_frequency > 0:
                    if image_number % self.print_frequency == self.print_frequency - 1:
                        print(image_number + 1)
                image_id = files_dict[image_number]["id"]
                save_results_coco_json(
                    predictions.results, jdict, image_id, self.classmap
                )
        with open(self.pred_path, "w") as f:
            json.dump(jdict, f)

        pred = anno.loadRes(self.pred_path)
        eval_obj = COCOeval(anno, pred, "bbox")
        eval_obj.params.imgIds = [
            file["id"] for file in files_dict
        ]  # image IDs to evaluate
        eval_obj.evaluate()
        eval_obj.accumulate()
        eval_obj.summarize()
        map_all, map50 = eval_obj.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        return eval_obj.stats
