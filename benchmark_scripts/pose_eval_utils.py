#
# detection_eval.py: evaluation toolkit for detection models used in PySDK samples
#
# Copyright DeGirum Corporation 2023
# All rights reserved
#

import yaml
import json
import os
from typing import List
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
# from custom_pp import Yolov8PoseDetection

def xyxy2xywh(x):
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray): The input bounding box coordinates in (x1, y1, x2, y2) format.

    Returns:
        y (np.ndarray): The bounding box coordinates in (x, y, width, height) format.
    """
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def get_keypoints(keypoints_res):
    keypoints = []
    for ldmks in keypoints_res:
        kypts = ldmks["landmark"]
        kypts_score = ldmks["score"]
        keypoints.extend(float(x) for x in kypts)
        keypoints.append(kypts_score)
    return keypoints


def save_results_coco_json(results, jdict, image_id, class_map=None):
    """Serialize YOLO predictions to COCO json format."""
    max_category_id = 0
    for result in results:
        box = xyxy2xywh(np.asarray(result["bbox"]).reshape(1, 4) * 1.0)  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        box = box.reshape(-1).tolist()
        category_id = (
            class_map[result["category_id"]] if class_map else result["category_id"]
        )
        jdict.append(
            {
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [np.round(x, 3) for x in box],
                "keypoints": get_keypoints(result["landmarks"]),
                "score": np.round(result["score"], 5),
            }
        )
        max_category_id = max(max_category_id, category_id)

    return max_category_id


class PoseModelEvaluator:
    def __init__(
        self,
        dg_model,
        classmap=None,
        pred_path="predictions-pose.json",
        output_confidence_threshold=0.001,
        output_nms_threshold=0.7,
        output_max_detections=300,
        output_max_detections_per_class=100,
        output_max_classes_per_detection=1,
        output_use_regular_nms=True,
        input_resize_method="bilinear",
        input_pad_method="letterbox",
        image_backend="opencv",
        input_img_fmt="JPEG",
        input_letterbox_fill_color=(114, 114, 114),
        input_numpy_colorspace="auto",
    ):
        """
        Constructor.
            This class evaluates the mAP for Object Detection models.

            Args:
                dg_model (Detection model): Detection model from the Degirum model zoo.
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
                input_letterbox_fill_color (tuple): the RGB color for padding used in letterbox
                input_numpy_colorspace (str): input colorspace: ("BGR" to match OpenCV image backend)
        """

        self.dg_model = dg_model
        self.classmap = classmap
        self.pred_path = pred_path

        if (
            self.dg_model.output_postprocess_type == "Detection"
            or "DetectionYolo"
            or "DetectionYoloV8"
        ):
            self.dg_model.output_confidence_threshold = output_confidence_threshold
            self.dg_model.output_nms_threshold = output_nms_threshold
            self.dg_model.output_max_detections = output_max_detections
            self.dg_model.output_max_detections_per_class = (
                output_max_detections_per_class
            )
            self.dg_model.output_max_classes_per_detection = (
                output_max_classes_per_detection
            )
            self.dg_model.output_use_regular_nms = output_use_regular_nms
            self.dg_model.input_resize_method = input_resize_method
            self.dg_model.input_pad_method = input_pad_method
            self.dg_model.image_backend = image_backend
            self.dg_model.input_image_format = input_img_fmt
            self.dg_model.input_numpy_colorspace = input_numpy_colorspace
            self.dg_model.input_letterbox_fill_color = input_letterbox_fill_color
        else:
            raise Exception("Model loaded for evaluation is not a Detection Model")

        # self.dg_model.output_postprocess_type = 'None'
        # self.dg_model.custom_postprocessor = Yolov8PoseDetection

    @classmethod
    def init_from_yaml(cls, dg_model, config_yaml):
        """
        config_yaml (str) : Path of the yaml file that contains all the arguments.

        """
        with open(config_yaml) as f:
            args = yaml.load(f, Loader=yaml.FullLoader)

        return cls(
            dg_model=dg_model,
            classmap=args["classmap"],
            pred_path=args["pred_path"],
            output_confidence_threshold=args["output_confidence_threshold"],
            output_nms_threshold=args["output_nms_threshold"],
            output_max_detections=args["output_max_detections"],
            output_max_detections_per_class=args["output_max_detections_per_class"],
            output_max_classes_per_detection=args["output_max_classes_per_detection"],
            output_use_regular_nms=args["output_use_regular_nms"],
            input_resize_method=args["input_resize_method"],
            input_pad_method=args["input_pad_method"],
            image_backend=args["image_backend"],
            input_img_fmt=args["input_img_fmt"],
            input_letterbox_fill_color=tuple(args["input_letterbox_fill_color"]),
            input_numpy_colorspace=args["input_numpy_colorspace"],
        )

    def evaluate(
        self,
        image_folder_path: str,
        ground_truth_annotations_path: str,
        num_val_images: int = 0,
        print_frequency: int = 0,
    ):
        """Evaluation for the Detection model.

            Args:
                image_folder_path (str): Path to the image dataset.
                ground_truth_annotations_path (str): Path to the groundtruth json annotations.
                num_val_images (int): max number of images used for evaluation. 0: all images in image_folder_path is used.
                print_frequency (int): Number of image batches to be evaluated before printing num evaluated images

        Returns the mAP statistics.
        """
        jdict: List[dict] = []
        anno = COCO(ground_truth_annotations_path)
        num_images = len(anno.dataset["images"])
        files_dict = anno.dataset["images"][0:num_images]
        path_list: List[str] = []
        img_id_list: List[str] = []
        for image_number in range(0, num_images):
            image_id = files_dict[image_number]["id"]
            path = os.path.join(
                image_folder_path, files_dict[image_number]["file_name"]
            )
            if os.path.exists(path):
                path_list.append(path)
                img_id_list.append(image_id)

        # sort the image ids to match ultralytic repo
        sorted_indices = sorted(range(len(img_id_list)), key=lambda i: img_id_list[i])
        sorted_img_id_list = [img_id_list[i] for i in sorted_indices]
        sorted_path_list = [path_list[i] for i in sorted_indices]

        if num_val_images > 0:
            sorted_path_list = sorted_path_list[0:num_val_images]

        with self.dg_model:
            for image_number, predictions in enumerate(
                self.dg_model.predict_batch(sorted_path_list)
            ):
                if print_frequency > 0:
                    if image_number % print_frequency == print_frequency - 1:
                        print(image_number + 1)
                image_id = sorted_img_id_list[image_number]
                save_results_coco_json(
                    predictions.results, jdict, image_id, self.classmap
                )

        with open(self.pred_path, "w") as f:
            json.dump(jdict, f, indent=4)

        pred = anno.loadRes(self.pred_path)
        # bbox
        eval_obj_bb = COCOeval(anno, pred, "bbox")
        eval_obj_bb.params.imgIds = [
            id for id in sorted_img_id_list
        ]  # image IDs to evaluate 
        eval_obj_bb.evaluate()
        eval_obj_bb.accumulate()
        eval_obj_bb.summarize()
        # keypoints
        eval_obj_kp = COCOeval(anno, pred, "keypoints")
        eval_obj_kp.params.imgIds = [
            id for id in sorted_img_id_list
        ]  # image IDs to evaluate
        eval_obj_kp.evaluate()
        eval_obj_kp.accumulate()
        eval_obj_kp.summarize()
        
        return eval_obj_bb.stats, eval_obj_kp.stats