import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from dg_coco_utils import *
from tqdm import tqdm


def load_evaluation_model(
    eval_model,
    OutputConfThreshold=0.001,
    OutputNMSThreshold=0.6,
    MaxDetections=100,
    MaxDetectionsPerClass=20,
    MaxClassesPerDetection=1,
    UseRegularNMS=True,
    InputResizeMethod="bilinear",
    InputPadMethod="letterbox",
    ImageBackend="opencv",
    InputImgFmt="JPEG",
):
    eval_model.output_confidence_threshold = OutputConfThreshold
    eval_model.output_nms_threshold = OutputNMSThreshold
    eval_model.output_max_detections = MaxDetections
    eval_model.output_max_detections_per_class = MaxDetectionsPerClass
    eval_model.output_max_classes_per_detection = MaxClassesPerDetection
    eval_model.output_use_regular_nms = UseRegularNMS
    eval_model.input_resize_method = InputResizeMethod
    eval_model.input_pad_method = InputPadMethod
    eval_model.image_backend = ImageBackend
    eval_model.input_image_format = InputImgFmt
    return eval_model


def coco_evaluation(
    eval_model, pred_path, image_folder_path, ground_truth_annotations_path, class_map
):
    jdict = []
    anno = COCO(ground_truth_annotations_path)
    num_images = len(anno.dataset["images"])
    files_dict = anno.dataset["images"][0:num_images]
    path_list = []
    print_frequency = 50
    for image_number in tqdm(range(0, num_images)):
        image_id = files_dict[image_number]["id"]
        path = image_folder_path + files_dict[image_number]["file_name"]
        path_list.append(path)
    with eval_model:
        for image_number, predictions in enumerate(eval_model.predict_batch(path_list)):
            if image_number % print_frequency == print_frequency - 1:
                image_number = image_number + 1
                print(image_number + 1)
            image_id = files_dict[image_number]["id"]
            save_results_coco_json(predictions.results, jdict, image_id, class_map)
    with open(pred_path, "w") as f:
        json.dump(jdict, f)

    pred = anno.loadRes(pred_path)
    eval_obj = COCOeval(anno, pred, "bbox")
    eval_obj.params.imgIds = [
        file["id"] for file in files_dict
    ]  # image IDs to evaluate
    eval_obj.evaluate()
    eval_obj.accumulate()
    eval_obj.summarize()
    map_all, map50 = eval_obj.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
    return eval_obj.stats
