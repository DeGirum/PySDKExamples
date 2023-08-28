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
              class_map,
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
              print_frequency=0
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
        self._dg_model = dg_model
        self._image_folder_path = image_folder_path
        self._ground_truth_annotations_path = ground_truth_annotations_path
        self._class_map = class_map
        self._pred_path = pred_path
        self._output_conf_threshold = output_conf_threshold
        self._output_nms_threshold = output_nms_threshold
        self._max_detections = max_detections
        self._max_detections_per_class = max_detections_per_class
        self._max_classes_per_detection = max_classes_per_detection
        self._use_regular_nms = use_regular_nms
        self._input_resize_method = input_resize_method
        self._input_pad_method = input_pad_method
        self._image_backend = image_backend
        self._input_img_fmt = input_img_fmt
        self._print_frequency = print_frequency
    
    
        if self._dg_model.output_postprocess_type == "Detection" or "DetectionYolo" or "DetectionYoloV8":
            self._dg_model.output_confidence_threshold=self._output_conf_threshold
            self._dg_model.output_nms_threshold=self._output_nms_threshold
            self._dg_model.output_max_detections=self._max_detections
            self._dg_model.output_max_detections_per_class=self._max_detections_per_class
            self._dg_model.output_max_classes_per_detection=self._max_classes_per_detection
            self._dg_model.output_use_regular_nms=self._use_regular_nms
            self._dg_model.input_resize_method=self._input_resize_method
            self._dg_model.input_pad_method=self._input_pad_method
            self._dg_model.image_backend=self._image_backend
            self._dg_model.input_image_format=self._input_img_fmt
        else:
            raise Exception("Model loaded for evaluation is not a Detection Model")
    
    @classmethod
    def mAP_evaluate(cls, dg_model, image_folder_path, ground_truth_annotations_path, class_map, pred_path, args_yaml):
        """ 
        args_yaml (str) : Path of the yaml file that contains all the arguments.
        
        """
        cls.dg_model = dg_model
        cls.args_yaml = args_yaml
        with open(cls.args_yaml) as f:
            load_yaml = yaml.load(f, Loader=yaml.FullLoader)
            
        if cls.dg_model.output_postprocess_type == "Detection" or "DetectionYolo" or "DetectionYoloV8":    
            cls.dg_model.output_confidence_threshold = load_yaml["OutputConfThreshold"]
            cls.dg_model.output_nms_threshold=load_yaml["OutputNMSThreshold"]
            cls.dg_model.output_max_detections=load_yaml["MaxDetections"]
            cls.dg_model.output_max_detections_per_class=load_yaml["MaxDetectionsPerClass"]
            cls.dg_model.output_max_classes_per_detection=load_yaml["MaxClassesPerDetection"]
            cls.dg_model.output_use_regular_nms=load_yaml["UseRegularNMS"]
            cls.dg_model.input_resize_method=load_yaml["InputResizeMethod"]
            cls.dg_model.input_pad_method=load_yaml["InputPadMethod"]
            cls.dg_model.image_backend=load_yaml["ImageBackend"]
            cls.dg_model.input_image_format=load_yaml["InputImgFmt"]  
        else:
            raise Exception("Model loaded for evaluation is not a Detection Model")
            
        return cls(cls.dg_model, image_folder_path, ground_truth_annotations_path, class_map, pred_path)

    def evaluate(self):
        """Evaluation for the Detection model.
        
        Returns the mAP statistics.
        """     
        jdict=[]
        anno = COCO(self._ground_truth_annotations_path)
        num_images=len(anno.dataset['images'])
        files_dict=anno.dataset['images'][0:num_images]
        path_list=[]
        for image_number in tqdm(range(0,num_images)):
            image_id=files_dict[image_number]['id']
            path=self._image_folder_path + files_dict[image_number]['file_name']
            path_list.append(path)
        with self._dg_model:
            for image_number,predictions in enumerate(self._dg_model.predict_batch(path_list)):
                if self._print_frequency > 0:
                    if image_number%self._print_frequency==self._print_frequency-1:
                        print(image_number+1)
                image_id=files_dict[image_number]['id']
                save_results_coco_json(predictions.results,jdict,image_id,self._class_map)
        with open(self._pred_path, 'w') as f:
            json.dump(jdict, f)

        pred = anno.loadRes(self._pred_path)
        eval_obj = COCOeval(anno, pred, 'bbox')
        eval_obj.params.imgIds = [file['id'] for file in files_dict]  # image IDs to evaluate
        eval_obj.evaluate()
        eval_obj.accumulate()
        eval_obj.summarize()
        map_all, map50 = eval_obj.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        return eval_obj.stats
  