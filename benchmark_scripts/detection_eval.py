
import sys
import os

# Get the parent directory
parent_dir = os.path.dirname(os.path.realpath(__file__))
base_dir = os.path.dirname(os.path.realpath(parent_dir))
# Add the parent directory to sys.path
sys.path.append(base_dir)
##

import degirum as dg
import mytools
from eval_utils.detection_eval import ObjectDetectionModelEvaluator

cloud_token = mytools.get_token() # get cloud API access token from env.ini file
cloud_zoo_url = mytools.get_cloud_zoo_url() # get cloud zoo URL from env.ini file
zoo = dg.connect(dg.CLOUD, cloud_zoo_url, cloud_token)
#
# print( zoo.list_models() )
# model_name='yolov8s_relu6_fruits--512x512_float_openvino_cpu_1'
# model_name='visdrone_yolov5n_relu6--640x640_float_openvino_cpu_1'
model_name='yolov5nu_relu6_coco--640x640_quant_n2x_orca1_1'
# model_name='yolov5nu_relu6_car--640x640_quant_n2x_orca1_1'

model=zoo.load_model(model_name)

map_evaluator=ObjectDetectionModelEvaluator(model, input_img_fmt="JPEG")

# img_folder_path = '/data/ml-data/dataset_51/exported_dataset/trncFruits/images/validation'
# anno_json = '/home/mehrdad/wa/DG/PySDKExamples/annotations.json'

# img_folder_path = '/data/VisDrone/VisDrone2019-DET-val/images'
# anno_json = '/home/mehrdad/wa/DG/PySDKExamples/visdrone_labels.json'

img_folder_path = '/data/ml-data/datasets/coco/images/val2017'
anno_json = '/home/mehrdad/wa/DG/PySDKExamples/coco_labels.json'

map_results=map_evaluator.evaluate(img_folder_path, ground_truth_annotations_path=anno_json, num_val_images=10, print_frequency=100)

