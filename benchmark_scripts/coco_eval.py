import json
import csv

import degirum as dg, mytools
cloud_token = mytools.get_token() # get cloud API access token from env.ini file
cloud_zoo_url = mytools.get_cloud_zoo_url() # get cloud zoo URL from env.ini file
zoo = dg.connect(dg.CLOUD, cloud_zoo_url, cloud_token)
#
# zoo.list_models()
model_name=''
model=zoo.load_model(model_name)

class_map = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33,34,35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
pred_path = 'predictions.json'
image_folder_path = ''
ground_truth_annotations_path = ''

map_evaluator=mytools.ObjectDetectionModelEvaluator(model, image_folder_path, ground_truth_annotations_path, class_map, pred_path, print_frequency=50)
map_results=map_evaluator.evaluate()