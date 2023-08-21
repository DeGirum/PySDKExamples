import argparse
import json
import degirum as dg, mytools
import requests
import os
import tempfile
from contextlib import contextmanager
from dg_coco_utils import *
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
import numpy as np
import csv

@contextmanager
def annotations_accessor(url:str):
    r = requests.get(url)
    r.raise_for_status()
    fd, fname = tempfile.mkstemp()
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(r.json(), f)
        yield fname
    finally:
        os.unlink(fname)
        
        
def parse_opt(known=False):
    parser = argparse.ArgumentParser()
#     parser.add_argument('--save_pred_path', type=str, help='save the predictions as a json')
    parser.add_argument('--OutputConfThreshold', type=float, default=0.001, help='OutputConfidenceThreshold')
    parser.add_argument('--OutputNMSThreshold', type=float, default=0.6, help='OutputNMSThreshold')
    parser.add_argument('--MaxDetections', type=int, default=100, help='Maximum detections')
    parser.add_argument('--MaxDetectionsPerClass', type=int, default=20, help='MaxDetectionsPerClass')
    parser.add_argument('--MaxClassesPerDetection', type=int, default=1, help='MaxClassesPerDetection')
    parser.add_argument('--UseRegularNMS',action='store_true', help='whether to use regular nms')
    parser.add_argument('--InputResizeMethod', type=str, default='bilinear', help='InputResizeMethod')
    parser.add_argument('--InputPadMethod', default='letterbox', help='InputPadMethod')
    parser.add_argument('--ImageBackend',type=str, default='opencv', help='ImageBackend')
    parser.add_argument('--InputImgFmt',type=str, default='JPEG', help='format of the input image')
#     parser.add_argument('--ClassMap', type=str, default = 'coco', help='classmap for each category')

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt, models):
    with open('./coco_configs/classmap_config.json') as config:
        classmap_categories=json.load(config)
    
#     print (models.list_models())
    
    with open('./models.json') as json_data:
        models_list=json.load(json_data)
        
    with open('./coco_configs/input_data.json') as df:
        input_json = json.load(df)
    
    
    mAP_list = []
    for key,model_name in models_list.items():
        print (model_name)
        if "coco" in model_name:
            image_folder_path = input_json["image_folder_path"]["coco"]
            ground_truth_annotations_path = input_json["ground_truth_annotations_path"]["coco"]
            class_map = classmap_categories["ClassMap"]["coco"]
                     
        elif "car" in model_name:
            image_folder_path = input_json["image_folder_path"]["car"]
            ground_truth_annotations_path = input_json["ground_truth_annotations_path"]["car"]
            class_map = classmap_categories["ClassMap"]["car"]
        
        elif "hand" in model_name:
            image_folder_path = input_json["image_folder_path"]["hand"]
            ground_truth_annotations_path = input_json["ground_truth_annotations_path"]["hand"]
            class_map = classmap_categories["ClassMap"]["hand"]
            
        elif "face" in model_name:
            image_folder_path = input_json["image_folder_path"]["face"]
            ground_truth_annotations_path = input_json["ground_truth_annotations_path"]["face"]
            class_map = classmap_categories["ClassMap"]["face"]
            
        elif "lp" in model_name:
            image_folder_path = input_json["image_folder_path"]["lp"]
            ground_truth_annotations_path = input_json["ground_truth_annotations_path"]["lp"]
            class_map = classmap_categories["ClassMap"]["lp"]
            
        pred_path = model_name + "_predictions.json"
            
        eval_model=models.load_model(model_name)
#         print (eval_model)
        eval_model.output_confidence_threshold=opt.OutputConfThreshold
        eval_model.output_nms_threshold=opt.OutputNMSThreshold
        eval_model.output_max_detections=opt.MaxDetections
        eval_model.output_max_detections_per_class=opt.MaxDetectionsPerClass
        eval_model.output_max_classes_per_detection=opt.MaxClassesPerDetection
        eval_model.output_use_regular_nms=opt.UseRegularNMS
        eval_model.input_resize_method=opt.InputResizeMethod
        eval_model.input_pad_method=opt.InputPadMethod
        eval_model.image_backend=opt.ImageBackend
        eval_model.input_image_format=opt.InputImgFmt
        jdict=[]
        with annotations_accessor(ground_truth_annotations_path) as annotations_file:
            anno = COCO(annotations_file)
        num_images=len(anno.dataset['images'])
        files_dict=anno.dataset['images'][0:num_images]
        path_list=[]
        print_frequency=50
        for image_number in tqdm(range(0,num_images)):
            image_id=files_dict[image_number]['id']
            path=image_folder_path + files_dict[image_number]['file_name']
            path_list.append(path)
        with eval_model:
            for image_number,predictions in tqdm(enumerate(eval_model.predict_batch(path_list))):
                if image_number%print_frequency==print_frequency-1:
                    print(image_number+1)
                image_id=files_dict[image_number]['id']
                save_results_coco_json(predictions.results,jdict,image_id,class_map)
        with open(pred_path, 'w') as f:
            json.dump(jdict, f)

        pred = anno.loadRes(pred_path)
        eval_obj = COCOeval(anno, pred, 'bbox')
        eval_obj.params.imgIds = [file['id'] for file in files_dict]  # image IDs to evaluate
        eval_obj.evaluate()
        eval_obj.accumulate()
        eval_obj.summarize()
        map_all, map50 = eval_obj.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        print ("------------------saving the results to a csv file-----------------------")
        header = ['model_name','mAP50', 'mAP75', 'mAP0.50:0.95', 'mAP0.50-0.95','mAP0.50-0.95','mAP0.50-0.95','mAP0.50-0.95','mAP0.50-0.95','mAP0.50-0.95', 'mAP0.50-0.95','mAP0.50-0.95']
        data = eval_obj.stats.tolist()
        data.insert(0,model_name)
        mAP_list.append(data)
        
        with open('./map_results.csv', 'w', encoding='UTF8', newline ='') as f:
            writer = csv.writer(f, delimiter=',') 
            writer.writerow(header)
            writer.writerows(mAP_list)
            f.close()


if __name__ == '__main__':
    opt = parse_opt()
    cloud_token = mytools.get_token() # get cloud API access token from env.ini file
    cloud_zoo_url = mytools.get_cloud_zoo_url() # get cloud zoo URL from env.ini file
    zoo = dg.connect(dg.CLOUD, cloud_zoo_url, cloud_token)
    main(opt, zoo)