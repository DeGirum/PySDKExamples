
import os
import sys
import re
import os.path as osp
import json
import cv2
import numpy as np
from tqdm import tqdm

from collections import OrderedDict


def create_annotations(img_list, ann_list, cat_list):
    return OrderedDict({'categories': cat_list,
                        'images': img_list,
                        'annotations': ann_list})

def create_images_entry(image_id, width=None, height=None, file_name=None):
    if width is None or height is None:
        return OrderedDict({'id':image_id })
    else:
        return OrderedDict({'id':image_id, 'file_name':file_name if file_name else str(image_id), 'width':width, 'height':height })

def create_categories(class_names, class_ids):
    return [{'id':class_ids[i], 'name':cls} for i, cls in enumerate(class_names)]

def create_annotations_entry(image_id, bbox, category_id, ann_id, iscrowd=0, area=None, segmentation=None):
    if area is None:
        if segmentation is None:
            #Calulate area with bbox
            area = bbox[2] * bbox[3]
        else:
            raise NotImplementedError()
            
    return OrderedDict({
            "id": ann_id,
            "image_id": image_id,
            "category_id": category_id,
            "iscrowd": iscrowd,
            "area": area,
            "bbox": bbox
           })


def get_image_id_from_path(image_path):
    image_path = osp.splitext(image_path)[0]
    m = re.search(r'\d+$', image_path)
    return int(m.group())

def bbox_cxcywh_to_xywh(box):
    x, y = box[..., 0] - box[..., 2] / 2, box[..., 1] - box[..., 3] / 2
    box[..., 0], box[..., 1] = x, y
    return box

def bbox_relative_to_absolute(box, img_dim, x_idx=[0,2], y_idx=[1,3]):
    box[..., x_idx] *= img_dim[0]
    box[..., y_idx] *= img_dim[1]
    return box

def add_img_ann(label_path, ann_list, image_id, width, height, class_map=None):
    # Read Labels
    # label_path = img_path.replace('jpg', 'txt').replace('images', 'labels')
    max_category_id = 0
    if osp.exists(label_path):
        labels = np.genfromtxt(label_path, delimiter=' ', usecols=range(5)).reshape(-1,5)
        labels[..., 1:5] = bbox_relative_to_absolute(bbox_cxcywh_to_xywh(labels[..., 1:5]), (width, height))
                    # class_id, x_center, y_center, width, height = line.split()
					# class_id = int(class_id)
					# bbox_x = (float(x_center) - float(width) / 2) * im.size[0]
					# bbox_y = (float(y_center) - float(height) / 2) * im.size[1]
					# bbox_width = float(width) * im.size[0]
					# bbox_height = float(height) * im.size[1]
    else:
        labels = []
                                                    
    for label in labels:
        category_id = class_map[int(label[0])] if class_map else int(label[0])
        bbox = list(label[1:5])
        ann_id = len(ann_list)
        ann_list.append(create_annotations_entry(image_id, bbox, category_id, ann_id))
        max_category_id = max(max_category_id, category_id)
    
    return max_category_id

def save_gt_coco_json(img_list, ann_list, image_id, img_path, label_path, class_map=None):
    if label_path==None:
        label_path = img_path.replace('jpg', 'txt').replace('images', 'labels')
    
    max_category_id = 0
    if osp.exists(img_path) and osp.exists(label_path):
        img = cv2.imread(img_path)
        height, width = img.shape[0], img.shape[1]
        img_list.append(create_images_entry(image_id, width, height))
        max_category_id = add_img_ann(label_path, ann_list, image_id, width, height, class_map)
    
    return max_category_id


def get_img_ann_list(img_path_list, label_path_list, class_map):
    img_list, ann_list = [],[]
    for img_path, label_path in tqdm(zip(img_path_list, label_path_list), file=sys.stdout, leave=True, total=len(img_path_list)):
        image_id = get_image_id_from_path(img_path)
        save_gt_coco_json(img_list, ann_list, image_id, img_path, label_path, class_map)
            
    return img_list, ann_list

def create_annotations_dict(target_txt, class_names, class_ids=None):
    if class_ids is None:
        class_ids = [i for i in range(len(class_names))]

    with open(target_txt, 'r') as f:
        img_path_list = [lines.strip() for lines in f.readlines()]
    label_path_list = [img_path.replace('jpg', 'txt').replace('images', 'labels') for img_path in img_path_list]
    
    #img_path_list, label_path_list = [img_path_list[1]], [label_path_list[1]]
    img_list, ann_list = get_img_ann_list(img_path_list, label_path_list, class_ids)
    cat_list = create_categories(class_names, class_ids)
    
    ann_dict = create_annotations(cat_list, img_list, ann_list)
        
    return ann_dict

def generate_annotations_file(target_txt, class_names, out, class_ids=None):
    ann_dict = create_annotations_dict(target_txt, class_names, class_ids=class_ids)
    with open(out, 'w') as f:
        json.dump(ann_dict, f, indent=4, separators=(',', ':'))

def yolo2coco(image_folder_path, anno_folder_path=None, out_json_path='annotations.json', num_classes=None):
    """Converting the dataset annotation format from yolo txt format to coco json
       There should be a filename.txt in anno_folder_path when there is filename.jpg in image_folder_path
    """ 
    if anno_folder_path == None:
        anno_folder_path = image_folder_path.replace('images', 'labels')

    ann_list = []
    img_list = []
    max_category_id = 0
    img_path_list=[]
    label_path_list=[]
    for filename in os.listdir(image_folder_path):
        if filename.endswith('.jpg'):
            img_path_list.append( os.path.join(image_folder_path, filename) )
            label_path_list.append( os.path.join(anno_folder_path, filename).replace('.jpg', '.txt') )
        elif filename.endswith('.jpeg'):
            img_path_list.append( os.path.join(image_folder_path, filename) )
            label_path_list.append( os.path.join(anno_folder_path, filename).replace('.jpeg', '.txt') )

    for img_path, label_path in zip(img_path_list, label_path_list):
        image_id = os.path.basename(img_path)
        max_category_id_gt = save_gt_coco_json(img_list=img_list, ann_list=ann_list, image_id=image_id, img_path=img_path, label_path=label_path)
        max_category_id = max(max_category_id, max_category_id_gt)

    # creating category list
    if num_classes:
        max_category_id = num_classes
    classmap = [i for i in range(max_category_id + 1)]
    class_names = [str(i) for i in classmap]
    cat_list = create_categories(class_names, classmap)
    # combining all three lists in gt_json {img_list, ann_list, cat_list}
    gt_jdict = create_annotations(img_list=img_list, ann_list=ann_list, cat_list=cat_list)
    
    with open(out_json_path, 'w') as f:
        json.dump(gt_jdict, f)


import argparse

def parser_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='image folder path (required)')
    parser.add_argument('--anno', type=str, default=None, help='annotation folder path')
    parser.add_argument('--out', type=str, default='annotations.json', help='output json path')
    return parser.parse_args()

if __name__ == '__main__':

    args = parser_arguments()
   
    yolo2coco(image_folder_path=args.path, anno_folder_path=args.anno, out_json_path=args.out, num_classes=None)