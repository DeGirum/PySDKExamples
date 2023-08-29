import json
import yaml
from tqdm import tqdm
import csv
import os
from random import shuffle

class ClassificationModelAccuracyEvaluator:
    
    def __init__(
            self,
            dg_model,
            image_folder_path,
            image_label_map_json,
            num_classes,
            k,
            print_frequency=500
      ):
    
        """
        Constructor.
            This class computes the Top-k Accuracy for Classification models.

            Args:
                dg_model (Detection model): Classification model from the Degirum model zoo.
                image_folder_path (str): Path to the image dataset.
                image_labekl_map_json (str) : Path to the image-label-map json file.
                num_classes (int) : Number of classes in the test dataset (Example: ImageNet has 1000 classes).
                k (int) : Th value of `k` in top-k.
                print_frequency (int): Number of image batches to be evaluated at a time.

        """
           
        self.dg_model = dg_model
        self.image_folder_path = image_folder_path
        self.image_label_map_json = image_label_map_json
        self.num_classes = num_classes
        self.k = k
        self.print_frequency = print_frequency
     
    
    def load_images_and_label_map(self):
        
        if os.path.exists(os.path.join(self.image_folder_path, self.image_label_map_json)):
            json_file = open(os.path.join(self.image_folder_path, self.image_label_map_json))
        else:
            raise Exception('Error: {} not found!'.format(self.image_label_map_json))
        labels_map = json.load(json_file)
        images = []
        for path, directory, files in os.walk(self.image_folder_path):
            for file in files:
                if file in labels_map.keys():
                    images.append((file, os.path.join(path, file)))
        return (images,labels_map)
        
    
    @classmethod
    def init_from_yaml(cls, dg_model, config_yaml):
        
        with open(config_yaml) as f:
            load_yaml = yaml.load(f, Loader=yaml.FullLoader)
         
        image_folder_path = load_yaml["ImageFolderPath"]
        image_label_map_json = load_yaml["ImageLabelMap"]
        num_classes = load_yaml["NumClasses"]
        k = load_yaml["K"]
        print_frequency = load_yaml["PrintFrequency"]
        
        return cls(dg_model, image_folder_path, image_label_map_json, num_classes, k, print_frequency)
        
    def top_k_accuracy_evaluate(self):
        top_one_accuracy = 0
        top_k_accuracy = 0
        inference_count = 0
        test_images,image_label_map = self.load_images_and_label_map()
        num_output_classes = len(self.dg_model.label_dictionary)
        if self.k > 0 and self.k <= num_output_classes:
            self.dg_model.output_top_k = int(self.k)
            
        for test_img, test_img_path in test_images:  # Calculate top-1 and top-k accuracies
            category_id = image_label_map[test_img]  # O(1) look-up 
#             print("Inferencing image {}, {}".format(inference_count, test_img_path))
            result = self.dg_model(test_img_path).results
            if len(result) > 0:
                max_category_id = None
                max_score = -10e5
                category_ids = [res_obj['category_id'] + self.num_classes - num_output_classes for res_obj in result]
                max_category_id = result[0]['category_id'] + self.num_classes - num_output_classes
                if category_id == max_category_id:
                    top_one_accuracy += 1
                    top_k_accuracy += 1
                elif category_id in category_ids:
                    top_k_accuracy += 1
                inference_count += 1
                if inference_count % self.print_frequency == 0:
                    print ("Inferencing image {}".format(inference_count))
                    print ("True class: {}, Predicted class: {}".format(category_id, max_category_id))
                    print ("Top-1 accuracy: {}, Top-{} accuracy: {}".format(top_one_accuracy * 100 / inference_count, self.dg_model.output_top_k, top_k_accuracy * 100 / inference_count))
                    print("\n")
        print("Top-1 accuracy is {}".format(top_one_accuracy * 100 / inference_count))
        print("Top-{} accuracy is {}".format(self.dg_model.output_top_k, top_k_accuracy * 100 / inference_count))
        return (self.dg_model.output_top_k, top_one_accuracy * 100 / inference_count, top_k_accuracy * 100 / inference_count)
        