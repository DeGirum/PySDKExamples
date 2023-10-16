import os
import yaml
import json

class ImageClassificationModelEvaluator:
    
    def __init__(
            self,
            dg_model,
            image_folder_path,
            image_label_map_json,
            num_classes,
            top_k,
            input_resize_method="bilinear",
            input_pad_method="letterbox",
            image_backend="opencv",
            input_img_fmt="JPEG",
            print_frequency=500
      ):
    
        """
        Constructor.
            This class computes the Top-k Accuracy for Classification models.

            Args:
                dg_model (Detection model): Classification model from the Degirum model zoo.
                image_folder_path (str): Path to the image dataset.
                image_label_map_json (str) : Path to the image-label-map json file.
                num_classes (int) : Number of classes in the test dataset (Example: ImageNet has 1000 classes).
                k (int) : The value of `k` in top-k.
                input_resize_method (str): Input Resize Method.
                input_pad_method (str): Input Pad Method.
                image_backend (str): Image Backend.
                input_img_fmt (str): InputImgFmt.
                print_frequency (int): Number of image batches to be evaluated at a time.

        """
           
        self.dg_model = dg_model
        self.image_folder_path = image_folder_path
        self.image_label_map_json = image_label_map_json
        self.num_classes = num_classes
        self.top_k = top_k
        self.input_resize_method = input_resize_method
        self.input_pad_method = input_pad_method
        self.image_backend = image_backend
        self.input_img_fmt = input_img_fmt
        self.print_frequency = print_frequency
        
        
        if (self.dg_model.output_postprocess_type == "Classification"): 
            self.dg_model.input_resize_method = self.input_resize_method
            self.dg_model.input_pad_method = self.input_pad_method
            self.dg_model.image_backend = self.image_backend
            self.dg_model.input_image_format = self.input_img_fmt
        else:
            raise Exception("Model loaded for evaluation is not a Classification Model")
    
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
        top_k = load_yaml["TopK"]
        input_resize_method = load_yaml["InputResizeMethod"]
        input_pad_method = load_yaml["InputPadMethod"]
        image_backend = load_yaml["ImageBackend"]
        input_image_format = load_yaml["InputImgFmt"]
        print_frequency = load_yaml["PrintFrequency"]
        
        return cls(dg_model, image_folder_path, image_label_map_json, num_classes, top_k, input_resize_method, input_pad_method, image_backend, input_image_format, print_frequency)
        
    def evaluate(self):
        top_one_accuracy = 0
        top_k_accuracy = 0
        inference_count = 0
        test_images,image_label_map = self.load_images_and_label_map()
        num_output_classes = len(self.dg_model.label_dictionary)
        if self.top_k > 0 and self.top_k <= num_output_classes:
            self.dg_model.output_top_k = int(self.top_k)
            
        for test_img, test_img_path in test_images:  # Calculate top-1 and top-k accuracies
            category_id = image_label_map[test_img]  # O(1) look-up 
#             print("Inferencing image {}, {}".format(inference_count, test_img_path))
            result = self.dg_model(test_img_path).results
            if len(result) > 0:
                top_category_id = None
                max_score = -10e5
                top_k_category_ids = [res_obj['category_id'] + self.num_classes - num_output_classes for res_obj in result]
                top_category_id = result[0]['category_id'] + self.num_classes - num_output_classes
                if category_id == top_category_id:
                    top_one_accuracy += 1
                    top_k_accuracy += 1
                elif category_id in top_k_category_ids:
                    top_k_accuracy += 1
                inference_count += 1
                if inference_count % self.print_frequency == 0:
                    print ("Inferencing image {}".format(inference_count))
                    print ("True class: {}, Predicted class: {}".format(category_id, top_category_id))
                    print ("Top-1 accuracy: {}, Top-{} accuracy: {}".format(top_one_accuracy * 100 / inference_count, self.dg_model.output_top_k, top_k_accuracy * 100 / inference_count))
                    print("\n")
        print("Top-1 accuracy is {}".format(top_one_accuracy * 100 / inference_count))
        print("Top-{} accuracy is {}".format(self.dg_model.output_top_k, top_k_accuracy * 100 / inference_count))
        return (self.dg_model.output_top_k, top_one_accuracy * 100 / inference_count, top_k_accuracy * 100 / inference_count)
        
