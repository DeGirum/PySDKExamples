import json
import os
import yaml
import numpy as np
from custom_postprocessors.AgeGenderRecognition import AgeGenderRecognition

class AgeGenderRecognitionModelEvaluator:
    
    def __init__(
        self,
        dg_model,
        image_folder_path,
        ground_truth_annotations_path
    ):
        """
        Constructor.
            This class evaluates the Accuracy and Mean Absolute Error for Age Gender Recognition models.

            Args:
                dg_model (model): Age-Gender model from the Degirum model zoo.
                image_folder_path (str): Path to the image dataset.
                ground_truth_annotations_path (str): Path to the groundtruth json annotations.
        """

        self.dg_model = dg_model
        self.image_folder_path = image_folder_path
        self.ground_truth_annotations_path = ground_truth_annotations_path
        
    @classmethod
    def init_from_yaml(cls, dg_model, config_yaml):
        """
        args_yaml (str) : Path of the yaml file that contains all the arguments.

        """
        with open(config_yaml) as f:
            load_yaml = yaml.load(f, Loader=yaml.FullLoader)

        image_folder_path = load_yaml["ImageFolderPath"]
        ground_truth_annotations_path = load_yaml["GroundTruthAnnotationsPath"]

        return cls(
            dg_model,
            image_folder_path,
            ground_truth_annotations_path
        )
    
    def evaluate(self):
        """Evaluation for the Age Gender Recognition model.

        Returns the Accuracy for Gender and Mean Absolute Error for Age.
        """
        with open(self.ground_truth_annotations_path, 'r') as json_file:
            data = json.load(json_file)
        age_groundtruth=[]
        gender_groundtruth=[]
        for i in data:
            age_groundtruth.append(int(i["age"]))
            gender_groundtruth.append(int(i["gender"])) 
            
        gender_predictions = []
        age_predictions = []
        for image_name in os.listdir(self.image_folder_path):
            image_path = os.path.join(self.image_folder_path, image_name)            
            self.dg_model.custom_postprocessor = AgeGenderRecognition
            res=self.dg_model(image_path)   
            gender_predictions.append(res.results[0]["gender_id"])
            age_predictions.append(res.results[0]["age"])
        gender_accuracy = np.mean(np.array(gender_predictions) == np.array(gender_groundtruth))
        mae_age = np.mean(np.abs(np.array(age_predictions) - np.array(age_groundtruth)))
        return (gender_accuracy,mae_age)
