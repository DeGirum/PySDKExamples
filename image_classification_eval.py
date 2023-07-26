#
# image_classification_eval.py: This script is used to perform inference on Image Classification models using DeGirum PySDK
# Usage: python3 image_classification_eval.py --args-file-path <path>
# Arguments needed for evaluation are provided in the eval_script_args.yaml file. Please update the file with required values.
# Copyright DeGirum Corporation 2023
# All rights reserved
#

import argparse
import sys
from enum import Enum
import degirum as dg
from random import shuffle


def parse_args() -> str:
    """Parse script arguments

    Raises:
        Exception: An exception occurred while parsing script arguments

    Returns:
        str: Path to arguments YAML file required for evaluation
    """
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--args-file-path",
            type=str,
            help="Path to arguments YAML file required for evaluation",
        )
        args = parser.parse_args()
        if args.args_file_path is None:
            raise Exception("Please provide script args path")
        return args.args_file_path
    except Exception as e:
        print("Usage: python image_classification_eval.py -args-file-path <path>")
        sys.exit(1)


def load_dataset_from_path(dataset_path: str) -> list:
    """Load images from dataset path

    Args:
        dataset_path (str): Glob path to dataset

    Returns:
        list: List of images
    """
    from glob import glob

    try:
        images = []
        for img in glob(dataset_path):
            images.append(img)
        return images
    except Exception as e:
        print("Error while getting images dataset: {}".format(e))
        return []


def get_script_args(script_args_yaml_path: str) -> dict:
    """Get script arguments from YAML file

    Args:
        script_args_yaml_path (str): The path to YAML file containing script arguments

    Returns:
        dict: Script arguments
    """
    import yaml

    try:
        with open(script_args_yaml_path, "r") as f:
            script_args = yaml.load(f, Loader=yaml.SafeLoader)
            return script_args
    except Exception as e:
        print("An error occurred while parsing yaml file: {}".format(e))
        return None


class InferenceType(Enum):
    """Inference Type Enum"""

    CloudInference = 1
    AIServerInference = 2
    LocalHWInference = 3


def connect_to_model_zoo(script_args: dict) -> object:
    """Connect to DeGirum Model Zoo

    Args:
        script_args (dict): Script arguments dictionary

    Raises:
        Exception: If invalid inference type is provided

    Returns:
        object: DeGirum Model Zoo object
    """
    try:
        CLOUD_ZOO_URL = script_args["CloudPortalUtils"]["CLOUD_ZOO_URL"]
        if (
            script_args["CloudPortalUtils"]["INFERENCE_TYPE"]
            == InferenceType.CloudInference.value
        ):
            # Inference on Cloud Platform
            DEGIRUM_CLOUD_TOKEN = script_args["CloudPortalUtils"]["DEGIRUM_CLOUD_TOKEN"]
            CLOUD_URL = script_args["CloudPortalUtils"]["CLOUD_URL"]
            cloud_url = "dgcps://" + CLOUD_URL if CLOUD_URL != "" else "cs.degirum.com"
            if CLOUD_ZOO_URL != "":
                cloud_url += "/" + CLOUD_ZOO_URL
            zoo = dg.connect_model_zoo(cloud_url, DEGIRUM_CLOUD_TOKEN)
        elif (
            script_args["CloudPortalUtils"]["INFERENCE_TYPE"]
            == InferenceType.AIServerInference.value
        ):
            hostname = script_args["CloudPortalUtils"]["AI_SERVER_HOSTNAME"]
            if hostname == "":
                hostname = "localhost"
            if CLOUD_ZOO_URL != "":
                # Using Cloud Zoo
                CLOUD_URL = script_args["CloudPortalUtils"]["CLOUD_URL"]
                DEGIRUM_CLOUD_TOKEN = script_args["CloudPortalUtils"][
                    "DEGIRUM_CLOUD_TOKEN"
                ]
                cloud_url = (
                    "https://" + CLOUD_URL if CLOUD_URL != "" else "cs.degirum.com"
                )
                cloud_url += "/" + CLOUD_ZOO_URL
                zoo = dg.connect_model_zoo((hostname, cloud_url), DEGIRUM_CLOUD_TOKEN)
            else:
                # Using Local Zoo
                zoo = dg.connect_model_zoo(hostname)
        elif (
            script_args["CloudPortalUtils"]["INFERENCE_TYPE"]
        ) == InferenceType.LocalHWInference.value:
            DEGIRUM_CLOUD_TOKEN = script_args["CloudPortalUtils"]["DEGIRUM_CLOUD_TOKEN"]
            CLOUD_URL = script_args["CloudPortalUtils"]["CLOUD_URL"]
            cloud_url = "https://" + CLOUD_URL if CLOUD_URL != "" else "cs.degirum.com"
            if CLOUD_ZOO_URL != "":
                cloud_url += "/" + CLOUD_ZOO_URL
            zoo = dg.connect_model_zoo(cloud_url, DEGIRUM_CLOUD_TOKEN)
        else:
            raise Exception("Invalid Inference Type")
        return zoo
    except Exception as e:
        print("An error occurred while connecting to model zoo: {}".format(e))
        return None


def form_ground_truth_mapping(ground_truth_json_path: str) -> dict:
    """Form image to class mapping from ground truth json file

    Args:
        ground_truth_json_path (str): Path to ground truth json file

    Returns:
        dict: Image to class mapping for evaluation
    """
    import json

    try:
        IMAGE_CLASS_MAP = {}
        with open(ground_truth_json_path, "r") as f:
            labels = json.load(f)
        f.close()
        for idx, ele in enumerate(labels):
            for img in ele["images"]:
                IMAGE_CLASS_MAP[img] = ele["category_id"]
        return IMAGE_CLASS_MAP
    except Exception as e:
        print("An error occurred while parsing json file: {}".format(e))
        return None


def extract_dg_results(dg_predictions: object, verbose=False) -> list:
    """Extract results from DeGirum PySDK Object

    Args:
        dg_predictions (object): DeGirum Image Classification Postprocessing Object
        verbose (bool, optional): Option for results directed to stdout . Defaults to False.

    Returns:
        list: Top Predictions
    """
    try:
        top_predicts = []
        for index, pred in enumerate(dg_predictions.results):
            if verbose:
                print(
                    "{}. Predicted Labels: {} with probability: {}".format(
                        index + 1, pred["label"], pred["score"]
                    )
                )
            top_predicts.append(pred["category_id"])
        return top_predicts
    except Exception as e:
        print("An error occurred while extracting dg results: {}".format(e))
        return None


def run_dg_model() -> dict:
    """Run DeGirum Image Classification Model

    Returns:
        dict: Top-1 and Top-5 Accuracy
    """
    try:
        script_args_yaml_path = parse_args()
        script_args = get_script_args(script_args_yaml_path=script_args_yaml_path)
        SUBSET_SIZE = 10
        images = load_dataset_from_path(dataset_path=script_args["DatasetPath"])[
            :SUBSET_SIZE
        ]
        shuffle(images)
        GROUND_TRUTH_MAPPING = form_ground_truth_mapping(
            ground_truth_json_path=script_args["GroundTruthJSONPath"]
        )
        zoo = connect_to_model_zoo(script_args=script_args)
        model_name = script_args["ModelName"]

        top1_correct, top5_correct = 0, 0
        with zoo.load_model(model_name) as model:
            for image_counter, dg_prediction in enumerate(
                model.predict_batch(iter(images))
            ):
                image = images[image_counter].split("/")[-1].strip()
                expected_class = GROUND_TRUTH_MAPPING[image]
                top_dg_predicts = extract_dg_results(dg_prediction, verbose=False)
                if expected_class in top_dg_predicts:
                    top5_correct += 1
                    if expected_class == top_dg_predicts[0]:
                        top1_correct += 1
                image_counter += 1
        top1_accuracy = top1_correct / SUBSET_SIZE
        top5_accuracy = top5_correct / SUBSET_SIZE
        print("Top-1 Accuracy: {}".format(top1_accuracy))
        print("Top-5 Accuracy: {}".format(top5_accuracy))
        return {"top1_accuracy": top1_accuracy, "top5_accuracy": top5_accuracy}
    except Exception as e:
        print("An error occurred while running dg model: {}".format(e))
        return {}


if __name__ == "__main__":
    run_dg_model()

