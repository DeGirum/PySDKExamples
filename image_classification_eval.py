import degirum as dg
from mytools import connect_model_zoo


def arg_parser():
    try:
        import argparse
        import sys

        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", type=str, help="Dataset Path")
        parser.add_argument(
            "--config-json",
            type=str,
            help="DeGirum Cloud Zoo JSON Config File Path",
        )
        parser.add_argument(
            "--ground-truth-json", type=str, help="Ground Truth JSON File Path"
        )
        args = parser.parse_args()
        if (
            args.dataset is None
            or args.config_json is None
            or args.ground_truth_json is None
        ):
            raise Exception("Config JSON Path or Ground Truth JSON Path is None")
        return args.dataset, args.config_json, args.ground_truth_json
    except Exception as e:
        print(
            "python3 image_classification_eval.py --dataset-path <path> --config-json-path <path> --ground-truth-json-path <path>"
        )
        sys.exit(1)


def load_dataset_from_path(dataset_path: str):
    from glob import glob

    try:
        images = []
        for img in glob(dataset_path):
            images.append(img)
        return images
    except Exception as e:
        print("Error while getting images dataset: {}".format(e))
        return []


def parse_json_config(config_json_path: str) -> dict:
    import json

    try:
        with open(config_json_path, "r") as f:
            config = json.load(f)
            return config
    except Exception as e:
        print("An error occurred while parsing json file: {}".format(e))
        return None


def get_model_name(config: dict) -> str:
    try:
        model_name = config["MODEL_PARAMETERS"][0]["ModelPath"]
        return model_name.split(".")[0].strip()
    except Exception as e:
        print("An error occurred while getting model name: {}".format(e))
        return None


def form_ground_truth_mapping(ground_truth_json_path: str) -> dict:
    import json

    try:
        IMAGE_CLASS_MAP = {}
        with open(ground_truth_json_path, "r") as f:
            labels = json.load(f)
        f.close()
        for idx, ele in enumerate(labels):
            for img in ele["images"]:
                IMAGE_CLASS_MAP[img] = idx
        return IMAGE_CLASS_MAP
    except Exception as e:
        print("An error occurred while parsing json file: {}".format(e))
        return None


def extract_dg_results(dg_predictions: object, verbose=False) -> list:
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


def run_dg_model():
    # dataset_path, config_json_path, ground_truth_json_path = arg_parser()
    ## Developmental Code
    from random import shuffle

    dataset_path = "/home/degirum/srv1.share/exchange/ml_data/imagenet/val_images/*/*"
    config_json_path = "efficientnet_es_imagenet--224x224_float_n2x_orca_1.json"
    ground_truth_json_path = "imagenet_labels.json"
    SUBSET_SIZE = 10
    ##
    images = load_dataset_from_path(dataset_path=dataset_path)[:SUBSET_SIZE]
    shuffle(images)
    GROUND_TRUTH_MAPPING = form_ground_truth_mapping(
        ground_truth_json_path=ground_truth_json_path
    )
    zoo = connect_model_zoo()
    config = parse_json_config(config_json_path=config_json_path)
    model_name = get_model_name(config)

    top1_correct, top5_correct = 0, 0
    with zoo.load_model(model_name) as model:
        image_counter = 0
        for dg_prediction in model.predict_batch(iter(images)):
            image = images[image_counter].split("/")[-1].strip()
            expected_class = GROUND_TRUTH_MAPPING[image]
            top_dg_predicts = extract_dg_results(dg_prediction, verbose=False)
            if expected_class in top_dg_predicts:
                top5_correct += 1
                if expected_class == top_dg_predicts[0]:
                    top1_correct += 1
            image_counter += 1
    print("Top-1 Accuracy: {}".format(top1_correct / SUBSET_SIZE))
    print("Top-5 Accuracy: {}".format(top5_correct / SUBSET_SIZE))


if __name__ == "__main__":
    run_dg_model()
