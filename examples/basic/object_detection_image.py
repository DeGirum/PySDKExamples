#
# object_detection_image.py: AI Inference on Images
#
# Copyright DeGirum Corporation 2023
# All rights reserved
#
# This script performs AI inference on an image and displays the results both in text format and as an annotated image overlay. It takes a YAML configuration file as input, which specifies the hardware location for running inference, the model zoo URL, the name of the model to use for inference, and the source of the image.
#
# Parameters:
# - hw_location (str): Determines where to run inference with options '@cloud' for DeGirum cloud, '@local' for local machine, or an IP address for AI server inference.
# - model_zoo_url (str): Provides the URL or path for the model zoo with options 'cloud_zoo_url' for various inference options, '' for AI server serving models from a local folder, or a path to a JSON file for a single model zoo in case of @local inference.
# - model_name (str): Specifies the name of the model for running AI inference.
# - image_source: Defines the source of the image for inference with options being a path to an image file, a URL of an image, a PIL image object, or a numpy array.
#
# The script utilizes the 'degirum' and 'degirum_tools' modules to connect to the AI inference engine, load the specified model, and perform inference on the provided image source.
#
# Usage:
# python object_detection_image.py --config path/to/config.yaml
#

import yaml
import argparse
import degirum as dg, degirum_tools

if __name__ == "__main__":
    # Get configuration data from configuration yaml file
    parser = argparse.ArgumentParser(description="Parse YAML file.")
    parser.add_argument(
        "--config", help="Path to the YAML configuration file", required=True
    )
    args = parser.parse_args()
    with open(args.config, "r") as file:
        config_data = yaml.safe_load(file)

    # Set all config options
    hw_location = config_data["hw_location"]
    model_zoo_url = config_data["model_zoo_url"]
    model_name = config_data["model_name"]
    image_source = config_data["image_source"]

    # connect to AI inference engine getting token from env.ini file
    zoo = dg.connect(hw_location, model_zoo_url, degirum_tools.get_token())

    # load object detection AI model for DeGirum Orca AI accelerator
    model = zoo.load_model(
        model_name,
        overlay_font_scale=1.5,
        overlay_alpha=1,
        overlay_show_probabilities=True,
    )

    # perform AI model inference on given image source
    inference_result = model(image_source)

    # show results of inference
    print(inference_result)  # numeric results
    with degirum_tools.Display("AI Camera") as display:
        display.show_image(inference_result.image_overlay)  # graphical results
