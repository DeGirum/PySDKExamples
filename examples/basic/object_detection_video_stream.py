#
# object_detection_video_stream.py: Simple python script to run AI inference on a video stream.
#
# Copyright DeGirum Corporation 2023
# All rights reserved
#
# This script runs AI inference on a video source and displays the video stream with annotated results.
# The script take a config.yaml file as an input. Tha config file specifies the following parameters.
# hw_location: where you want to run inference
#     @cloud to use DeGirum cloud
#     @local to run on local machine
#     IP address for AI server inference
# model_zoo_url: url/path for model zoo
#     cloud_zoo_url: valid for @cloud, @local, and ai server inference options
#     '': ai server serving models from local folder
#     path to json file: single model zoo in case of @local inference
# model_name: name of the model for running AI inference
# video_source: video source for inference
#     camera index for local web camera
#     URL of RTSP stream
#     URL of YouTube Video
#     path to video file (mp4 etc)

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
    video_source = config_data["video_source"]

    # connect to AI inference engine getting token from env.ini file
    zoo = dg.connect(hw_location, model_zoo_url, degirum_tools.get_token())

    # load object detection AI model for DeGirum Orca AI accelerator
    model = zoo.load_model(model_name, overlay_show_probabilities=True)

    # AI prediction loop
    # Press 'x' or 'q' to stop
    with degirum_tools.Display("AI Camera") as display:
        for inference_result in degirum_tools.predict_stream(model, video_source):
            display.show(inference_result)
