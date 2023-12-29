# object_in_zone_counting_video_stream.py: AI Inference with Object Counting in Zones on Video Streams
#
# Copyright DeGirum Corporation 2023
# All rights reserved
#
# This script performs AI inference on a video stream using a specified model and counts objects within defined polygon zones. It displays the video with annotated results and object counts. The script requires a YAML configuration file as input, which specifies the hardware location for running inference, the model zoo URL, the name of the model for AI inference, the source of the video, the polygon zones for counting, the list of classes to be counted, and whether to display per class counts.
#
# Parameters:
# - hw_location (str): Specifies where to run inference with options '@cloud' for DeGirum cloud, '@local' for local machine, or an IP address for AI server inference.
# - model_zoo_url (str): Provides the URL or path for the model zoo with options 'cloud_zoo_url' for various inference options, '' for AI server serving models from a local folder, or a path to a JSON file for a single model zoo in case of @local inference.
# - model_name (str): Specifies the name of the model for running AI inference.
# - video_source: Defines the source of the video for inference with options being a camera index for local camera, a URL of an RTSP stream, a URL of a YouTube video, or a path to a video file (e.g., mp4).
# - polygon_zones (list): Specifies the zones in which objects need to be counted, defined as a list of polygon points.
# - class_list (list): Specifies the list of classes to be counted.
# - per_class_display (bool): Specifies if per class counts are to be displayed.
#
# The script uses the 'degirum' and 'degirum_tools' modules to connect to the AI inference engine, load the specified model, and perform inference on the provided video source with interactive zone adjustment for object counting.
#
# Usage:
# python object_in_zone_counting_video_stream.py --config path/to/config.yaml
#

import yaml
import argparse
import degirum as dg
import degirum_tools

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
    polygon_zones = config_data["polygon_zones"]
    class_list = config_data["class_list"]
    # connect to AI inference engine
    zoo = dg.connect(hw_location, model_zoo_url, degirum_tools.get_token())
    # load object detection AI model for DeGirum Orca AI accelerator
    # load model
    model = zoo.load_model(model_name, overlay_line_width=2, overlay_font_scale=1.0)

    with degirum_tools.Display("AI Camera") as display:
        # create zone counter
        zone_counter = degirum_tools.ZoneCounter(
            polygon_zones,
            class_list=class_list,
            triggering_position=degirum_tools.AnchorPoint.CENTER,
            window_name=display.window_name,  # attach display window for interactive zone adjustment
        )

        # AI prediction loop
        # Press 'x' or 'q' to stop
        # Drag zone by left mouse button to move zone
        # Drag zone corners by right mouse button to adjust zone shape
        for inference_result in degirum_tools.predict_stream(
            model, video_source, zone_counter=zone_counter
        ):
            display.show(inference_result)
