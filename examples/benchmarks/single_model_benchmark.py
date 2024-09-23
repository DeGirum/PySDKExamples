#
# object_detection_image.py: AI Inference on Multiple Models with FPS Calculation using Config File and Command Line Overrides
#
# This script performs AI inference on multiple models using a pre-built utility for profiling and calculates the frames per second (FPS).
# Parameters can be provided via a YAML configuration file or use default values. Some parameters can be overridden by command-line arguments.
#

import yaml
import argparse
import degirum as dg
import degirum_tools
import os


def load_config(config_file):
    """Load parameters from the YAML configuration file if provided."""
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run AI inference on multiple models using a config file."
    )
    parser.add_argument(
        "--config_file",
        type=str,
        help="Optional path to the YAML configuration file that contains model names, iterations, and other parameters.",
    )
    parser.add_argument(
        "--inference_host_address",
        type=str,
        help="Override inference host address from the config file. Example: @cloud, @local, or an IP address.",
    )
    parser.add_argument(
        "--device_type",
        type=str,
        help="Override device type from the config file. Example: CPU, GPU.",
    )
    args = parser.parse_args()

    # Default values (used if no config file is provided)
    default_config = {
        "inference_host_address": "@cloud",
        "model_zoo_url": "degirum/public",
        "model_names": [
            "mobilenet_v1_imagenet--224x224_quant_n2x_orca1_1",
            "mobilenet_v2_imagenet--224x224_quant_n2x_orca1_1",
            "resnet50_imagenet--224x224_pruned_quant_n2x_orca1_1",
            "efficientnet_es_imagenet--224x224_quant_n2x_orca1_1",
            "efficientdet_lite1_coco--384x384_quant_n2x_orca1_1",
            "mobiledet_coco--320x320_quant_n2x_orca1_1",
            "yolov8n_relu6_coco--640x640_quant_n2x_orca1_1",
            "yolov8n_relu6_face--640x640_quant_n2x_orca1_1",
            "deeplab_seg--513x513_quant_n2x_orca1_1",
        ],
        "iterations": 100,
        "device_type": None,
        "token": None,
    }

    # Load the config file if provided, otherwise use default values
    if args.config_file:
        config = load_config(args.config_file)
    else:
        config = default_config

    # Override config values with command-line arguments if provided
    inference_host_address = args.inference_host_address or config.get(
        "inference_host_address", default_config["inference_host_address"]
    )
    device_type = args.device_type or config.get(
        "device_type", default_config["device_type"]
    )

    # Extract other parameters from the config
    model_zoo_url = config.get("model_zoo_url", default_config["model_zoo_url"])
    model_names = config.get("model_names", default_config["model_names"])
    iterations = config.get("iterations", default_config["iterations"])
    token = (
        config.get("token") or degirum_tools.get_token()
    )  # Token from config or environment

    if not token:
        print(
            "Error: Please provide a cloud platform token in the config file or ensure it is retrievable from the environment."
        )
        return

    # Print the number of models and iterations
    print(f"Models    : {len(model_names)}")
    print(f"Iterations: {iterations}\n")

    # Print the header
    CW = (62, 19, 16, 16)  # column widths
    header = f"{'Model name':{CW[0]}}| {'Postprocess Type':{CW[1]}} | {'Observed FPS':{CW[2]}} | {'Max Possible FPS':{CW[3]}} |"
    print(f"{'-'*len(header)}")
    print(header)
    print(f"{'-'*len(header)}")

    # Loop through each model name, measure FPS, and print results
    for model_name in model_names:
        # Prepare arguments for loading the model
        model_args = {
            "model_name": model_name,
            "inference_host_address": inference_host_address,
            "zoo_url": model_zoo_url,
            "token": token,
        }

        # Optionally add device_type to the model arguments if provided
        if device_type:
            model_args["device_type"] = device_type

        # Load the model
        model = dg.load_model(**model_args)

        # Use the model_time_profile utility to measure the FPS
        result = degirum_tools.model_time_profile(model, iterations)

        # Print the result for the current model
        print(
            f"{model_name:{CW[0]}}|"
            + f" {result.parameters.OutputPostprocessType:{CW[1]}} |"
            + f" {result.observed_fps:{CW[2]}.1f} |"
            + f" {result.max_possible_fps:{CW[3]}.1f} |"
        )


if __name__ == "__main__":
    main()
