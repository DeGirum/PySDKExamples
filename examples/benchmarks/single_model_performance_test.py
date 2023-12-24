#
# single_model_performance_test.py: AI Model Performance Profiling
#
# Copyright DeGirum Corporation 2023
# All rights reserved
#
# This script is designed to connect to an AI inference engine, list available AI models,
# and run a batch prediction for each model to record time measurements. It generates a
# performance report that includes the observed frames per second (FPS) and the maximum
# possible FPS for each model.
#
# Parameters:
# - --config: Path to the YAML configuration file containing the following keys:
#     * hw_location (str): Specifies where to run inference. Options are:
#         - '@cloud': Use DeGirum cloud for inference.
#         - '@local': Run inference on the local machine.
#         - 'IP address': Specify the IP address of the AI server for inference.
#     * model_zoo_url (str): URL or path for the model zoo. Options are:
#         - 'cloud_zoo_url': Valid for @cloud, @local, and AI server inference options.
#         - '': Indicates the AI server is serving models from a local folder.
#         - 'path to json file': Path to a single model zoo JSON file in case of @local inference.
#     * iterations (int): Number of iterations to run for each model during profiling.
#     * device_family (str): Device family of models to profile.
#
# The script requires the 'degirum' and 'degirum_tools' modules to interact with the
# AI inference engine and perform the profiling tasks.
#
# Usage:
# python script_name.py --config path/to/config.yaml
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
    iterations = config_data["iterations"]
    device_family = config_data["device_family"]

    # connect to AI inference engine
    zoo = dg.connect(hw_location, model_zoo_url, degirum_tools.get_token())

    # list of models to test
    model_names = zoo.list_models(device=device_family)

    # run batch predict for each model and record time measurements
    results = {}
    prog = degirum_tools.Progress(len(model_names), speed_units="models/s")
    for model_name in model_names:
        try:
            results[model_name] = degirum_tools.model_time_profile(
                zoo.load_model(model_name), iterations
            )
        except NotImplementedError:
            pass  # skip models for which time profiling is not supported
        prog.step()

    # print results
    CW = (62, 19, 16, 16)  # column widths
    header = f"{'Model name':{CW[0]}}| {'Postprocess Type':{CW[1]}} | {'Observed FPS':{CW[2]}} | {'Max Possible FPS':{CW[3]}} |"

    print(f"Models    : {len(model_names)}")
    print(f"Iterations: {iterations}\n")
    print(f"{'-'*len(header)}")
    print(header)
    print(f"{'-'*len(header)}")

    for model_name, result in results.items():
        print(
            f"{model_name:{CW[0]}}|"
            + f" {result.parameters.OutputPostprocessType:{CW[1]}} |"
            + f" {result.observed_fps:{CW[2]}.1f} |"
            + f" {result.max_possible_fps:{CW[3]}.1f} |"
        )
