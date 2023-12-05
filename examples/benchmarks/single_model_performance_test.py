import yaml
import argparse
import degirum as dg, degirum_tools

if __name__ == '__main__':
    # Get configuration data from configuration yaml file
    parser = argparse.ArgumentParser(description='Parse YAML file.')
    parser.add_argument('--config', help='Path to the YAML configuration file', required=True)
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        config_data = yaml.safe_load(file)
    # Set all config options
    hw_location=config_data['hw_location']
    model_zoo_url = config_data['model_zoo_url']
    iterations= config_data['iterations']
    device_family=config_data['device_family']
    # connect to AI inference engine getting token from env.ini file
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
    CW = (62, 19, 16, 16) # column widths
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