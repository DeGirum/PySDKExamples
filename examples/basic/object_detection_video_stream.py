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
    target=config_data['target']
    model_zoo_url = config_data['model_zoo_url']
    model_name= config_data['model_name']
    video_source=config_data['video_source']
    # connect to AI inference engine getting token from env.ini file
    zoo = dg.connect(target, 
                     model_zoo_url, 
                     degirum_tools.get_token())
    # load object detection AI model for DeGirum Orca AI accelerator
    model = zoo.load_model(model_name,
                           overlay_show_probabilities=True
                           )
    # AI prediction loop
    # Press 'x' or 'q' to stop
    with degirum_tools.Display("AI Camera") as display:    
        for res in degirum_tools.predict_stream(model, video_source):
            display.show(res)