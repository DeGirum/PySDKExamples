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
    model_name= config_data['model_name']
    image_source=config_data['image_source']
    # connect to AI inference engine getting token from env.ini file
    zoo = dg.connect(hw_location, model_zoo_url, degirum_tools.get_token())
    # load object detection AI model for DeGirum Orca AI accelerator
    model = zoo.load_model(model_name,
                        overlay_font_scale=1.5,
                        overlay_alpha=1,
                        overlay_show_probabilities=True
                        )
    # perform AI model inference on given image source 
    inference_result = model(image_source)
    # show results of inference
    print(inference_result) # numeric results
    with degirum_tools.Display("AI Camera") as display:    
        display.show_image(inference_result.image_overlay) # graphical results