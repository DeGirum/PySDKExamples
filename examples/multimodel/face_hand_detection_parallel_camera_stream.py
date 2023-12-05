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
    hand_det_model_name= config_data['hand_det_model_name']
    face_det_model_name= config_data['face_det_model_name']
    person_det_model_name= config_data['person_det_model_name']
    video_source=config_data['video_source']    
    # connect to AI inference engine getting token from env.ini file
    zoo = dg.connect(hw_location, model_zoo_url, degirum_tools.get_token())
    # load models for hand, face, and person detection
    hand_det_model = zoo.load_model(hand_det_model_name)
    face_det_model = zoo.load_model(face_det_model_name)
    person_det_model = zoo.load_model(person_det_model_name, 
                                      overlay_line_width = 2
                                      )
    # AI prediction loop, press 'x' or 'q' to stop video
    with degirum_tools.Display("Hands, Faces, and Persons") as display:
        for inference_result in degirum_tools.predict_stream(degirum_tools.CombiningCompoundModel(degirum_tools.CombiningCompoundModel(hand_det_model,
                                                                                                                          face_det_model
                                                                                                                          ), 
                                                                                    person_det_model
                                                                                    ), 
                                                video_source
                                                ):
            display.show(inference_result)