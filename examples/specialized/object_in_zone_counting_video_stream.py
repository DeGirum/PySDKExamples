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
    polygon_zones = config_data["polygon_zones"]
    class_list = config_data["class_list"]
    # connect to AI inference engine getting token from env.ini file
    zoo = dg.connect(hw_location, model_zoo_url, degirum_tools.get_token())
    # load object detection AI model for DeGirum Orca AI accelerator
    # load model
    model = zoo.load_model(model_name, overlay_line_width=2, overlay_font_scale=1.0)
    # AI prediction loop
    # Press 'x' or 'q' to stop
    # Drag zone by left mouse button to move zone
    # Drag zone corners by right mouse button to adjust zone shape
    with degirum_tools.Display("AI Camera") as display:
        # create zone counter
        zone_counter = degirum_tools.ZoneCounter(
            polygon_zones,
            class_list=class_list,
            triggering_position=degirum_tools.ZoneCounter.CENTER,
            window_name=display.window_name,  # attach display window for interactive zone adjustment
        )

        # do AI predictions on video stream
        for inference_result in degirum_tools.predict_stream(
            model, video_source, zone_counter=zone_counter
        ):
            display.show(inference_result)
