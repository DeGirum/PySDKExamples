#
# object_detection_image.py: AI Inference on Images
#
# This script performs AI inference on an image using command-line arguments and displays the results both in text format and as an annotated image overlay.
#

import argparse
import degirum as dg
import degirum_tools
import os


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run AI inference on an image.")
    parser.add_argument(
        "--inference_host_address",
        type=str,
        default="@cloud",
        help="Hardware location for inference, e.g., @cloud (default), @local, or IP address.",
    )
    parser.add_argument(
        "--model_zoo_url",
        type=str,
        default="degirum/public",
        help="URL or path to the model zoo. Default is 'degirum/public'.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="yolov8n_relu6_coco--640x640_quant_n2x_orca1_1",
        help="Name of the model to use for inference. Default is 'yolov8n_relu6_coco--640x640_quant_n2x_orca1_1'.",
    )
    parser.add_argument(
        "--image_source",
        type=str,
        default="https://raw.githubusercontent.com/DeGirum/PySDKExamples/main/images/ThreePersons.jpg",
        help="Path or URL to the image for inference. Default is an example image.",
    )
    parser.add_argument(
        "--device_type",
        type=str,
        help="Optional device type for inference (e.g., CPU, GPU). If not provided, the default device will be used.",
    )
    parser.add_argument(
        "--token",
        type=str,
        help="Cloud platform token to use for inference. If not provided, will attempt to load from degirum_tools.get_token().",
    )

    args = parser.parse_args()

    # Get the token from the command-line argument or degirum_tools.get_token()
    token = args.token or degirum_tools.get_token()
    if not token:
        print(
            "Error: Please provide the cloud platform token using '--token' or ensure the token is retrievable from degirum_tools.get_token()."
        )
        return

    # Prepare load_model arguments
    model_args = {
        "model_name": args.model_name,
        "inference_host_address": args.inference_host_address,
        "zoo_url": args.model_zoo_url,
        "token": token,
    }

    # Add device_type to model_args if it's provided
    if args.device_type:
        model_args["device_type"] = args.device_type

    # Load the AI model with optional device_type
    model = dg.load_model(**model_args)

    # Perform AI model inference on the given image source
    inference_result = model(args.image_source)

    # Display the results (numeric output always)
    print("Inference Result:", inference_result)  # Numeric results

    # Check if the script is running in a display environment
    if (
        os.environ.get("DISPLAY") or os.name == "nt"
    ):  # DISPLAY is usually set in graphical environments
        # If DISPLAY exists (Linux/macOS) or on Windows, show the graphical result
        try:
            with degirum_tools.Display("AI Camera") as display:
                display.show_image(inference_result)  # Graphical results
        except Exception as e:
            print(f"Error displaying results: {e}")
    else:
        print("No display found. Skipping graphical output.")


if __name__ == "__main__":
    main()
