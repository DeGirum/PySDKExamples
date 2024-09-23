#
# object_detection_video_stream.py: AI Inference on a Video Stream
#
# This script runs AI inference on a video stream using command-line arguments and displays the video with annotated results.
#

import argparse
import degirum as dg
import degirum_tools
import os


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run AI inference on a video stream.")
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
        "--video_source",
        type=str,
        default="https://raw.githubusercontent.com/DeGirum/PySDKExamples/main/images/example_video.mp4",
        help="Video source for inference. Can be a camera index (0, 1), RTSP stream URL, YouTube URL, or path to a video file. Default is an example video.",
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

    # Determine if video source is a camera index (integer) or a string (URL, path)
    if args.video_source.isdigit():
        video_source = int(args.video_source)
    else:
        video_source = args.video_source

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

    # Run AI inference on video stream
    inference_results = degirum_tools.predict_stream(model, video_source)

    # Display the results with a live video stream
    # Press 'x' or 'q' to stop
    with degirum_tools.Display("AI Camera") as display:
        for inference_result in inference_results:
            display.show(inference_result)


if __name__ == "__main__":
    main()
