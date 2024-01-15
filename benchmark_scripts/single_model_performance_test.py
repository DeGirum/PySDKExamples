
import degirum as dg
import degirum_tools
import argparse

#
# Please UNCOMMENT only ONE of the following lines to specify where to run AI inference
#

target = dg.CLOUD # <-- on the Cloud Platform
# target = degirum_tools.get_ai_server_hostname() # <-- on AI Server deployed in your LAN
# target = dg.LOCAL # <-- on ORCA accelerator installed on this computer

# connect to AI inference engine getting zoo URL and token from env.ini file
zoo = dg.connect(target, degirum_tools.get_cloud_zoo_url(), degirum_tools.get_token())

# define function to run a single model batch prediction
def do_test(model_name, 
            iterations:int = 100,
            use_jpeg: bool = True,
            batch_size = None,
            exclude_preprocessing: bool = True,

            ):

    # load model
    with zoo.load_model(model_name) as model:

        # skip non-image type models
        if model.model_info.InputType[0] != "Image":
            return None

        # configure model
        model.input_image_format = "JPEG" if use_jpeg else "RAW"
        model.measure_time = True
        if batch_size is not None:
            model.eager_batch_size = batch_size
            model.frame_queue_depth = batch_size

        # prepare input frame
        frame = "images/TwoCats.jpg"
        if exclude_preprocessing:
            frame = model._preprocessor.forward(frame)[0]

        # define source of frames
        def source():
            for _ in range(iterations):
                yield frame

        model(frame)  # run model once to warm up the system

        # run batch prediction
        t = degirum_tools.Timer()
        for res in model.predict_batch(source()):
            pass

        return {
            "model_name": model_name,
            "iterations": iterations,
            "postprocess_type": model.output_postprocess_type,
            "elapsed": t(),
            "time_stats": model.time_stats(),
        }

def print_result(res, print_header : bool = False):
    latency_ms = res["time_stats"]["FrameTotalDuration_ms"].avg
    inference_ms = res["time_stats"]["CoreInferenceDuration_ms"].avg
    frame_duration_ms = 1e3 * res["elapsed"] / res["iterations"]

    # print results
    if print_header:
        print(
            f"{'Model name':62}| {'Postprocess type':19} | {'Observed FPS':12} | {'Expected FPS':12} | "
        )

    print(
        f"{res['model_name']:62}|"
        + f" {res['postprocess_type']:19} |"
        + f" {1e3 / frame_duration_ms:12.1f} |"
        + f" {1e3 / inference_ms:12.1f} |"
    )

def parser_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=str, nargs='+', help='model name')
    parser.add_argument('--iterations', type=int, default=100, help='how many iterations to run for each model')
    parser.add_argument('--raw', action='store_true', default=False, help='use RAW instead of JPEG input')
    parser.add_argument('--include-pp', action='store_true', default=False, help='include preprocessing step from timing measurements')
    parser.add_argument('--batch-size', type=str, default=None, help='eager batch size to test; None to use default')

    return parser.parse_args()

if __name__ == '__main__':

    args = parser_arguments()

    for i, model_name in enumerate(args.models):
        res = do_test(model_name, 
                      iterations=args.iterations, 
                      use_jpeg=not args.raw, 
                      exclude_preprocessing=not args.include_pp, 
                      batch_size=args.batch_size
                      )
        
        print_result(res, print_header=i==0)