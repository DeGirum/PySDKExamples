import degirum as dg
import degirum_tools
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


def run_inference(model, image_batch):
    """Run inference on a given model and batch of images."""
    results = list(
        model.predict_batch(image_batch)
    )  # Convert to list to ensure full execution
    return model.name, results


def main():
    # Load the models
    model_names = [
        "mobilenet_v1_imagenet--224x224_quant_n2x_orca1_1",
        "mobilenet_v2_imagenet--224x224_quant_n2x_orca1_1",
        "resnet50_imagenet--224x224_pruned_quant_n2x_orca1_1",
        "efficientnet_es_imagenet--224x224_quant_n2x_orca1_1",
    ]

    inference_host_address = "@cloud"  # Example inference location
    zoo_url = "degirum/public"  # Example model zoo URL
    token = degirum_tools.get_token()  # Get token from environment or configuration

    models = []
    for model_name in model_names:
        # Load each model from the model zoo
        model = dg.load_model(
            model_name=model_name,
            inference_host_address=inference_host_address,
            zoo_url=zoo_url,
            token=token,
        )
        models.append(model)

    # Prepare the batch of images (same image repeated 100 times)
    image_source = "https://raw.githubusercontent.com/DeGirum/PySDKExamples/main/images/ThreePersons.jpg"
    image_batch = [image_source] * 100

    # Step 1: Measure time for sequential execution (x1, x2, x3, x4)
    sequential_times = []
    for model in models:
        start_time = time.perf_counter()
        results = list(
            model.predict_batch(image_batch)
        )  # Convert to list to ensure full execution
        end_time = time.perf_counter()
        sequential_time = end_time - start_time
        sequential_times.append(sequential_time)
        print(f"Time for model {model.name}: {sequential_time:.2f} seconds")

    # Calculate total sequential time
    total_sequential_time = sum(sequential_times)
    print(f"Total time for sequential execution: {total_sequential_time:.2f} seconds")

    # Step 2: Measure time for parallel execution (y) using ProcessPoolExecutor
    start_parallel_time = time.perf_counter()
    with ProcessPoolExecutor() as executor:
        # Run inference on all models in parallel
        futures = [
            executor.submit(run_inference, model, image_batch) for model in models
        ]
        for future in as_completed(futures):
            model_name, results = future.result()
            print(f"Completed inference for model: {model_name}")

    end_parallel_time = time.perf_counter()
    parallel_time = end_parallel_time - start_parallel_time

    print(f"Time for parallel execution: {parallel_time:.2f} seconds")

    # Step 3: Calculate and display model multiplexing efficiency
    model_multiplexing_efficiency = total_sequential_time / parallel_time
    print(f"Model Multiplexing Efficiency: {model_multiplexing_efficiency:.2f}")


if __name__ == "__main__":
    main()
