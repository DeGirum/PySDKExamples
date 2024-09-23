import degirum as dg
import degirum_tools
import os
import time


def main():
    # Model names
    model_names = [
        "yolov8n_relu6_face--640x640_quant_n2x_orca1_1",
        "yolov8n_relu6_hand--640x640_quant_n2x_orca1_1",
        "yolov8n_relu6_car--640x640_quant_n2x_orca1_1",
        "yolov8n_relu6_lp--640x640_quant_n2x_orca1_1",
    ]

    inference_host_address = "@local"  # Example inference location
    zoo_url = "degirum/models_n2x"  # Example model zoo URL
    token = degirum_tools.get_token()  # Get token from environment or configuration

    # Load the models, store the model and its name together
    models = []
    for model_name in model_names:
        model = dg.load_model(
            model_name=model_name,
            inference_host_address=inference_host_address,
            zoo_url=zoo_url,
            token=token,
        )
        models.append((model_name, model))  # Store as a tuple (model_name, model)

    # Prepare the batch of images (same image repeated 100 times)
    image_source = "../../images/ThreePersons.jpg"
    image_batch = [image_source] * 100

    # Step 0: Warmup inference to avoid cold-start overhead
    print("Running warmup inference...")
    for model_name, model in models:
        list(
            model.predict_batch([image_source])
        )  # Run inference on a single image to warm up
    print("Warmup inference completed.\n")

    # Step 1: Measure time for sequential execution (x1, x2, x3, x4)
    sequential_times = []
    for model_name, model in models:
        start_time = time.perf_counter()
        results = list(
            model.predict_batch(image_batch)
        )  # Convert to list to ensure full execution
        end_time = time.perf_counter()
        sequential_time = end_time - start_time
        sequential_times.append(sequential_time)
        print(f"Time for model {model_name}: {sequential_time:.2f} seconds")

    # Calculate total sequential time
    total_sequential_time = sum(sequential_times)
    print(f"Total time for sequential execution: {total_sequential_time:.2f} seconds")

    # Step 2: Measure time for simulated parallel execution using zip
    start_parallel_time = time.perf_counter()

    # Use predict_batch for all models and zip them together to run through the batches in parallel
    batch_generators = [model.predict_batch(image_batch) for _, model in models]

    for results in zip(*batch_generators):
        for (model_name, _), result in zip(models, results):
            pass  # Simulate using the result, or print it if needed

    end_parallel_time = time.perf_counter()
    parallel_time = end_parallel_time - start_parallel_time

    print(f"Time for simulated parallel execution: {parallel_time:.2f} seconds")

    # Step 3: Calculate and display model multiplexing efficiency
    model_multiplexing_efficiency = total_sequential_time / parallel_time
    print(f"Model Multiplexing Efficiency: {model_multiplexing_efficiency:.2f}")


if __name__ == "__main__":
    main()
