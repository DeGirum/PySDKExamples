import pandas as pd
import csv
import argparse

def read_csv_to_array_of_arrays(csv_file_path):
    """
    Reads a CSV file and returns its contents as an array of arrays.
    
    Parameters:
    - csv_file_path: str, the path to the CSV file.
    
    Returns:
    - array_of_arrays: List[List[str]], the contents of the CSV file.
    """
    array_of_arrays = []
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            array_of_arrays.append(row)
    return array_of_arrays

def parse_description(description):
    """
    Parses the model description to extract model details.
    
    Parameters:
    - description: str, the description of the model including architecture and settings.
    
    Returns:
    - Tuple containing extracted model details.
    """
    parts = description.split("--")
    model_architecture = parts[0]
    details = parts[1].split("_")
    input_size = details[0]
    precision = "FP32" if details[1] == "float" else "INT8"
    device_type = details[2].upper()  # Ensuring consistency in casing
    runtime = details[3].upper()
    return model_architecture, input_size, precision, device_type, runtime

def process_csv_file(csv_file_path):
    """
    Processes the CSV file to format and print its contents as a markdown table.
    
    Parameters:
    - csv_file_path: str, the path to the CSV file.
    """
    data = read_csv_to_array_of_arrays(csv_file_path)
    header = ["Description", "Unused", "mAP 50-95", "mAP 50", "mAP 75", "mAP 50-95 S", "mAP 50-95 M", "mAP 50-95 L", 
                "Recall mD1", "Recall mD10", "Recall mD100", "Recall S", "Recall M", "Recall L", "Image Cnt", "Val Path", "Label Json", "Eval Yaml"]
    df = pd.DataFrame(data, columns=header)

    # Applying the parsing function to the 'Description' column
    df[['Model Architecture', 'Input Size', 'Precision', 'Runtime', 'Device Type']] = df.apply(
        lambda row: pd.Series(parse_description(row["Description"])), axis=1)

    # Converting and rounding numerical columns
    df["mAP 50-95"] = df["mAP 50-95"].astype(float).round(4)
    df["mAP 50"] = df["mAP 50"].astype(float).round(4)
    df["FPS"] = "--"  # Placeholder for FPS column

    # Selecting and reordering the DataFrame for the desired output
    df_final = df[["Model Architecture", "Input Size", "Precision", "Device Type", "Runtime", "mAP 50-95", "mAP 50", "FPS"]]
    markdown_table = df_final.to_markdown(index=False)
    print(markdown_table)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV file to markdown format.")
    parser.add_argument("csv_file_path", type=str, help="Path to the CSV file to be processed.")
    
    args = parser.parse_args()
    process_csv_file(args.csv_file_path)
