#
# test_notebooks.py: Tests for PySDK Examples notebooks
#
# Copyright DeGirum Corporation 2023
# All rights reserved
#
# Contains tests for notebooks
#

import nbformat
from pathlib import Path
import nbclient
from PIL import Image
from io import BytesIO
import base64
from os import environ, chdir
import pytest
from SSIM_PIL import compare_ssim

root_dir = Path(__file__).parent.parent
examples_dir = root_dir / "examples"
reference_dir = root_dir / "tests" / "reference"
images_dir = root_dir / "images"
output_dir = root_dir / "tests" / "output"
output_dir.mkdir(exist_ok=True)

# In order to add a new notebook to the test, add it as a tuple to the appropriate list of test
# parametrizations

# _image_notebooks is a list of notebooks with image outputs to test
# Tuple of (notebook_filename, input_file, cells_with_image, cells_with_exception)
# notebook_filename is filename relative to the PySDKExamples root directory
# input_file is name of file in the PySDKExamples/images directory to use as input for the test
#    input is patched in for notebooks that use degirum_tools.open_video_stream
#    None means use the notebook's input (currently used for Image notebooks)
# cells_with_image is a list or a dictionary of code cells with image outputs
#    if it is a list: each entry is a code cell index with 1 expected image output
#    if it is a dictionary: each key:value pair is a mapping of cell index and number of images expected in that code cell
#    code cells with no image output are omitted from the lists
#    NOTE: code cell indexes and image indexes start with 1
# cells_with_exception is a list of code cells with an expected exception during execution
#    NOTE: code cell indexes start with 1
# used to parametrize the 'test_notebook_image_output' test
# cell output is verified by checking the output image against a reference image in PySDKExamples/test/reference
# reference image names are of the format {notebook_name}_{cell_index}.{image_within_cell_index}.png
_image_notebooks = [
    ("basic/object_detection_camera_stream.ipynb", "Masked.mp4", [3], []),
    ("basic/object_detection_image.ipynb", None, [7], []),
    ("dgstreams/dgstreams_demo.ipynb", "Masked.mp4", [2, 3, 4, 5, 6,7], []),
    ("dgstreams/multi_camera_multi_model_detection.ipynb", "Masked.mp4", [4], []),
    ("dgstreams/person_pose_detection_pipelined_camera_stream.ipynb", "Masked.mp4", [7], []),
    ("multimodel/face_hand_detection_parallel_camera_stream.ipynb", "Masked.mp4", [4], []),
    ("multimodel/license_plate_recognition_pipelined_camera_stream.ipynb", "Car.mp4", [4], []),
    ("multimodel/license_plate_recognition_pipelined_image.ipynb", None, [3], []),
    ("specialized/sliced_object_detection.ipynb", "TrafficHD_short.mp4", [9], []),
    ("specialized/multi_object_tracking_video_file.ipynb", "Masked.mp4", [8], []),
]

# _imageless_notebooks is a list of notebooks without an image cell output
# they are tested for exceptionless execution without output verification
# Tuple of (notebook_filename, input_file) see _image_notebooks doc for more info
# used to parametrize the 'test_notebook' test
_imageless_notebooks = [
    ("basic/object_detection_video_file.ipynb", "Masked.mp4"),
    ("benchmarks/multi_model_performace_test.ipynb", ""),
    ("benchmarks/object_detection_multiplexing_multiple_streams.ipynb", "Masked.mp4"),
    ("benchmarks/single_model_performace_test.ipynb", ""),
]

# list of notebooks that are excluded from tests for a variety of reasons
_skipped_notebooks = [
    "basic/sound_classification_audio_stream.ipynb",
    "benchmarks/object_detection_dataset_evaluation.ipynb",
    "multimodel/sound_classification_and_object_detection_asynchronous.ipynb",
]


def open_and_execute(
    notebook_file: Path, code_cells_with_exception=[]
) -> nbformat.NotebookNode:
    """Helper function for executing a notebook using nbclient"""
    with open(notebook_file, "r") as file:
        nb: nbformat.NotebookNode = nbformat.read(file, as_version=4)

    code_cells = [cell for cell in nb.cells if cell["cell_type"] == "code"]

    for index in code_cells_with_exception:
        if "tags" not in (metadata := code_cells[index - 1]["metadata"]):
            metadata["tags"] = []
        metadata["tags"].append("raises-exception")

    chdir(notebook_file.parent)

    client = nbclient.NotebookClient(nb, timeout=600, kernel_name="python3")
    client.allow_errors = False

    # inject a monkeypatch for degirum_tools._reload_env so the environment variables set in the
    # test don't get overwritten by env.ini when we are running inside the kernel
    code_cells[0].source = (
        "import degirum_tools; degirum_tools._reload_env = lambda *a, **k: None\n"
        + code_cells[0].source
    )

    nb = client.execute()
    # save notebook with output, useful for debugging
    # nbformat.write(nb, output_dir / f"{notebook_file.stem}.ipynb")
    return nb


@pytest.mark.parametrize(
    "notebook_file, input_file, code_cells_with_image, code_cells_with_exception",
    _image_notebooks,
)
def test_notebook_image_output(
    notebook_file, input_file, code_cells_with_image, code_cells_with_exception
):
    """Test notebook by executing it and comparing image outputs with reference data"""
    filename = examples_dir / notebook_file
    if input_file:
        environ["CAMERA_ID"] = str(images_dir / input_file)

    nb = open_and_execute(filename, code_cells_with_exception)

    code_cells = [cell for cell in nb.cells if cell["cell_type"] == "code"]
    for id, cell in enumerate(code_cells, 1):
        assert "outputs" in cell
        image_data = [
            output.data["image/png"]
            for output in cell["outputs"]
            if "data" in output and "image/png" in output.data
        ]

        assert (id in code_cells_with_image) == bool(
            image_data
        ), f"code cell #{id}: has {'' if image_data else 'no'} output images, which is unexpected"

        if not image_data:
            continue

        expected_image_count = (
            code_cells_with_image[id] if isinstance(code_cells_with_image, dict) else 1
        )
        assert (
            len(image_data) == expected_image_count
        ), f"code cell #{id} expected {expected_image_count} images, got {len(image_data)}"

        for image_count, image_datum in enumerate(image_data, 1):
            cell_image = Image.open(BytesIO(base64.b64decode(image_datum)))
            cell_image.save(output_dir / f"{filename.stem}_{id}.{image_count}.png")
            ref_filename = f"{filename.stem}_{id}.{image_count}.png"
            ref_image = Image.open(reference_dir / ref_filename)
            assert (
                compare_ssim(cell_image, ref_image, GPU=False) > 0.95
            ), f"Image {ref_filename} in cell {id} of notebook {notebook_file} does not match reference"


@pytest.mark.parametrize("notebook_file, input_file", _imageless_notebooks)
def test_notebook(notebook_file, input_file):
    """Test notebook by executing it"""
    filename = examples_dir / notebook_file
    if input_file:
        environ["CAMERA_ID"] = str(images_dir / input_file)

    open_and_execute(filename)
