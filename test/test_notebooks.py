import nbformat
from pathlib import Path
import nbclient
from PIL import Image, ImageChops
from io import BytesIO
import base64
from os import environ
import pytest

examples_dir = Path(__file__).parent.parent
reference_dir = examples_dir / "test" / "reference"
images_dir = examples_dir / "images"

output_dir = examples_dir / "test" / "output"
output_dir.mkdir(exist_ok=True)


def open_and_execute(
    notebook_file: Path, code_cells_with_exception=[]
) -> nbformat.NotebookNode:
    """Helper function for executing a notebook using nbclient"""
    with open(examples_dir / notebook_file, "r") as file:
        nb: nbformat.NotebookNode = nbformat.read(file, as_version=4)

    code_cells = [cell for cell in nb.cells if cell["cell_type"] == "code"]

    for index in code_cells_with_exception:
        if "tags" not in (metadata := code_cells[index - 1]["metadata"]):
            metadata["tags"] = []
        metadata["tags"].append("raises-exception")

    client = nbclient.NotebookClient(nb, timeout=300, kernel_name="python3")
    client.allow_errors = False

    # inject a monkeypatch for mytools._reload_env so the environment variables set in the
    # test don't get overwritten by env.ini when we are running inside the kernel
    code_cells[0].source = (
        "import mytools; mytools._reload_env = lambda *a, **k: None\n"
        + code_cells[0].source
    )

    nb = client.execute()
    # save notebook with output, useful for debugging
    # nbformat.write(nb, output_dir / f"{notebook_file.stem}.ipynb")
    return nb


_image_notebooks = [
    ("mystreamsDemo.ipynb", "Masked.mp4", [1, 2, 3, 4, 5, 6], []),
    # an expected exception arises in cell 5 when using a file instead of a camera
    ("FaceHandDetectionParallelCameraStream.ipynb", "Masked.mp4", [5], [5]),
    ("FaceMaskDetectionPipelinedCameraStream.ipynb", "Masked.mp4", [5], []),
    # dictionary with image count for notebooks with cells with > 1 image
    ("FaceMaskDetectionPipelinedImage.ipynb", None, {3: 3}, []),
    ("ObjectDetectionCameraStream.ipynb", "Masked.mp4", [4], []),
    ("ObjectDetectionImage.ipynb", None, [6], []),
    ("PersonPoseDetectionPipelinedCameraStream.ipynb", "Masked.mp4", [6], []),
    ("PersonPoseDetectionPipelinedImage.ipynb", None, {3: 3, 4: 1}, []),
    ("TiledObjectDetectionVideoFile.ipynb", None, [8], []),
]


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
            # cell_image.save(output_dir / f"{filename.stem}_{id}.{image_count}.png")
            ref_image = Image.open(
                reference_dir / f"{filename.stem}_{id}.{image_count}.png"
            )
            assert not ImageChops.difference(
                cell_image, ref_image
            ).getbbox(), "Image does not match reference"


_imageless_notebooks = [
    ("ObjectDetectionMultiplexingMultipleStreams.ipynb", None),
    ("ObjectDetectionVideoFile.ipynb", None),
    ("ObjectDetectionVideoFile2Images.ipynb", None),
]


@pytest.mark.parametrize("notebook_file, input_file", _imageless_notebooks)
def test_notebook(notebook_file, input_file):
    """Test notebook by executing it"""
    filename = examples_dir / notebook_file
    if input_file:
        environ["CAMERA_ID"] = str(images_dir / input_file)

    open_and_execute(filename)


# currently skipped
_skipped_notebooks = [
    "ObjectDetectionDataset.ipynb",
    "ObjectDetectionDatasetMultithreaded.ipynb",
    "MultiObjectTrackingVideoFile.ipynb",
    "SoundClassificationAndObjectDetectionAsynchronous.ipynb",
    "SoundClassificationAudioStream.ipynb",
]
