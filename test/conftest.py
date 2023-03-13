import pytest
from pytest_mock import MockerFixture
import mytools
from os import environ

def pytest_addoption(parser):
    """Add pysdk command line parameters"""
    parser.addoption(
        "--token", action="store", default="", help="cloud server token value to use"
    )

@pytest.fixture(scope="session")
def cloud_token(request):
    """Get cloud server token passed from the command line"""
    return request.config.getoption("--token")

@pytest.fixture(autouse=True)
def mock_env(cloud_token: str) -> None:
    environ["CLOUD_ZOO_URL"] = "degirum_com/public"
    environ["DEGIRUM_CLOUD_TOKEN"] = cloud_token
    environ["TEST_MODE"] = "1"

    