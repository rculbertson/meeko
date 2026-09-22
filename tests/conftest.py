import pytest
from dotenv import load_dotenv


@pytest.fixture(scope="session", autouse=True)
def _load_env():
    load_dotenv()
