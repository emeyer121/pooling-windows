import pytest
import torch


@pytest.fixture(scope="package")
def torch_img():
    return torch.rand((1, 1, 256, 256), dtype=torch.float32)
