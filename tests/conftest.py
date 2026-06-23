import pytest
import torch

import pooling


@pytest.fixture(scope="package")
def torch_img():
    return torch.rand((1, 1, 256, 256), dtype=torch.float32)


@pytest.fixture(scope="package")
def pool_win(torch_img):
    return pooling.PoolingWindows(0.5, torch_img.shape[2:], num_scales=2)
