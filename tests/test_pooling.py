#!/usr/bin/env python3
import os.path as op

import numpy as np
import pytest
import torch

import pooling


class TestPooling:
    def test_creation(self):
        pooling.pooling.create_pooling_windows(0.87, (256, 256))

    def test_creation_args(self):
        pooling.pooling.create_pooling_windows(
            0.87, (100, 100), 0.2, 30, 1.2, transition_region_width=0.7
        )
        pooling.pooling.create_pooling_windows(
            0.87, (100, 100), 0.2, 30, 1.2, transition_region_width=0.5
        )
        pooling.pooling.create_pooling_windows(
            0.87, (100, 100), 0.2, 30, 1.2, "gaussian", std_dev=1
        )

    def test_ecc_windows(self):
        pooling.pooling.log_eccentricity_windows((256, 256), n_windows=4)
        pooling.pooling.log_eccentricity_windows((256, 256), n_windows=4.5)
        pooling.pooling.log_eccentricity_windows((256, 256), window_spacing=0.5)
        pooling.pooling.log_eccentricity_windows((256, 256), window_spacing=1)

    def test_angle_windows(self):
        pooling.pooling.polar_angle_windows(4, (256, 256))
        pooling.pooling.polar_angle_windows(4, (1000, 1000))
        with pytest.raises(Exception):
            pooling.pooling.polar_angle_windows(1.5, (256, 256))
        with pytest.raises(Exception):
            pooling.pooling.polar_angle_windows(1, (256, 256))

    def test_calculations(self):
        # these really shouldn't change, but just in case...
        assert pooling.pooling.calc_angular_window_spacing(2) == np.pi
        assert pooling.pooling.calc_angular_n_windows(2) == np.pi
        with pytest.raises(Exception):
            pooling.pooling.calc_eccentricity_window_spacing()
        assert np.allclose(
            pooling.pooling.calc_eccentricity_window_spacing(n_windows=4),
            0.8502993454155389,
        )
        assert np.allclose(
            pooling.pooling.calc_eccentricity_window_spacing(scaling=0.87),
            0.8446653390527211,
        )
        assert np.allclose(
            pooling.pooling.calc_eccentricity_window_spacing(5, 10, scaling=0.87),
            0.8446653390527211,
        )
        assert np.allclose(
            pooling.pooling.calc_eccentricity_window_spacing(5, 10, n_windows=4),
            0.1732867951399864,
        )
        assert np.allclose(
            pooling.pooling.calc_eccentricity_n_windows(0.8502993454155389), 4
        )
        assert np.allclose(
            pooling.pooling.calc_eccentricity_n_windows(0.1732867951399864, 5, 10), 4
        )
        assert np.allclose(pooling.pooling.calc_scaling(4), 0.8761474337786708)
        assert np.allclose(pooling.pooling.calc_scaling(4, 5, 10), 0.17350368946058647)
        assert np.isinf(pooling.pooling.calc_scaling(4, 0))

    @pytest.mark.parametrize("num_scales", [1, 3])
    @pytest.mark.parametrize("transition_region_width", [0.5, 1])
    def test_PoolingWindows_cosine(
        self, torch_img, num_scales, transition_region_width
    ):
        pw = pooling.PoolingWindows(
            0.5,
            torch_img.shape[2:],
            num_scales=num_scales,
            transition_region_width=transition_region_width,
            window_type="cosine",
        )
        pw(torch_img)

    @pytest.mark.parametrize("num_scales", [1, 3])
    def test_PoolingWindows(self, torch_img, num_scales):
        pw = pooling.PoolingWindows(
            0.5,
            torch_img.shape[2:],
            num_scales=num_scales,
            window_type="gaussian",
            std_dev=1,
        )
        pw(torch_img)
        # we only support std_dev=1
        with pytest.raises(Exception):
            pooling.PoolingWindows(
                0.5,
                torch_img.shape[2:],
                num_scales=num_scales,
                window_type="gaussian",
                std_dev=2,
            )
        with pytest.raises(Exception):
            pooling.PoolingWindows(
                0.5,
                torch_img.shape[2:],
                num_scales=num_scales,
                window_type="gaussian",
                std_dev=0.5,
            )

    def test_PoolingWindows_project(self, torch_img):
        pw = pooling.PoolingWindows(0.5, torch_img.shape[2:])
        pooled = pw(torch_img)
        pw.project(pooled)
        pw = pooling.PoolingWindows(0.5, torch_img.shape[2:], num_scales=3)
        pooled = pw(torch_img)
        pw.project(pooled)

    def test_PoolingWindows_nonsquare(self, torch_img):
        # test PoolingWindows with weirdly-shaped iamges
        for sh in [(256, 128), (256, 127), (256, 125), (125, 125), (127, 125)]:
            tmp = torch_img[..., : sh[0], : sh[1]]
            pw = pooling.PoolingWindows(0.9, tmp.shape[-2:])
            pw(tmp)

    def test_PoolingWindows_caching(self, torch_img, tmp_path):
        # first time we save, second we load
        pooling.PoolingWindows(
            0.8, torch_img.shape[-2:], num_scales=2, cache_dir=tmp_path
        )
        pooling.PoolingWindows(
            0.8, torch_img.shape[-2:], num_scales=2, cache_dir=tmp_path
        )

    def test_PoolingWindows_cache_dne(self, torch_img, tmp_path):
        tmp_path = op.join(tmp_path, "new_dir")
        with pytest.raises(FileNotFoundError):
            pooling.PoolingWindows(
                0.8, torch_img.shape[-2:], num_scales=2, cache_dir=tmp_path
            )

    def test_PoolingWindows_sep(self, torch_img, pool_win):
        # test the window and pool function separate of the forward function
        pool_win.pool(pool_win.window(torch_img))

    @pytest.mark.parametrize("num_scales", [1, 3])
    @pytest.mark.parametrize("input_fmt", ["dict", "tensor"])
    def test_reweighting(self, num_scales, input_fmt):
        pw = pooling.PoolingWindows(0.5, (256, 256), num_scales=num_scales)
        im = {
            (i,): torch.rand((1, 1, 256 // 2**i, 256 // 2**i), dtype=torch.float32)
            for i in range(num_scales)
        }
        if input_fmt == "dict":
            pw(im, weights=torch.ones(num_scales, 1, 1, 1, 1))
        elif input_fmt == "tensor":
            for i in range(num_scales):
                pw(im[(i,)], idx=i, weights=torch.ones(num_scales, 1, 1, 1, 1))
