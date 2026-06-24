#!/usr/bin/env python3
import matplotlib as mpl
import numpy as np
import pytest
import torch

import pooling

# use the html backend, so we don't need to have ffmpeg
mpl.rcParams["animation.writer"] = "html"
# necessary to avoid issues with animate:
# https://github.com/matplotlib/matplotlib/issues/10287/
mpl.use("agg")


class TestSampling:
    @pytest.fixture(scope="class")
    def x_eval(self):
        return torch.linspace(-5, 5, 101)

    @pytest.mark.parametrize("val_samp", [0.5, 1, 2])
    def test_check_val_sampling(self, val_samp, x_eval):
        sampled, full, interps, coeffs, residuals = pooling.sampling.check_sampling(
            val_sampling=val_samp,
            pix_sampling=None,
            func=pooling.pooling.gaussian,
            x=x_eval,
        )
        pooling.sampling.plot_coeffs(coeffs[:20], 10)
        pooling.sampling.interpolation_plot(interps, residuals, None, val_samp, x_eval)
        pooling.sampling.create_movie(interps, residuals, x_eval, full=full)

    @pytest.mark.parametrize("pix_samp", [2, 10, 25])
    def test_check_pix_sampling(self, pix_samp, x_eval):
        sampled, full, interps, coeffs, residuals = pooling.sampling.check_sampling(
            val_sampling=None,
            pix_sampling=pix_samp,
            func=pooling.pooling.mother_window,
            x=x_eval,
        )
        pooling.sampling.plot_coeffs(coeffs[:20], 10)
        pooling.sampling.interpolation_plot(interps, residuals, pix_samp, None, x_eval)
        pooling.sampling.create_movie(interps, residuals, x_eval, full=full)

    def test_check_sampling_error(self, x_eval):
        with pytest.raises(Exception):
            pooling.sampling.check_sampling(
                val_sampling=2,
                pix_sampling=2,
                func=pooling.pooling.mother_window,
                x=x_eval,
            )

    def test_check_small_residuals(self, x_eval):
        _, _, _, _, residuals = pooling.sampling.check_sampling(0.5, x=x_eval)
        assert np.allclose(residuals, 0)

    def test_check_interp_fun(self, x_eval):
        _, _, interps, _, _ = pooling.sampling.check_sampling(
            0.5, func=pooling.pooling.gaussian, x=x_eval
        )
        orig_fun = pooling.pooling.gaussian(x_eval)
        cent = np.argmin(abs(x_eval - 0))
        assert np.allclose(interps[:, cent], np.array(orig_fun))

    def test_check_bad_interp(self, x_eval):
        _, _, interps, _, _ = pooling.sampling.check_sampling(
            2, func=pooling.pooling.gaussian, x=x_eval
        )
        orig_fun = pooling.pooling.gaussian(x_eval)
        cent = np.argmin(abs(x_eval - 0))
        assert not np.allclose(interps[:, cent], np.array(orig_fun))

    def test_check_interp_raisedcos(self, x_eval):
        _, _, interps, _, _ = pooling.sampling.check_sampling(
            0.5, func=pooling.pooling.gaussian, x=x_eval
        )
        orig_fun = pooling.pooling.mother_window(x_eval)
        cent = np.argmin(abs(x_eval - 0))
        assert not np.allclose(interps[:, cent], np.array(orig_fun))
