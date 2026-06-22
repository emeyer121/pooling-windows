#!/usr/bin/env python3
import matplotlib as mpl
import torch

import pooling

# use the html backend, so we don't need to have ffmpeg
mpl.rcParams["animation.writer"] = "html"
# necessary to avoid issues with animate:
# https://github.com/matplotlib/matplotlib/issues/10287/
mpl.use("agg")


class TestSampling:
    def test_check_val_sampling(self):
        x = torch.linspace(-5, 5, 101)
        for samp in [0.5, 1, 2]:
            sampled, full, interps, coeffs, residuals = pooling.sampling.check_sampling(
                val_sampling=samp, pix_sampling=None, func=pooling.pooling.gaussian, x=x
            )
            pooling.sampling.plot_coeffs(coeffs[:20], 10)
            pooling.sampling.interpolation_plot(interps, residuals, None, samp, x)
            pooling.sampling.create_movie(interps, residuals, x, full=full)

    def test_check_pix_sampling(self):
        x = torch.linspace(-5, 5, 101)
        for samp in [2, 10, 25]:
            sampled, full, interps, coeffs, residuals = pooling.sampling.check_sampling(
                val_sampling=None,
                pix_sampling=samp,
                func=pooling.pooling.mother_window,
                x=x,
            )
            pooling.sampling.plot_coeffs(coeffs[:20], 10)
            pooling.sampling.interpolation_plot(interps, residuals, samp, None, x)
            pooling.sampling.create_movie(interps, residuals, x, full=full)
