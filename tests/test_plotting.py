#!/usr/bin/env python3
import matplotlib as mpl
import matplotlib.pyplot as plt
import plenoptic as po
import pytest

import pooling

# necessary to avoid issues with animate:
# https://github.com/matplotlib/matplotlib/issues/10287/
mpl.use("agg")


# following https://github.com/scverse/scanpy/issues/1662
@pytest.fixture(autouse=True)
def close_figures_on_teardown():
    yield
    plt.close("all")


class TestPlotting:
    def test_plotting_window_areas(self, pool_win):
        pool_win.plot_window_areas()

    @pytest.mark.parametrize("scale_num", [0, 1])
    def test_plotting_window_areas_args(self, pool_win, scale_num):
        fig, axes = plt.subplots(1, 1, figsize=(4, 4))
        pool_win.plot_window_areas("pixels", scale_num, ax=axes)

    def test_plotting_window_widths(self, pool_win):
        pool_win.plot_window_widths()

    @pytest.mark.parametrize("scale_num", [0, 1])
    def test_plotting_window_widths_args(self, pool_win, scale_num):
        fig, axes = plt.subplots(1, 1, figsize=(4, 4))
        pool_win.plot_window_widths("pixels", scale_num, jitter=None, ax=axes)

    def test_plotting_windows(self, pool_win):
        pool_win.plot_windows()

    @pytest.mark.parametrize("win_scale", [0, 1])
    def test_plotting_windows_args(self, pool_win, win_scale):
        pool_win.plot_windows(
            contour_levels=0, colors="b", subset=False, windows_scale=win_scale
        )

    def test_plotting_window_values(self, pool_win):
        pool_win.plot_window_values()

    @pytest.mark.parametrize("win_scale", [0, 1])
    def test_plotting_window_values_args(self, pool_win, rand_img, win_scale):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        pool_win.plot_window_values(im=rand_img, ax=axes[0], subset=False)
        pool_win.plot_window_values(
            im=None, ax=axes[1], subset=False, windows_scale=win_scale
        )

    def test_po_tensor_plot(self):
        angle_w, ecc_w = pooling.pooling.create_pooling_windows(0.87, (256, 256))
        po.plot.imshow(ecc_w.unsqueeze(0))
        po.plot.imshow(angle_w.unsqueeze(0))

    def test_plot_window_checks(self, pool_win):
        pool_win.plot_window_checks()

    @pytest.mark.parametrize("angle_n", [0, 4])
    @pytest.mark.parametrize("scale", [0, 1])
    def test_plot_window_checks_args(self, pool_win, angle_n, scale):
        pool_win.plot_window_checks(angle_n, scale)

    @pytest.mark.parametrize("scale", [0, 1])
    def test_plot_window_checks_arglist(self, pool_win, scale):
        pool_win.plot_window_checks([0, 4], scale)
