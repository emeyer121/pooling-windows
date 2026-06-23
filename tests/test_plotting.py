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
    @pytest.fixture(scope="class")
    def get_ax(self):
        fig, axes = plt.subplots(1, 1, figsize=(4, 4))
        return axes

    @pytest.mark.parametrize("scale_num", [0, 1])
    def test_plotting_window_areas(self, pool_win, scale_num, get_ax):
        pool_win.plot_window_areas()
        pool_win.plot_window_areas("pixels", scale_num, ax=get_ax)

    @pytest.mark.parametrize("scale_num", [0, 1])
    def test_plotting_window_widths(self, pool_win, scale_num, get_ax):
        pool_win.plot_window_widths()
        pool_win.plot_window_widths("pixels", scale_num, jitter=None, ax=get_ax)

    @pytest.mark.parametrize("win_scale", [0, 1])
    def test_plotting_windows(self, pool_win, win_scale):
        pool_win.plot_windows()
        pool_win.plot_windows(
            contour_levels=0, colors="b", subset=False, windows_scale=win_scale
        )

    @pytest.mark.parametrize("win_scale", [0, 1])
    def test_plotting_window_values(self, pool_win, torch_img, win_scale, get_ax):
        pool_win.plot_window_values()
        pool_win.plot_window_values(im=torch_img, ax=get_ax, subset=False)
        pool_win.plot_window_values(ax=get_ax, subset=False, windows_scale=win_scale)

    def test_po_tensor_plot(self):
        angle_w, ecc_w = pooling.pooling.create_pooling_windows(0.87, (256, 256))
        po.plot.imshow(ecc_w.unsqueeze(0))
        po.plot.imshow(angle_w.unsqueeze(0))

    @pytest.mark.parametrize("angle_n", [0, 4])
    @pytest.mark.parametrize("scale", [0, 1])
    def test_plot_window_checks(self, pool_win, angle_n, scale):
        pool_win.plot_window_checks()
        pool_win.plot_window_checks(angle_n, scale)
        pool_win.plot_window_checks([0, 4], scale)
