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
    @pytest.mark.parametrize("scale_num", [0, 1])
    def test_plotting(self, pool_win, scale_num):
        pool_win.plot_window_areas()
        pool_win.plot_window_widths()
        pool_win.plot_window_areas("pixels", scale_num)

    @pytest.mark.parametrize("windows_scale", [0, 1])
    def test_plotting_windows(self, pool_win, windows_scale):
        pool_win.plot_windows()
        pool_win.plot_windows(
            contour_levels=0, colors="b", subset=False, windows_scale=windows_scale
        )

    @pytest.mark.parametrize("windows_scale", [0, 1])
    def test_plotting_window_values(self, pool_win, torch_img, windows_scale):
        pool_win.plot_window_values()
        fig, axes = plt.subplots(1, 1, figsize=(4, 4))
        plt.imshow(torch_img.squeeze(), cmap="Greys_r", interpolation="none")
        pool_win.plot_window_values(im=torch_img, ax=axes, subset=False)
        pool_win.plot_window_values(windows_scale=windows_scale)

    def test_plot_angle_ecc(self):
        angle_w, ecc_w = pooling.pooling.create_pooling_windows(0.87, (256, 256))
        plt.imshow(angle_w[0], cmap="Grays_r", interpolation="none")
        plt.imshow(ecc_w[0], cmap="Grays_r", interpolation="none")

    def test_po_tensor_plot(self):
        angle_w, ecc_w = pooling.pooling.create_pooling_windows(0.87, (256, 256))
        po.plot.imshow(ecc_w.unsqueeze(0))
        po.plot.imshow(angle_w.unsqueeze(0))
