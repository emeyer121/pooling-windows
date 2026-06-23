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
    def test_plotting(self, torch_img):
        pw = pooling.PoolingWindows(0.8, torch_img.shape[-2:], num_scales=2)
        pw.plot_window_areas()
        pw.plot_window_widths()
        for i in range(2):
            pw.plot_window_areas("pixels", i)
            pw.plot_window_widths("pixels", i)

    def test_plotting_windows(self, torch_img):
        pw = pooling.PoolingWindows(0.8, torch_img.shape[-2:], num_scales=2)
        pw.plot_windows()
        for i in range(2):
            pw.plot_windows(contour_levels=0, colors="b", subset=False, windows_scale=i)

    def test_plotting_window_values(self, torch_img):
        pw = pooling.PoolingWindows(0.8, torch_img.shape[-2:], num_scales=2)
        pw.plot_window_values()
        fig, axes = plt.subplots(1, 1, figsize=(4, 4))
        plt.imshow(torch_img.squeeze(), cmap="Greys_r", interpolation="none")
        pw.plot_window_values(im=torch_img, ax=axes, subset=False)
        for i in range(2):
            pw.plot_window_values(windows_scale=i)

    def test_plot_angle_ecc(self):
        angle_w, ecc_w = pooling.pooling.create_pooling_windows(0.87, (256, 256))
        plt.imshow(angle_w[0], cmap="Grays_r", interpolation="none")
        plt.imshow(ecc_w[0], cmap="Grays_r", interpolation="none")

    def test_po_tensor_plot(self):
        angle_w, ecc_w = pooling.pooling.create_pooling_windows(0.87, (256, 256))
        po.plot.imshow(ecc_w.unsqueeze(0))
        po.plot.imshow(angle_w.unsqueeze(0))
