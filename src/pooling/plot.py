"""Helpful utility functions for plotting or displaying information"""

import matplotlib.pyplot as plt
import numpy as np

__all__ = []


def __dir__() -> list[str]:
    return __all__


def _setup_fig(image_size):
    """Setup figure for displaying pooling windows.

    Arguments
    ---------
    image_size : `np.array`
        the size of the image to plot. Image can be either grayscale or RGB(A),
        but here only the first two dimensions (h,w) matter for setting up the figure.

    Returns
    -------
    fig : `Figure`
        matplotlib figure containing the plotted images

    """

    # this is an arbitrary value
    ppi = 96

    # get the figure and axes created
    fig = plt.figure(figsize=(image_size[1] / ppi, image_size[0] / ppi), dpi=ppi)

    # define axis parameters
    fig.add_axes([0, 0, 1, 1], frameon=False, xticks=[], yticks=[])
    axes = fig.axes

    # set axes parameters
    axes[0].imshow(np.ones(image_size), cmap="gray_r", interpolation="none")

    return fig
