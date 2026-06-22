"""Helpful utility functions for plotting or displaying information"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def setup_fig(image, cmap=None, **kwargs):
    """Setup figure for displaying pooling windows.

    Arguments
    ---------
    image : `np.array`
        the image to plot. Image can be either grayscale, in which case
        they must be 2d arrays of shape `(h,w)`, or RGB(A), in which case they
        must be 3d arrays of shape `(h,w,c)` where `c` is 3 (for RGB) or 4 (to
        also plot the alpha channel).
    cmap : matplotlib colormap, optional
        colormap to use when showing these images
    kwargs :
        Passed to `ax.imshow`

    Returns
    -------
    fig : `Figure`
        matplotlib figure containing the plotted images

    """

    # determine figure shape
    img_shape = np.shape(image)

    # this is an arbitrary value
    ppi = 96

    # get the figure and axes created
    fig = plt.figure(figsize=(img_shape[1] / ppi, img_shape[0] / ppi), dpi=ppi)

    # define axis parameters
    fig.add_axes([0, 0, 1, 1], frameon=False, xticks=[], yticks=[])
    axes = fig.axes

    # set cmap for figure
    if cmap is None:
        cmap = cm.gray

    # set axes parameters
    axes[0].imshow(
        image,
        cmap=cmap,
        interpolation="none",
        **kwargs,
    )

    return fig
