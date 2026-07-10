---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: pooling
  language: python
  name: python3
---

```{code-cell} ipython3
:tags: [hide-input]

import warnings

warnings.filterwarnings(
    "ignore",
    message="torch.meshgrid: in an upcoming release,",
    category=UserWarning,
)

warnings.filterwarnings(
    "ignore",
    message="torch.range is deprecated and will be removed",
    category=UserWarning,
)
```

```{admonition} Run this notebook yourself!
:class: important

Download the executed notebook: **{nb-download}`Generate_Windows.ipynb`**!

```

(generate-windows-nb)=
# Generate Windows

This notebook provides tutorials on the most common ways of generating and interacting with pooling windows. These can be used generally for building foveated windows and interacting with the resulting weight from averages across windows. We will also discuss how to use these windows for metamer generation in the next tutorial (link).

```{code-cell} ipython3
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch

import pooling

mpl.rcParams['xtick.bottom'] = False
mpl.rcParams['xtick.labelbottom'] = False
mpl.rcParams['ytick.left'] = False
mpl.rcParams['ytick.labelleft'] = False

%load_ext autoreload
%autoreload 2
%matplotlib inline
```

## Creating PoolingWindows Objects

Let's begin by creating a `PoolingWindows` object for image size `(256,256)` and visualize the window contours that are created. We must also input a `scaling` value that determines the size of the window (see [](choosing-scaling-values)).

```{code-cell} ipython3
pw = pooling.PoolingWindows(0.5, (256,256))
pw.plot_windows(subset=False)
```

We can also change a number of other parameters that define the windows: `min_eccentricity` and `max_eccentricity` that define the extent of the windows within the image in degrees of visual angle, `num_scales` which controls the number of window scales generated, `cache_dir` for specifying a path to cache the model in for loading/saving, and `window_type` which can be defined as `gaussian` or `cosine` (**see tutorial for comparison**).

```{code-cell} ipython3
pw = pooling.PoolingWindows(0.5, (256,256), min_eccentricity=1, max_eccentricity=10, window_type='cosine')
pw.plot_windows()
```

If you want to take advantage of just the eccentricity rings or angular wedges separately, you can also call `pooling.create_pooling_windows`. We will again use `scaling=0.5` and and image size of `(256,256)`. We will also take advantage of [plenoptic's](https://plenoptic.org/) plotting function `po.imshow`.

```{code-cell} ipython3
import plenoptic as po

angle_w, ecc_w = pooling.pooling.create_pooling_windows(2, (256, 256))
fig = po.plot.imshow(ecc_w.unsqueeze(0))
fig = po.plot.imshow(angle_w.unsqueeze(0))
plt.show()
```

It is also simple to reconstruct the windows from this angle and eccentricity data.

```{code-cell} ipython3
windows = torch.einsum('ahw,ehw->eahw', [angle_w, ecc_w]).flatten(0, 1)
```

## Displaying Window Values

Now let's generate a figure with a noisy gradient across the image. We can then use `plot_window_values` to display the average values within each window.

```{code-cell} ipython3
img = torch.rand((1, 1, 256, 256), dtype=torch.float32) * torch.range(1/256,1,1/256)
plt.imshow(torch.squeeze(img), cmap='gray')
```

```{code-cell} ipython3
pw = pooling.PoolingWindows(1, (256,256))
pw.plot_window_values(img, subset=False)
```

If you would like a summary of the size and values associated with the pooling windows, you can call `summarize_window_sizes`.

```{code-cell} ipython3
summary = pw.summarize_window_sizes(print_summary=True)
```

(choosing-scaling-values)=
## Choosing Scaling Values

However, the `scaling` values used in previous examples were arbitrary. Let's say you are displaying images for an experiment and want to build pooling windows ranging from 1-10 degrees of eccentricity, tiling the space with 5 angular windows. You can then find the precise scaling value to generate the corresponding `PoolingWindows` object.

```{code-cell} ipython3
scaling = pooling.calculate.scaling(n_windows=5, min_ecc=1, max_ecc=10, std_dev=1)
pw = pooling.PoolingWindows(scaling, (256,256), min_eccentricity=1, max_eccentricity=10)
ax = pw.plot_windows()
ax.set_title(f"Scaling = {scaling:.4f}");
```
