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
import numpy as np
import torch

import pooling

%load_ext autoreload
%autoreload 2
%matplotlib inline
```

## Creating PoolingWindows Objects

Let's begin by creating a `PoolingWindows` object for image size `(256,256)` and visualize the windows that are created. We must also input a `scaling` value that determines the size of the window (choosing this value will be discussed more later). We have two built-in window types that can be used to generate windows: `gaussian` and `raised cosine`. **remove std_dev once this has been changed in code reorg**

```{code-cell} ipython3
pw_gauss = pooling.PoolingWindows(.5, (256,256), window_type='gaussian',std_dev=1)
pw_cosine = pooling.PoolingWindows(.5, (256,256), window_type='cosine')
pw_gauss.plot_windows()
pw_cosine.plot_windows()
```

If you want to take advantage of just the eccentricity rings or angular wedges separately, you can also call `pooling.create_pooling_windows`. We will again use `scaling=0.5` and and image size of `(256,256)`. We will also take advantage of [plenoptic's](https://plenoptic.org/) plotting function `po.imshow`.

```{code-cell} ipython3
import plenoptic as po

angle_w, ecc_w = pooling.pooling.create_pooling_windows(0.5, (256, 256))
fig = po.plot.imshow(ecc_w.unsqueeze(0))
fig = po.plot.imshow(angle_w.unsqueeze(0))
plt.show()
```

It is also simple to reconstruct the window contours from this angle and eccentricity data.

```{code-cell} ipython3
windows = torch.einsum('ahw,ehw->eahw', [angle_w, ecc_w]).flatten(0, 1)
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
for w in windows:
    ax.contour(w, [.5], colors='r')
plt.show()
```

Now let's generate a figure with a noisy gradient across the image. We can then use `plot_window_values` to display the average values within each window.

```{code-cell} ipython3
img = torch.rand((1, 1, 256, 256), dtype=torch.float32) * torch.range(1/256,1,1/256)
plt.imshow(torch.squeeze(img), cmap='gray')
```

```{code-cell} ipython3
pw_gauss.plot_window_values(img, subset=False)
pw_cosine.plot_window_values(img, subset=False)
```
