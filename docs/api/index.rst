.. _api:

API
===

Pooling Windows object
----------------------

The main object you will interact with is `PoolingWindows` which will create
and plot the generated windows. Additionally, `create_pooling_windows` will return
the angle and eccentricity windows without the added functionality of `PoolingWindows`.

.. currentmodule:: fenestration
.. autosummary::
    :toctree: generated
    :signatures: none

    PoolingWindows
    create_pooling_windows

Pooling
-------

.. currentmodule:: fenestration.pooling
.. autosummary::
    :toctree: generated
    :signatures: none

    gaussian
    raised_cosine
    normalize_windows

Calculate
---------

.. currentmodule:: fenestration.calculate
.. autosummary::
    :toctree: generated
    :signatures: none

    scaling
    deg_to_pix

Sampling
--------

.. currentmodule:: fenestration.sampling
.. autosummary::
    :toctree: generated
    :signatures: none

    check_sampling
    plot_coeffs
    interpolation_plot
    create_movie
