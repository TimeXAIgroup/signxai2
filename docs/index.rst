====================
SIGN-XAI2 Documentation
====================

SIGN (Sign-based Improvement of Gradient-based explaNations) is a novel XAI method intended to reduce bias in explanations that are intrinsically induced by several state-of-the-art XAI methods. The `SIGN-XAI2 package <https://pypi.org/project/signxai/>`_ enables simple application of this method in your projects and is based on `Zennit <https://pypi.org/project/zennit/>`_ and `LRP for Transformers <https://pypi.org/project/lxt/>`_. If your are using TensorFlow instead of PyTorch, have a look at our `TF-version of SIGN-XAI <https://pypi.org/project/signxai/>`_.

SIGN-based explanations are particularly well suited for generating bias-reduced heatmaps for both **image** and **time series data**, enhancing interpretability by more reliably uncovering relevant features.

Install
-------

.. code-block:: console

   $ pip install signxai2

Contents
--------

.. toctree::
    :maxdepth: 2

    getting-started
    how-to
    tutorial/index
    rules
    composites
    dft-lrp

Citing
------

If you use this package or parts of it in your own work, please consider citing our paper:

.. code-block:: bibtex

    @article{Gumpfer2023SIGN,
        title = {SIGNed explanations: Unveiling relevant features by reducing bias},
        author = {Nils Gumpfer and Joshua Prim and Till Keller and Bernhard Seeger and Michael Guckert and Jennifer Hannig},
        journal = {Information Fusion},
        pages = {101883},
        year = {2023},
        issn = {1566-2535},
        doi = {https://doi.org/10.1016/j.inffus.2023.101883}
    }
