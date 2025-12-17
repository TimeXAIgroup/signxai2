================
 Getting started
================

Install
-------

SIGN-XAI2 can be installed directly from PyPI:

.. code-block:: console

   $ pip install signxai2


Basic Usage
-----------

SIGN-XAI2 is based on Zennit, which implements propagation-based attribution methods by overwriting the
gradient of PyTorch modules in PyTorch's auto-differentiation engine. This means
that Zennit will only work on models which are strictly implemented using
PyTorch modules, including activation functions. The following demonstrates a
setup to compute SIGN-based Layer-wise Relevance Propagation (LRP) relevance for a simple
model and random data.

.. code-block:: python

   from zennit.attribution import Gradient
   from signxai2.composites import EpsilonStdXSIGN

   composite = EpsilonStdXSIGN(mu=0, stdfactor=0.3, signstdfactor=0.3)
   with Gradient(model=model, composite=composite) as attributor:
        output, relevance = attributor(data, target)

   print('EpsilonStdXSIGN:', relevance)

More information on attributors can be found here:

- `Writing Custom Composites <https://zennit.readthedocs.io/en/latest/how-to/write-custom-composites.html>`_
- `Writing Custom Canonizers <https://zennit.readthedocs.io/en/latest/how-to/write-custom-canonizers.html>`_
- `Writing Custom Rules <https://zennit.readthedocs.io/en/latest/how-to/write-custom-rules.html>`_
- `Writing Custom Attributors <https://zennit.readthedocs.io/en/latest/how-to/write-custom-attributors.html>`_

Example Scripts
--------------

Ready-to use examples to analyze image and time series models can be found here:

- `vgg16_simple.py <https://github.com/TimeXAIgroup/signxai2/blob/main/examples/vgg16_simple.py>`_
- `vision_transformer.py <https://github.com/TimeXAIgroup/signxai2/blob/main/examples/vision_transformer.py>`_
- `dftlrp_synthetic.py <https://github.com/TimeXAIgroup/signxai2/blob/main/examples/dftlrp_synthetic.py>`_
