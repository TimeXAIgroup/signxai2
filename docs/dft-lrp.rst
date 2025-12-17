====================
DFT-LRP
====================

Based on the below paper, we included the previously published utility codes to enable generating frequency-aware explanations though *virtual inspection layers*. This is essentially applicable to **time series data**, enabling a detailled view on frequency components.

.. code-block:: bibtex

    @article{Vielhaben2024,
        title = {Explainable {AI} for time series via Virtual Inspection Layers},
        journal = {Pattern Recognition},
        volume = {150},
        pages = {110309},
        year = {2024},
        issn = {0031-3203},
        doi = {10.1016/j.patcog.2024.110309},
        author = {Johanna Vielhaben and Sebastian Lapuschkin and Grégoire Montavon and Wojciech Samek},
    }

Basic usage:

.. code-block:: python

   from zennit.attribution import Gradient
   from signxai2.composites import EpsilonStdX

   composite = EpsilonStdX(mu=0, stdfactor=0.3, signstdfactor=0.3)
   with Gradient(model=model, composite=composite) as attributor:
        output, relevance = attributor(data, target)

   signal_freq, relevance_freq, signal_timefreq, relevance_timefreq = calculate_dft_explanation(signal_time, relevance_time)

   print('EpsilonStdX-DFT-Freq:', relevance_freq)
   print('EpsilonStdX-DFT-TimeFreq:', relevance_timefreq)

