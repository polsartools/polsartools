Polarimetric Radar Vegetation Index (``prvi_dp``)
==================================================
.. This functionality computes the polarimetric radar vegetation index for dual polarimetric (HH | HV), (VV | VH) SAR data. The required input and the computed output are as follows:

.. .. code-block:: python

..         input : input_c2_folder, window_size
..         output: PRVI_dp.bin

.. autofunction:: polsartools.prvi_dp
   :noindex:

The formulation of PRVI is as follows: 

.. math::

    \text{PRVI}_{dp}=\Big(1-\sqrt{1-\frac{4\times\text{det([C2])}}{\text{(Trace[C2])}^2}}\Big)\sigma^\circ_{\text{XY}}

where, :math:`\text{[C2]}` is co-variance matrix and :math:`\sigma^\circ_{\text{XY}}` is corss-pol backscatter intensity.
