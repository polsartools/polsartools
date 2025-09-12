Radar Vegetation Index (``rvi_dp``)
===================================
.. This functionality computes the radar vegetation index for dual polarimetric (HH | HV), (VV | VH) SAR data. The required input and the computed output are as follows:

.. .. code-block:: python

..         input : input_c2_folder, window_size
..         output: RVI_dp.bin

.. autofunction:: polsartools.rvi_dp
   :noindex:

The formulation of RVI is as follows:

.. math::

    \text{RVI}_{dp} = \frac{4 \times \sigma^\circ_{\text{XY}}}{\sigma^\circ_{\text{XX}}+\sigma^\circ_{\text{XY}}}  

where, :math:`\sigma^\circ_{\text{XX}}` is co-pol backscatter intensity and :math:`\sigma^\circ_{\text{XY}}` is corss-pol backscatter intensity