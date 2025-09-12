Degree of Polarization (``dop_dp``)
====================================
.. This functionality computes the 2D Barakat degree of polarization for dual polarimetric (HH | HV), (VV | VH) SAR data. The required input and the computed output are as follows:

.. .. code-block:: python

..     input : input_c2_folder, window_size
..     output: dop_dp.bin

.. autofunction:: polsartools.dop_dp
   :noindex:


.. math::

    \text{DOP}_{dp}=\sqrt{1-\frac{4\times\text{det([C2])}}{\text{(Trace[C2])}^2}}

where, :math:`\text{[C2]}` is co-variance matrix. Further details on the Barakat Degree of polarization can be found in [[10]](#10)


