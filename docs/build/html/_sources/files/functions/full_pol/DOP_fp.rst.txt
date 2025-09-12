Degree of Polarization (``dop_fp``)
====================================
.. This functionality computes the 3D Barakat degree of polarization for full polarimetric SAR data. The required input and the computed output are as follows:

.. .. code-block:: python

..     input : input_T3/C3_folder, window_size
..     output: DOP_FP.bin

.. autofunction:: polsartools.dop_fp
   :noindex:


.. math::

    \text{DOP}_{fp}=\sqrt{1-\frac{27\times\text{det([T3])}}{\text{(Trace[T3])}^3}}

Further details on the Barakat Degree of polarization can be found in [[10]](#10)

