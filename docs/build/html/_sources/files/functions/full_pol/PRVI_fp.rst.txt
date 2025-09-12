Polarimetric Radar Vegetation Index (``prvi_fp`` )
==================================================
.. This functionality computes the polarimetric Radar vegetation index for full polarimetric SAR data. The required input and the computed output are as follows:

.. .. code-block:: python

..         input : input_T3/C3_folder, window_size
..         output: PRVI_FP.bin

.. autofunction:: polsartools.prvi_fp
   :noindex:

   
The formlation of PRVI interms of degree of polarization and cross-pol backscatter intensity can be expressed as follows: 

.. math::

    \text{PRVI}_{fp}=(1-\text{DOP}_{fp})\sigma^\circ_{\text{XY}}


where, :math:`\text{DOP}_{fp}` 3D Barakt degree of polarization and can be expressed as shown below. Further details on the PRVI can be found in [[1]](#1)

.. math::

    \text{DOP}_{fp}=\sqrt{1-\frac{27\times\text{det([T3])}}{\text{(Trace[T3])}^3}}

