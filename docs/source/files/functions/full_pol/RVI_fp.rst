Radar Vegetation Index (``rvi_fp`` )
=====================================
.. This functionality computes the Radar vegetation index for full polarimetric SAR data. The required input and the computed output are as follows:
    
.. .. code-block:: python

..     input : input_T3/C3_folder, window_size
..     output : RVI_FP.bin

.. autofunction:: polsartools.rvi_fp
   :noindex:

The formulation of RVI is as follows:

.. math::

    \text{RVI}_{fp} = \frac{4 \times \lambda_3}{\lambda_1+\lambda_2+\lambda_3} 

where, :math:`\lambda_1, \lambda_2` and :math:`\lambda_3` are the eigen values of coherency matrix (T3) in descending order (:math:`\lambda_1 > \lambda_2 > \lambda_3`). Further details can be found in [[8]](#8)