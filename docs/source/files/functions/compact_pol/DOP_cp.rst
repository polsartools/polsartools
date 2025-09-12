Degree of Polarization (``dop_cp``)
====================================
.. This functionality computes the degree of polarization for compact polarimetric SAR data. The required input and the computed output are as follows:

.. .. code-block:: python

..         input : input_c2_folder, window_size, tau
..         output: DOP_CP.bin


.. autofunction:: polsartools.dop_cp
   :noindex:



The conventional degree of polarization in terms of stokes paramters can be written as follows:

.. math::

    \text{DOP}_{cp}=\frac{\sqrt{S^2_1+S^2_2+S^2_3}}{S_0}

where, 

.. math::
    
    S_0=\text{C11+C22};\qquad{}S_1=\text{C11-C22};\\
    S_2=\text{C12+C21};\qquad{}S_3=\pm\text{j(C12-C21)}



