improved S-:math:`\Omega` decomposition (``s_omega``)
========================================================
.. This functionality computes the scattering powers for compact polarimetric SAR data. This is an improved decomposition technique based on Stokes vector(S) and the polarized power fraction (:math:`\Omega`). The required input and the computed output are as follows:

.. .. code-block:: python

..     input : input_C2_folder, window_size, tau, psi, chi
..     output: Ps_iSOmega.bin, Pd_iSOmega.bin,Pv_iSOmega.bin

.. autofunction:: polsartools.s_omega
   :noindex:



The stokes paramters can be written in terms of the covariance matrx (C2) elements as follows:

.. math::

    S_0=\text{C11+C22};\qquad{}S_1=\text{C11-C22};\\
    S_2=\text{C12+C21};\qquad{}S_3=\pm\text{j(C12-C21)}

Then, the parameters Same-sense Circular (:math:`\text{SC}`) and Opposite-sense Circular (:math:`\text{OC}`) can be expressed as follows:

.. math::

    \text{SC}=\frac{S_0-S_3}{2};\qquad{}\text{OC}=\frac{S_0+S_3}{2};

Now, based on the ratio of :math:`\text{SC}` and :math:`\text{OC}` the decomposition powers can be derived as given below. Further details can be found in [[7]](#7)

.. math::

    \text{SC/OC}<1;\qquad{}\qquad{}\qquad{}\qquad{}\qquad{}\qquad{}\qquad{}\text{SC/OC}>1\\P_s=\Omega\left(S_{0}-\left(1-\Omega\right)\text{SC}\right);\qquad{}\qquad{}\qquad{}P_s=\Omega\left(1-\Omega\right)\text{OC}\\P_d=\Omega\left(1-\Omega\right)\text{SC};\qquad{}\qquad{}\qquad{}P_d=\Omega\left(S_{r0}-\left(1-\Omega\right)\text{OC}\right)
    \\P_v=S_{0}\left(1-\Omega\right)\qquad{}\qquad{}\qquad{}\qquad{}\qquad{}P_v=S_{0}\left(1-\Omega\right)