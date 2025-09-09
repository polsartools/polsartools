Getting started
===============

.. `polsartools` package generates derived SAR parameters (polarimetric decomposition parameters and descriptors) from input polarimetric matrix (C3, T3, C2, T2). The input data needs to be in `PolSARpro`_/`ENVI`_ format (\*.bin and \*.hdr). It requires `gdal`_, `numpy`_, `scipy`_, `matplotlib`_ python libraries pre-installed.

**PolSARtools** package generates derived SAR parameters (viz. polarimetric descriptors, vegetation indices, polarimetric decomposition parameters) from different SAR sensors or input polarimetric matrix (C3, T3, C2, T2).

Installation
-------------

1. Install **`gdal`** Package

.. code-block:: bash

    conda install gdal -c conda-forge

2. Install **`polsartools`** Package

You can install it using `pip`:

.. code-block:: bash

    pip install polsartools

If you encounter an error like "function not found" or need the latest version, you can install the updated version directly from GitHub:

.. code-block:: bash

    pip install git+https://github.com/polsartools/polsartools.git#egg=polsartools

Example Usage
-------------

Sample use cases and notebooks are provided at `polsartools-notebooks <https://github.com/polsartools/polsartools-tutorials>`_ repo. Detailed documentation is available at `polsartools.readthedocs.io <https://polsartools.readthedocs.io/en/latest/>`_.

.. Available functions
.. -------------------
.. * Supported sensors

..   * `UAVSAR (GRD, MLC) <sensors/uavsar.html>`_
..   * `NISAR (RSLC, GSLC) <sensors/nisar.html>`_
..   * `RADARSAT-2 (Full-pol) <sensors/radarsat2.html>`_
..   * `ALOS-2 (Fine Beam Dual-pol (FBD) Level 1.1 CEOS) <sensors/alos2.html>`_
..   * `Chandrayaan-II DFSAR (Full-pol) <sensors/chandrayaan2.html>`_

.. * Full-pol

..   * `H-Alpha decomposition <functions/full_pol/halphafp.html>`_
..   * Shannon Entropy parameters
..   * Non-negative Eigen value decomposition
..   * Neumann Decomposition
..   * Model free 4-Component decomposition for full-pol data
..   * Model free 3-Component decomposition for full-pol data 
..   * Radar Vegetation Index 
..   * Generalized volume Radar Vegetation Index
..   * Polarimetric Radar Vegetation Index
..   * Degree of Polarization

.. * Compact-pol

..   * Model free 3-Component decomposition for compact-pol data
..   * Improved S-Omega decomposition for compact-pol data
..   * Compact-pol Radar Vegetation Index
..   * Degree of Polarization

.. * Dual-pol

..   * H-Alpha parameters
..   * Shannon Entropy parameters
..   * Dual-pol Radar Vegetation Index 
..   * Dual-pol Radar Vegetation Index for GRD data
..   * Radar Vegetation Index
..   * Degree of Polarization
..   * Polarimetric Radar Vegetation Index
..   * Dual-pol descriptors
..   * Model free 3-Component decomposition for dual-copol data

.. * Polarimetric speckle filters

..   * boxcar
..   * refine lee
  
.. * other functions

..   * Generate pauliRGB for FP data
..   * Generate false color RGB for DP/CP data
..   * convert_C3_T3
..   * convert_T3_C3
..   * multi-looking



Contributing
-------------

We welcome contributions! Whether it's fixing bugs, adding new features, or improving documentation, your help is greatly appreciated.

How to Contribute
~~~~~~~~~~~~~~~~~

1. **Fork the repository** - Fork this repository to your GitHub account.

2. **Clone your fork** - Clone the repository to your local machine:

   .. code-block:: bash

       git clone https://github.com/polsartools/polsartools.git

3. Create a branch - Create a new branch for your changes:

   .. code-block:: bash

       git checkout -b feature-branch

4. **Make changes** - Implement your changes or additions.

5. **Test your changes** - Run the tests to ensure that your changes don’t break anything.

6. **Commit and push** - Commit your changes and push them to your fork:

   .. code-block:: bash

       git commit -am "Description of changes"
       git push origin feature-branch

7. **Create a Pull Request** - Open a pull request to the main repository with a clear description of the changes.

Bug Reporting
~~~~~~~~~~~~~~

If you encounter a bug or issue, please follow these steps to report it:

1. Check the existing issues: Before submitting a new bug report, check if the issue has already been reported in the `Issues section <https://github.com/polsartools/polsartools/issues>`_.

2. Submit a bug report: If the issue hasn’t been reported, please open a new issue and include the following information:
    * A clear description of the problem.
    * Steps to reproduce the issue.
    * Expected vs actual behavior.
    * Any error messages or stack traces.
    * Relevant code snippets or files if possible.
    * Version of `polsartools` and Python you're using.

`Click here to report a bug <https://github.com/polsartools/polsartools/issues/new?template=bug_report.md>`_

Feature Requests
~~~~~~~~~~~~~~~~~
We’re always open to suggestions for new features or improvements!

1. **Check existing feature requests:** Please make sure the feature request hasn't already been made in the `Issues section <https://github.com/polsartools/polsartools/issues>`_.

2. **Submit a feature request:** If it hasn’t been requested already, please open a new issue with the following information:
    * A clear description of the feature.
    * Why you think this feature would be beneficial.
    * Any specific use cases or examples.

`Click here to request a feature <https://github.com/polsartools/polsartools/issues/new?template=feature_request.md>`_




.. _PolSARpro: https://earth.esa.int/web/polsarpro/home
.. _ENVI: https://www.l3harrisgeospatial.com/Software-Technology/ENVI
.. _gdal: https://gdal.org/en/latest/
.. _scipy: https://scipy.org/
.. _numpy: https://numpy.org/
.. _matplotlib: https://matplotlib.org/
.. _releases: https://github.com/polsartools/polsartools/releases