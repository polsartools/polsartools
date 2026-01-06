<p align="center">
  <img src="logo.png" alt=""/>
</p>

## A python package for processing Polarimetric Synthetic Aperture Radar (PolSAR) data.

![Silver Medal](https://img.shields.io/badge/Award-ISRS--ICON_Silver-silver?style=flat-square&logo=github) 
[![Build](https://github.com/polsartools/polsartools/actions/workflows/ci.yml/badge.svg)](https://github.com/polsartools/polsartools/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/polsartools/badge/?version=latest)](https://polsartools.readthedocs.io/en/latest/?badge=latest)

[![image](https://img.shields.io/pypi/v/polsartools.svg)](https://pypi.python.org/pypi/polsartools)
[![](https://static.pepy.tech/badge/polsartools)](https://pepy.tech/project/polsartools)

[![image](https://anaconda.org/bnarayanarao/polsartools/badges/version.svg)](https://anaconda.org/bnarayanarao/polsartools)
[![image](https://anaconda.org/bnarayanarao/polsartools/badges/downloads.svg)](https://anaconda.org/bnarayanarao/polsartools/files)

[![GitHub commits](https://img.shields.io/github/commits-since/polsartools/polsartools/0.9.svg)](https://GitHub.com/polsartools/polsartools/commit/)
[![License: GPL 3.0](https://img.shields.io/badge/License-GPL_3.0-green.svg)](https://opensource.org/licenses/gpl-license)



<!-- ## Jointly Developed By

| [![MRSLab](docs/contributions/mrslab_logo.png)](http://mrslab.in) | [![MiRSL](docs/contributions/mirsl_logo.png)](https://www.umass.edu/microwave-remote-sensing) |
|:--:|:--:|
| **Microwave Remote Sensing Lab (MRSLab),** <br> Indian Institute of Technology Bombay, India | **Microwave Remote Sensing Laboratory (MiRSL),** <br> University of Massachusetts Amherst, USA | 
 -->

<h2>Jointly Developed By</h2>

<table align="center">
  <tr>
    <td align="center" style="padding: 20px;">
      <a href="http://mrslab.in">
        <img src="docs/contributions/mrslab_iitb.png" alt="MRSLab" width="200"><br>
        <strong>Microwave Remote Sensing Lab (MRSLab)<br>Indian Institute of Technology Bombay, India</strong>
      </a>
    </td>
    <td align="center" style="padding: 20px;">
      <a href="https://www.umass.edu/microwave-remote-sensing">
        <img src="docs/contributions/mirsl_umass.png" alt="MIRSL" width="250"><br>
        <strong>Microwave Remote Sensing Laboratory (MiRSL)<br>University of Massachusetts Amherst, USA</strong>
      </a>
    </td>
  </tr>
</table>



## 🏆 Awards & Recognition

### **ISRS I-CON (Silver)**
This award was presented by the **Indian Society of Remote Sensing (ISRS)** during the *Innovation Contest for Geo Spatial Information Technology-2025*.

**Why this work was recognized:**
Our project was honored for addressing the "Big Data" challenges of the current generation of SAR missions (such as NISAR, EOS-04, BIOMASS, and Sentinel-1). The judges recognized **PolSARtools** for:

* ☁️ **Cloud-Native Innovation:** Bridging the gap between raw polarimetric data and **Analysis Ready Data (ARD)** using cloud-optimized formats.
* 🔓 **Open Science:** Promoting inclusive research by lowering the coding barrier through our integrated **QGIS plugin**.
* 🏗️ **Scalability:** Providing a reproducible architecture capable of handling the petabyte-scale influx of modern satellite imagery.




## 💠 General Information
This package generates derived SAR parameters (viz. polarimetric descriptors, vegetation indices, polarimetric decomposition parameters) from various SAR sensors or input polarimetric matrix (S2, C4, C3, T4, T3, Sxy, C2, T2). 


## 💠 Installation
1. **Install `gdal` Package**
	
	```bash
	conda install gdal -c conda-forge
	```

2. **Install `polsartools` Package**

	You may choose any of the following options 
	  
	  - a. `pip` (stable release):
	
		```bash
		pip install polsartools
		```
	
	 - b. `conda` (stable release)
	
		```bash
		conda install polsartools -c bnarayanarao
		```
	
	 - c. GitHub (Weekly Build)
	
		```bash
		pip install git+https://github.com/polsartools/polsartools.git#egg=polsartools
		```
		Use this if you encounter errors like: `AttributeError: module 'polsartools' has no attribute 'xyzabc'`
		
		>**Note for Windows users:** 
		> If installing via GitHub (option c), make sure to install Microsoft C++ build tools first. 
		Download here: https://visualstudio.microsoft.com/visual-cpp-build-tools


## 💠 Example Usage

The following code example demonstrates a typical processing pipeline using ```polsartools``` on NASA-ISRO SAR (NISAR) Geocoded Single-look Complex (GSLC) data. It includes polarimetric covariance matrix (C3) extraction, Pauli RGB visualization, speckle filtering,  H/A/α decomposition, and plotting in the H/α and H/A/α feature spaces.  

```python
import polsartools as pst

def main():
  # Generate C3 from NISAR GSLC full-pol data with 5x5 multilooking
  pst.import_nisar_gslc('path/to/nisar_gslc.h5',mat='C3',
                    azlks=5, rglks=5,
                    fmt='tif', cog=False, 
                    )

  # Visualize Pauli-decomposition RGB
  c3path = 'path/to/nisar_gslc/C3'
  pst.pauli_rgb(c3path)

  # Apply 3x3 refined-Lee speckle filter
  pst.filter_refined_lee(c3path,win=3)

  # Perform H-A-Alpha decomposition
  c3_rlee = 'path/to/nisar_gslc/rlee_3x3/C3'
  pst.h_a_alpha_fp(c3_rlee)

  # Generate H-Alpha 2d plot
  entropy_path = c3_rlee+'/H_fp.tif'
  alpha_path = c3_rlee+'/alpha_fp.tif'
  pst.plot_h_alpha_fp(entropy_path, alpha_path,
                      ppath='./halpha_2D.png')

  # Generate H-A-Alpha 3d plot
  ani_path = c3_rlee + '/anisotropy_fp.tif'
  pst.plot_h_a_alpha_fp(entropy_path, ani_path,alpha_path,
                      ppath='./haalpha_3D.png')

if __name__ == "__main__":
    main()

```


More example use cases and notebooks are provided at [polsartools-notebooks](https://github.com/polsartools/polsartools-tutorials) repo. Detailed documentation is available at [polsartools.readthedocs.io](https://polsartools.readthedocs.io/en/latest/) 


## 💠 Available functionalities:
Full list of available functions is provided here : [Functions](https://polsartools.readthedocs.io/en/latest/files/02functions.html)


## 🎨 Logo Vibes Explained

The package logo visually encapsulates key concepts in PolSAR data processing:

- **Poincaré Sphere Representation**: Central to the logo is a stylized visualization of the Poincaré sphere, highlighting the diversity of polarization states encountered in SAR imaging.
- **Huynen Polarization Fork**: A dotted elliptical curve represents the great circle containing Huynen's characteristic polarization states: co-polarized maximum, saddle point, and minima. These vectors lie in a common plane, revealing target symmetries and scattering behaviors.
- **Quadrant Background**: The four background squares reflect brightness variations in SAR intensity data, showcasing:
  - Bright vs. dark reflectivity regions
  - Co-polarized (diagonal) vs. cross-polarized (off-diagonal) intensity distribution

> Designed to reflect what this package does best: demystify PolSAR, one pixel at a time.

## 💠 Contributing

We welcome contributions! Whether it's fixing bugs, adding new features, or improving documentation, your help is greatly appreciated.

### How to Contribute
1. **Fork the repository** - Fork this repository to your GitHub account.

2. **Clone your fork** - Clone the repository to your local machine:

    ```bash
    git clone https://github.com/polsartools/polsartools.git
    ```

3. Create a branch - Create a new branch for your changes:

    ```bash 
    git checkout -b feature-branch
    ```
4. **Make changes** - Implement your changes or additions.

5. **Test your changes** - Run the tests to ensure that your changes don’t break anything.

6. **Commit and push** - Commit your changes and push them to your fork:
    ```bash
        git commit -am "Description of changes"
        git push origin feature-branch
    ```
7. **Create a Pull Request** - Open a pull request to the main repository with a clear description of the changes.

<!-- For more detailed guidelines on contributing, see the CONTRIBUTING.md (if available). -->


## 💠 Bug Reporting

If you encounter a bug or issue, please follow these steps to report it:

1. Check the existing issues: Before submitting a new bug report, check if the issue has already been reported in the [Issues section](https://github.com/polsartools/polsartools/issues).

2. Submit a bug report: If the issue hasn’t been reported, please open a new issue and include the following information:
    * A clear description of the problem.
    * Steps to reproduce the issue.
    * Expected vs actual behavior.
    * Any error messages or stack traces.
    * Relevant code snippets or files if possible.
    * Version of `polsartools` and Python you're using.

[Click here to report a bug](https://github.com/polsartools/polsartools/issues/new?template=bug_report.md)


## 💠 Feature Requests

We’re always open to suggestions for new features or improvements!

1. **Check existing feature requests:** Please make sure the feature request hasn't already been made in the [Issues section](https://github.com/polsartools/polsartools/issues).

2. **Submit a feature request:** If it hasn’t been requested already, please open a new issue with the following information:
      * A clear description of the feature.
      * Why you think this feature would be beneficial.
      * Any specific use cases or examples.

     [Click here to request a feature](https://github.com/polsartools/polsartools/issues/new?template=feature_request.md)

## 💠 Cite

If you use **`polsartoos`** in your research or projects, please cite the official journal paper:

> Bhogapurapu, N., Siqueira, P., & Bhattacharya, A. 2025. **polsartools: A Cloud-Native Python Library for Processing Open Polarimetric SAR Data at Scale**. *SoftwareX*, 33, 102490. doi: [10.1016/j.softx.2025.102490](http://dx.doi.org/10.1016/j.softx.2025.102490)

### BibTeX

```bibtex
@article{bhogapurapu2025polsartools,
title = {Polsartools: A cloud-native python library for processing open polarimetric SAR data at scale},
author = {Narayanarao Bhogapurapu and Paul Siqueira and Avik Bhattacharya},journal = {SoftwareX},
volume = {33},
pages = {102490},
year = {2026},
publisher={Elsevier},
issn = {2352-7110},
doi = {https://doi.org/10.1016/j.softx.2025.102490},
}

## 💠 Funding
The author(s) would like to thank a NASA NISAR grant (#80NSSC22K1869) and the Multi-Mission Algorithm and Analysis Platform (MAAP) for supporting and motivating this work.