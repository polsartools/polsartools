Functions
=========

Full-pol functions
-------------------
Full-pol functionalities require the SAR data in the form of scattering (S2), covariance (C3) or coherency matrix (T3). 
A typical file structures of S2, T3 and C3 matrices are provded in: 

.. toctree::
    :maxdepth: 2

    functions/dataFP

Following are the avaialble functions for full-pol data:

.. toctree::
    :maxdepth: 3

    functions/full_pol/halphafp
    functions/full_pol/tsvm
    functions/full_pol/yam4cfp
    functions/full_pol/freeman_3c
    functions/full_pol/freeman_2c
    functions/full_pol/MF3CF
    functions/full_pol/mf4cf
    functions/full_pol/neufp
    functions/full_pol/nnedfp
    functions/full_pol/shannon_h_fp
    functions/full_pol/praks_parm_fp
    functions/full_pol/RVI_fp
    functions/full_pol/GRVI
    functions/full_pol/PRVI_fp
    functions/full_pol/DOP_fp
 

Compact-pol functions
----------------------
Compact-pol functionalities require the SAR data in the form of 2x2 covariance matrix (C2). 
A typical file structure of C2 matrix is provded in:

.. toctree::
    :maxdepth: 2

    functions/dataCP

Following are the avaialble functions for compact-pol data:

.. toctree::
    :maxdepth: 3

    functions/compact_pol/cprvi
    functions/compact_pol/iS_Omega
    functions/compact_pol/MF3CC
    functions/compact_pol/DOP_cp



Dual-pol functions
------------------

Dual-pol functionalities require the SAR data in the form of 2x2 covariance matrix (C2) or 2x2 coherency matrix (T2) for dual-co-pol. 
A typical file structures of C2/T2 matrix is provded in:

.. toctree::
    :maxdepth: 2

    functions/dataDP


Following are the avaialble functions for dual-pol data:

.. toctree::
    :maxdepth: 3

    functions/dual_pol/RVI_dp
    functions/dual_pol/DpRVI
    functions/dual_pol/PRVI_dp
    functions/dual_pol/DOP_dp
    functions/dual_pol/halphadxp
    functions/dual_pol/shannon_h_dp
    functions/dual_pol/mf3cd
    functions/dual_pol/dprvic
    functions/dual_pol/dp_desc
    functions/dual_pol/dprbi
    functions/dual_pol/dprsi
    functions/dual_pol/powers_dp
    functions/dual_pol/dprbic
    functions/dual_pol/dprsic
    functions/dual_pol/powers_dp_grd



Analysis functions
-------------------

.. toctree::
    :maxdepth: 3

    functions/analysis
    functions/full_pol/halpha_plot_fp
    functions/dual_pol/halpha_plot_dp
    functions/analysis/haalpha_plot_fp
    functions/analysis/halpha_cluster_fp
    functions/analysis/htheta_plot_fp
    functions/analysis/htheta_plot_cp
    
Polarimetric speckle filters
----------------------------

.. toctree::
    :maxdepth: 2

    functions/speckle_filts

Preprocessing
--------------

.. toctree::
    :maxdepth: 3

    functions/preprocess/clip
    functions/preprocess/mlook
    functions/preprocess/convert_S

Other functions
-----------------------

.. toctree::
    :maxdepth: 3

    functions/misc

    


