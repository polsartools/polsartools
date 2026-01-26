🔸NISAR
========


RSLC (``import_nisar_rslc``)
----------------------------
The `import_nisar_rslc` function extracts Sxy/C2/T2 (for dual-pol), S2/C4/T4/C3/T3/C2/T2 (for full-pol) matrix elements from the given NISAR RSLC `.h5` file and saves them into a C2 directory.

.. autofunction:: polsartools.import_nisar_rslc
   :noindex:



GSLC (``import_nisar_gslc``)
----------------------------

The `import_nisar_gslc` function extracts Sxy/C2/T2 (for dual-pol), S2/C4/T4/C3/T3/C2/T2 (for full-pol) matrix elements from the given NISAR GSLC `.h5` file and saves them into a C2 directory.

.. autofunction:: polsartools.import_nisar_gslc
   :noindex:


GCOV (``import_nisar_gcov``)
----------------------------

The `import_nisar_gcov` function extracts intensity elements (dual-pol: HHHH,HVHV or VVVV,VHVH; Full-pol: HHHH,HVHV,VHVH,VVVV) from the given NISAR GCOV `.h5` file and saves them into a II directory.

.. autofunction:: polsartools.import_nisar_gcov
   :noindex:
