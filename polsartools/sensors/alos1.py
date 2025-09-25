"""
Some how not matching the polsarpro output, could be because of GDAL CEOS driver issue
https://gdal.org/en/stable/drivers/raster/sar_ceos.html
https://github.com/OSGeo/gdal/blob/release/3.11/frmts/ceos2/sar_ceosdataset.cpp

"""


import numpy as np
from osgeo import gdal
import os, glob
import tempfile,shutil
from polsartools.utils.utils import time_it, mlook_arr
# from polsartools.utils.io_utils import write_T3, write_C3
from polsartools.preprocess.convert_S2 import convert_S
gdal.UseExceptions()
from polsartools.sensors.alos2 import write_a2_rst


def get_band_by_pol(pol_name,pol_map,dataset):
    band_index = pol_map.get(pol_name)
    if band_index is None:
        raise ValueError(f"Polarization '{pol_name}' not found.")
    band = dataset.GetRasterBand(band_index)
    return band.ReadAsArray()

@time_it    
def alos1_l11(in_dir,mat='T3', azlks=8,rglks=4,
                  fmt='tif', cog=False,ovr = [2, 4, 8, 16],comp=False,
                  out_dir=None,
                  recip=False,cf_dB=-83):

    """
    Extracts single look S2 or Multi-look T3/C3 matrix elements from ALOS-2 Quad-Pol (HBQ) CEOS data 
    and saves them into respective binary files.

    Example:
    --------
    >>> alos1_l11("path_to_folder", azlks=5, rglks=3)
    This will extract the T3 matrix elements from the ALOS-1 Level 1.1 data 
    in the specified folder and save them in the 'C2' directory.
    
    Parameters:
    -----------
    in_dir : str
        The path to the folder containing the ALOS-2 Quad-Pol (HBQ) CEOS data folder.
    
    mat : str, optional (default='T3')
        Type of matrix to extract. Valid options: 'S2',  'C4, 'C3', 'T4', 
        'T3', 'C2HX', 'C2VX', 'C2HV','T2HV'
        
    azlks : int, optional (default=8)
        The number of azimuth looks for multi-looking.

    rglks : int, optional (default=4)
        The number of range looks for multi-looking.

    fmt : {'tif', 'bin'}, optional (default='tif')
        Output format:
        - 'tif': GeoTIFF
        - 'bin': Raw binary format

    cog : bool, optional (default=False)
        If True, outputs will be saved as Cloud Optimized GeoTIFFs with internal tiling and overviews.

    ovr : list of int, optional (default=[2, 4, 8, 16])
        Overview levels for COG generation. Ignored if cog=False.

    comp : bool, optional (default=False)
        If True, applies LZW compression to GeoTIFF outputs.

    out_dir : str or None, optional (default=None)
        Directory to save output files. If None, a folder named after the matrix type will be created
        in the same location as the input file.
        
    recip : bool, optional (default=False)
        If True, scattering matrix reciprocal symmetry is applied, i.e, S_HV = S_VH.        
    cf_dB : float, optional (default=-83)
        The calibration factor in dB used to scale the raw radar data. It is applied to 
        the HH and HV polarization data before matrix computation.

    Returns:
    --------
    None
        The function does not return any value. Instead, it creates a folders named 'S2` or 'C3` or 'T3` 
        (depending on the chosen matrix) and saves the corresponding binary files.

    """
    valid_full_pol = ['S2', 'C4', 'C3', 'T4', 'T3', 'C2HX', 'C2VX', 'C2HV', 'T2HV']
    valid_matrices = valid_full_pol

    if mat not in valid_matrices:
        raise ValueError(f"Invalid matrix type '{mat}'. \n Supported types are:\n"
                        f"  Full-pol: {sorted(valid_full_pol)}")



    temp_dir = None
    ext = 'bin' if fmt == 'bin' else 'tif'
    driver = 'ENVI' if fmt == 'bin' else None

    # Final output directory
    if out_dir is None:
        final_out_dir = os.path.join(in_dir, mat)
    else:
        final_out_dir = os.path.join(out_dir, mat)
    os.makedirs(final_out_dir, exist_ok=True)

    # Intermediate output directory
    if mat in ['S2', 'Sxy']:
        base_out_dir = final_out_dir
    else:
        temp_dir = tempfile.mkdtemp(prefix='temp_S2_')
        base_out_dir = temp_dir

    if cf_dB==0 or cf_dB is None:
        calfac_linear = 1
    else:
        calfac_linear = np.sqrt(10 ** ((cf_dB - 32) / 10))

    vol_file = glob.glob(os.path.join(in_dir, "VOL-*"))[0]

    if len(vol_file) == 0:
        raise ValueError("No ALOS-1 VOLUME file found in the directory.")

    dataset = gdal.Open(vol_file)
    if dataset is None:
        raise RuntimeError("Failed to open dataset.")
    pol_map = {}
    for i in range(1, dataset.RasterCount + 1):
        band = dataset.GetRasterBand(i)
        pol = band.GetMetadata().get('POLARIMETRIC_INTERP')
        if pol:
            pol_map[pol] = i

    print(f"Available polarizations ({len(pol_map)}): {list(pol_map.keys())}")

    #%%
    if len(pol_map)==4:
        S11 = get_band_by_pol('HH',pol_map,dataset).astype(np.complex64)*calfac_linear
        write_a2_rst(os.path.join(base_out_dir, f's11.{ext}'),S11,   driver=driver, mat=mat, cog=cog, ovr=ovr, comp=comp)
        del S11        
        
        S21 = get_band_by_pol('HV',pol_map,dataset).astype(np.complex64)*calfac_linear
        S12 = get_band_by_pol('VH',pol_map,dataset).astype(np.complex64)*calfac_linear
        if recip:
            S12 = (S12 + S21)/2
            S21 = S12
        write_a2_rst(os.path.join(base_out_dir, f's12.{ext}'),S12,   driver=driver, mat=mat, cog=cog, ovr=ovr, comp=comp)
        write_a2_rst(os.path.join(base_out_dir, f's21.{ext}'),S21,  driver=driver, mat=mat, cog=cog, ovr=ovr, comp=comp)
        
        del S12, S21
        S22 = get_band_by_pol('VV',pol_map,dataset).astype(np.complex64)*calfac_linear       
        write_a2_rst(os.path.join(base_out_dir, f's22.{ext}'),S22,  driver=driver, mat=mat, cog=cog, ovr=ovr, comp=comp)
        del S22
        # Matrix conversion if needed
        if mat not in ['S2', 'Sxy']:
            convert_S(base_out_dir, mat=mat, azlks=azlks, rglks=rglks, cf=1,
                    fmt=fmt, out_dir=final_out_dir, cog=cog, ovr=ovr, comp=comp)

            # Clean up temp directory
            if temp_dir:
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    print(f"Warning: Could not delete temporary directory {temp_dir}: {e}")
