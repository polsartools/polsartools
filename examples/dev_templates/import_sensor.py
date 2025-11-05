
import numpy as np
from osgeo import gdal
import os, glob
import tempfile,shutil
from polsartools.utils.utils import time_it, mlook_arr
from polsartools.preprocess.convert_S2 import convert_S
gdal.UseExceptions()


def read_bin(file):
    ds = gdal.Open(file)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    return arr

def read_a2(file):
    
    fp = open(file,mode='rb')
    fp.seek(232)
    ch = int(fp.read(4))
    # print(ch)
    
    fp.seek(236)
    nline = int(fp.read(8))
    # print(nline)
    fp.seek(248)
    npixel = int(fp.read(8))
    # print(npixel)
    
    nrec = 544 + npixel*8
    # print(nrec)
    fp.seek(720)
    data = np.frombuffer(fp.read(int(nrec * nline)), dtype='>f4')
    data = np.array(data).reshape(-1,int(nrec/4)) 
    # print(np.shape(data))
    
    data = data[:,int(544/4):int(nrec/4)] 
    slc = data[:,::2] + 1j*data[:,1::2]
    # print(np.shape(slc))
    del data
    
    return slc

def write_a2_rst(out_file,data,
                driver='GTiff', out_dtype=gdal.GDT_CFloat32,
                mat=None,
               cog=False, ovr=[2, 4, 8, 16], comp=False
                 ):

    if driver =='ENVI':
        # Create GDAL dataset
        driver = gdal.GetDriverByName(driver)
        dataset = driver.Create(
            out_file,
            data.shape[1],      
            data.shape[0],      
            1,                   
            out_dtype    
        )


    else:
        driver = gdal.GetDriverByName("GTiff")
        options = ['BIGTIFF=IF_SAFER']
        if comp:
            # options += ['COMPRESS=DEFLATE', 'PREDICTOR=2', 'ZLEVEL=9']
            options += ['COMPRESS=LZW']
        if cog:
            options += ['TILED=YES', 'BLOCKXSIZE=512', 'BLOCKYSIZE=512']
        
        dataset = driver.Create(
            out_file,
            data.shape[1],      
            data.shape[0],      
            1,                   
            out_dtype,
            options    
        )

        
    dataset.GetRasterBand(1).WriteArray(data)
    # outdata.GetRasterBand(1).SetNoDataValue(0)##if you want these values transparent
    dataset.FlushCache() ##saves to disk!!
    
    if cog:
        dataset.BuildOverviews("NEAREST", ovr)
    dataset = None
    if mat == 'S2' or mat == 'Sxy':
        print(f"Saved file: {out_file}")

@time_it    
def sensor_dual_pol(in_dir,mat='C2', azlks=3,rglks=2,
                 fmt='tif', cog=False,ovr = [2, 4, 8, 16],comp=False,
                 out_dir=None,
                  cf_dB=-83):
    """
    Extracts the C2 matrix elements (C11, C22, and C12) from SENSOR_NAME Dual-Pol data 
    and saves them into respective binary files.

    Example:
    --------
    >>> sensor_dual_pol("path_to_folder", azlks=5, rglks=3)
    This will extract the C2 matrix elements from the SENSOR_NAME Dual-Pol data 
    in the specified folder and save them in the 'C2' directory.
    
    Parameters:
    -----------
    in_dir : str
        The path to the folder containing the SENSOR_NAME Dual-Pol data files.
    mat : str, optional (default = 'S2' or 'Sxy)
        Type of matrix to extract. Valid options: 'Sxy','C2', 'T2'.
    azlks : int, optional (default=3)
        The number of azimuth looks for multi-looking.

    rglks : int, optional (default=2)
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
                
    cf_dB : float, optional (default=-83)
        The calibration factor in dB used to scale the raw radar data. It is applied to 
        the HH and HV polarization data before matrix computation.

    Returns:
    --------
    None
        The function does not return any value. Instead, it creates a folder named `C2` 
        (if not already present) and saves the following binary files:

        - `C11.bin`: Contains the C11 matrix elements.
        - `C22.bin`: Contains the C22 matrix elements.
        - `C12_real.bin`: Contains the real part of the C12 matrix.
        - `C12_imag.bin`: Contains the imaginary part of the C12 matrix.
        - `config.txt`: A text file containing grid dimensions and polarimetric configuration.

    Raises:
    -------
    FileNotFoundError
        If the required SENSOR_NAME files (e.g., `IMG-HH` and `IMG-HV`) cannot be found in the specified folder.

    ValueError
        If the calibration factor is invalid or if the files are not in the expected format.


    """
    
    
    
    valid_dual_pol = ['Sxy', 'C2', 'T2']
    valid_matrices = valid_dual_pol

    if mat not in valid_matrices:
        raise ValueError(f"Invalid matrix type '{mat}'. \n Supported types are:\n"
                        f"  Dual-pol: {sorted(valid_dual_pol)}")
    
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
    if mat in ['Sxy']:
        base_out_dir = final_out_dir
    else:
        temp_dir = tempfile.mkdtemp(prefix='temp_S2_')
        base_out_dir = temp_dir
        

    # Get file paths    

    hh_file = list(glob.glob(os.path.join(in_dir,'IMG-HH-*-FBDR1.1__A')) + \
        glob.glob(os.path.join(in_dir, 'IMG-HH-*-FBDR1.1__D')))[0]

    hv_file = list(glob.glob(os.path.join(in_dir,'IMG-HV-*-FBDR1.1__A')) + \
        glob.glob(os.path.join(in_dir, 'IMG-HV-*-FBDR1.1__D')))[0]

    # Get Absolute calibration constant if required
    calfac_linear = np.sqrt(10 ** ((cf_dB - 32) / 10))

    # Read the dual-channel data into numpy arrays
    # Write the single look data; 
    S11 = read_a2(hh_file).astype(np.complex64)*calfac_linear 
    write_a2_rst(os.path.join(base_out_dir, f's11.{ext}'),S11,   driver=driver, mat=mat, cog=cog, ovr=ovr, comp=comp)
    del S11
    S12 = read_a2(hv_file).astype(np.complex64)*calfac_linear 
    write_a2_rst(os.path.join(base_out_dir, f's12.{ext}'),S12,   driver=driver, mat=mat, cog=cog, ovr=ovr, comp=comp)
    del S12
    
    
    # if required convert SLC into a multilooked matrices using convert_S function
    if mat in ['C2', 'T2']:
        convert_S(base_out_dir, mat=mat, azlks=azlks, rglks=rglks, cf=1,
                  fmt=fmt, out_dir=final_out_dir, cog=cog, ovr=ovr, comp=comp)

        # Clean up temp directory
        if temp_dir:
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Warning: Could not delete temporary directory {temp_dir}: {e}")

#################################################################################################

@time_it    
def sensor_full_pol(in_dir,mat='T3', azlks=8,rglks=4,
                  fmt='tif', cog=False,ovr = [2, 4, 8, 16],comp=False,
                  out_dir=None,
                  recip=False,cf_dB=-83):

    """
    Extracts single look S2 or Multi-look T3/C3 matrix elements from SENSOR_NAME Full-pol data 
    and saves them into respective binary files.

    Example:
    --------
    >>> alos2_hbq_l11("path_to_folder", azlks=5, rglks=3)
    This will extract the T3 matrix elements from the SENSOR_NAME Full-pol data 
    in the specified folder and save them in the 'C2' directory.
    
    Parameters:
    -----------
    in_dir : str
        The path to the folder containing the SENSOR_NAME data folder.
    
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
    
    # Get file paths  

    hh_file = list(glob.glob(os.path.join(in_dir,'IMG-HH-*-HBQR1.1__A')) + \
        glob.glob(os.path.join(in_dir, 'IMG-HH-*-HBQR1.1__D')))[0]

    hv_file = list(glob.glob(os.path.join(in_dir,'IMG-HV-*-HBQR1.1__A')) + \
        glob.glob(os.path.join(in_dir, 'IMG-HV-*-HBQR1.1__D')))[0]


    vh_file = list(glob.glob(os.path.join(in_dir,'IMG-VH-*-HBQR1.1__A')) + \
        glob.glob(os.path.join(in_dir, 'IMG-VH-*-HBQR1.1__D')))[0]

    vv_file = list(glob.glob(os.path.join(in_dir,'IMG-VV-*-HBQR1.1__A')) + \
        glob.glob(os.path.join(in_dir, 'IMG-VV-*-HBQR1.1__D')))[0]

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


    # Get Absolute calibration constant if required
    calfac_linear = np.sqrt(10 ** ((cf_dB - 32) / 10))
    
    # Read the Four-channel data into numpy arrays
    # Write the single look data; 

    S11 = read_a2(hh_file).astype(np.complex64)*calfac_linear 
    write_a2_rst(os.path.join(base_out_dir, f's11.{ext}'),S11,   driver=driver, mat=mat, cog=cog, ovr=ovr, comp=comp)
    del S11
    S21 = read_a2(hv_file).astype(np.complex64)*calfac_linear 
    S12 = read_a2(vh_file).astype(np.complex64)*calfac_linear 
    
    if recip:
        S12 = (S12 + S21)/2
        S21 = S12
    write_a2_rst(os.path.join(base_out_dir, f's12.{ext}'),S12,   driver=driver, mat=mat, cog=cog, ovr=ovr, comp=comp)
    write_a2_rst(os.path.join(base_out_dir, f's21.{ext}'),S21,  driver=driver, mat=mat, cog=cog, ovr=ovr, comp=comp)
    del S12, S21    
    S22 = read_a2(vv_file).astype(np.complex64)*calfac_linear 
    write_a2_rst(os.path.join(base_out_dir, f's22.{ext}'),S22,  driver=driver, mat=mat, cog=cog, ovr=ovr, comp=comp)
    del S22 
    # if required convert SLC into a multilooked matrices using convert_S function
    if mat not in ['S2', 'Sxy']:
        convert_S(base_out_dir, mat=mat, azlks=azlks, rglks=rglks, cf=1,
                  fmt=fmt, out_dir=final_out_dir, cog=cog, ovr=ovr, comp=comp)

        # Clean up temp directory
        if temp_dir:
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Warning: Could not delete temporary directory {temp_dir}: {e}")
