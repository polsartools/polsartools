import numpy as np
from osgeo import gdal
gdal.UseExceptions()
import os,tempfile,shutil
import xml.etree.ElementTree as ET
from polsartools.utils.utils import time_it
# from polsartools.utils.io_utils import write_T3, write_C3
from polsartools.preprocess.convert_S2 import convert_S

def read_rs2_tif(file):
    ds = gdal.Open(file)
    band1 = ds.GetRasterBand(1).ReadAsArray()
    band2 = ds.GetRasterBand(2).ReadAsArray()
    ds=None
    return np.dstack((band1,band2))

def write_rst(out_file,data,
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
def import_rs2_fp(in_dir,mat='T3',
           azlks=8,rglks=2,fmt='tif',
            cog=False,ovr = [2, 4, 8, 16],comp=False,
           bsc='sigma0', out_dir = None,
           recip=False,
           ):
    """
    Process radarsat-2 image data and generate the specified matrix ('S2', 'C4', 'C3', 'T4', 'T3', 'C2HX', 'C2VX', 'C2HV', 'T2HV') from the input imagery files.

    This function reads radarsat-2 image data in the form of .tif files (HH, HV, VH, VV) from the input folder (`in_dir`) 
    and calculates either the 'S2', 'C4', 'C3', 'T4', 'T3', 'C2HX', 'C2VX', 'C2HV', 'T2HV' matrix. The resulting matrix is then saved in a corresponding directory. The function uses lookup tables (`lutSigma.xml`, `lutGamma.xml`, `lutBeta.xml`) for scaling 
    the data based on the chosen backscatter coefficient `bsc` (sigma0, gamma0, or beta0). The processed data is written into binary files 
    in the output folder.

    Example Usage:
    --------------
    To process imagery and generate a T3 matrix:
    
    .. code-block:: python

        import_rs2_fp("/path/to/data", mat="T3", bsc="sigma0")

    To process imagery and generate a C3 matrix:

    .. code-block:: python

        import_rs2_fp("/path/to/data", mat="C3", bsc="beta0", azlks=10, rglks=3)
        
    Parameters:
    -----------
    in_dir : str
        Path to the folder containing the Radarsat-2 files and the lookup tables (`lutSigma.xml`, `lutGamma.xml`, `lutBeta.xml`).
    
    mat : str, optional (default='T3')
        Type of matrix to extract. Valid options: 'S2',  'C4, 'C3', 'T4', 
        'T3', 'C2HX', 'C2VX', 'C2HV','T2HV'
    
    azlks : int, optional (default=8)
        The number of azimuth looks to apply during the C/T processing.

    rglks : int, optional (default=2)
        The number of range looks to apply during the C/Tprocessing.
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

    bsc : str, optional (default='sigma0')
        The type of radar cross-section to use for scaling. Available options:
        
        - 'sigma0' : Uses `lutSigma.xml` for scaling.
        - 'gamma0' : Uses `lutGamma.xml` for scaling.
        - 'beta0' : Uses `lutBeta.xml` for scaling.
        
    out_dir : str or None, optional (default=None)
        Directory to save output files. If None, a folder named after the matrix type will be created
        in the same location as the input file.
        
    recip : bool, optional (default=False)
        If True, scattering matrix reciprocal symmetry is applied, i.e, S_HV = S_VH.
                
    """
    valid_full_pol = ['S2', 'C4', 'C3', 'T4', 'T3', 'C2HX', 'C2VX', 'C2HV', 'T2HV']
    # valid_dual_pol = ['Sxy', 'C2', 'T2']
    valid_matrices = valid_full_pol
    
    if mat not in valid_matrices:
        raise ValueError(f"Invalid matrix type '{mat}'. \n Supported types are:\n"
                        f"  Full-pol: {sorted(valid_full_pol)}\n")
    
    if bsc == 'sigma0':
        tree = ET.parse(os.path.join(in_dir,"lutSigma.xml"))
        root = tree.getroot()
        lut = root.find('gains').text.strip()
        lut = np.fromstring(lut, sep=' ')
    elif bsc == 'gamma0':
        tree = ET.parse(os.path.join(in_dir,"lutGamma.xml"))
        root = tree.getroot()
        lut = root.find('gains').text.strip()
        lut = np.fromstring(lut, sep=' ')
    elif bsc=='beta0':
        tree = ET.parse(os.path.join(in_dir,"lutBeta.xml"))
        root = tree.getroot()
        lut = root.find('gains').text.strip()
        lut = np.fromstring(lut, sep=' ')
    else:
        raise ValueError(f'Unknown type {bsc} \n Available bsc: sigma0,gamma0,beta0')

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

    inFile = os.path.join(in_dir,"imagery_HH.tif")
    S11 = read_rs2_tif(inFile)
    write_rst(os.path.join(base_out_dir, f's11.{ext}'),
              S11[:,:,0]/lut+1j*(S11[:,:,1]/lut),   
              driver=driver, mat=mat, cog=cog, ovr=ovr, comp=comp)
    
    del S11
    
    inFile = os.path.join(in_dir,"imagery_HV.tif")
    S12 = read_rs2_tif(inFile)

    inFile = os.path.join(in_dir,"imagery_VH.tif")
    S21 = read_rs2_tif(inFile)
    
    if recip:
        S12 = (S12 + S21)/2
        S21 = S12

    write_rst(os.path.join(base_out_dir, f's12.{ext}'),
              S12[:,:,0]/lut+1j*(S12[:,:,1]/lut),   
              driver=driver, mat=mat, cog=cog, ovr=ovr, comp=comp)
    del S12
    write_rst(os.path.join(base_out_dir, f's21.{ext}'),
              S21[:,:,0]/lut+1j*(S21[:,:,1]/lut),   
              driver=driver, mat=mat, cog=cog, ovr=ovr, comp=comp)
    del S21
    
    inFile = os.path.join(in_dir,"imagery_VV.tif")
    S22 = read_rs2_tif(inFile)
    write_rst(os.path.join(base_out_dir, f's22.{ext}'),
              S22[:,:,0]/lut+1j*(S22[:,:,1]/lut),   
              driver=driver, mat=mat, cog=cog, ovr=ovr, comp=comp)
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
