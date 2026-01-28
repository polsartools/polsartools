import numpy as np
from osgeo import gdal
gdal.UseExceptions()
import os,tempfile,shutil
import xml.etree.ElementTree as ET
from polsartools.utils.utils import time_it
# from polsartools.utils.io_utils import write_T3, write_C3
from polsartools.preprocess.convert_S2 import convert_S

def write_rst(out_file, data,
              driver='GTiff', out_dtype=gdal.GDT_CFloat32,
              mat=None, cog=False, ovr=[2,4,8,16], comp=False):

    if driver == 'ENVI':
        drv = gdal.GetDriverByName(driver)
        dataset = drv.Create(out_file, data.shape[1], data.shape[0], 1, out_dtype)
    else:
        drv = gdal.GetDriverByName("GTiff")
        options = ['BIGTIFF=IF_SAFER']
        if comp:
            options += ['COMPRESS=LZW']
        if cog:
            options += ['TILED=YES', 'BLOCKXSIZE=512', 'BLOCKYSIZE=512']
        dataset = drv.Create(out_file, data.shape[1], data.shape[0], 1, out_dtype, options)

    dataset.GetRasterBand(1).WriteArray(data)
    dataset.FlushCache()

    if cog:
        dataset.BuildOverviews("NEAREST", ovr)

    dataset = None


@time_it
def import_biomass_l1a(in_dir,mat='T3',
           azlks=8,rglks=2,fmt='tif',
            cog=False,ovr = [2, 4, 8, 16],comp=False,
           out_dir = None,
           recip=False,
           ):

    """
    Extract polarimetric S2/T4/C4/T3/C3 matrix data from BIOMASS L1A SLC data.

    Example Usage:
    --------------
    To process imagery and generate a T3 matrix:
    
    .. code-block:: python

        import_biomass_l1a("/path/to/data", mat="T3")

    To process imagery and generate a C3 matrix:

    .. code-block:: python

        import_biomass_l1a("/path/to/data", mat="C3", azlks=10, rglks=3)
        
    Parameters:
    -----------
    in_dir : str
        Path to the folder containing the BIOMASS L1A files.
    
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
        
    out_dir : str or None, optional (default=None)
        Directory to save output files. If None, a folder named after the matrix type will be created
        in the same location as the input file.
        
    recip : bool, optional (default=False)
        If True, scattering matrix reciprocal symmetry is applied, i.e, S_HV = S_VH.
                
    """
    bsc='sigma0'
    valid_full_pol = ['S2', 'C4', 'C3', 'T4', 'T3', 'C2HX', 'C2VX', 'C2HV', 'T2HV']
    # valid_dual_pol = ['Sxy', 'C2', 'T2']
    valid_matrices = valid_full_pol
    
    if mat not in valid_matrices:
        raise ValueError(f"Invalid matrix type '{mat}'. \n Supported types are:\n"
                        f"  Full-pol: {sorted(valid_full_pol)}\n")
    
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

    mat_tag = 'S2'
    biomassPath = in_dir.lower()
    biomassPath = biomassPath[-80:]
    biomassPath = biomassPath[:70]
    biomassDataAbsPath = os.path.join(in_dir,"measurement",biomassPath + "_i_abs.tiff")
    biomassDataPhasePath = os.path.join(in_dir,"measurement",biomassPath + "_i_phase.tiff")

    # try:
    dataAbsSet = gdal.Open(biomassDataAbsPath, gdal.GA_ReadOnly)
    dataPhaseSet = gdal.Open(biomassDataPhasePath, gdal.GA_ReadOnly)

    if dataAbsSet is None or dataPhaseSet is None:
        raise ValueError("Could not open input GeoTIFFs.")

    bands = dataAbsSet.RasterCount

    print("Extracting single-look elements...")

    s12_data = None
    s21_data = None

    for idx in range(1, bands + 1):
        bandAbs = dataAbsSet.GetRasterBand(idx).ReadAsArray()
        bandPhase = dataPhaseSet.GetRasterBand(idx).ReadAsArray()

        bandReal = bandAbs * np.cos(bandPhase)
        bandImag = bandAbs * np.sin(bandPhase)
        bandComplex = bandReal + 1j*bandImag

        del bandReal, bandImag

        if driver == "GTiff":
            ext = ".tif"
        else:
            ext = ".bin"

        if idx == 1:
            outFile = os.path.join(base_out_dir, f"s11{ext}")
            write_rst(outFile, bandComplex, driver=driver,
                    out_dtype=gdal.GDT_CFloat32, mat=mat_tag if mat is None else mat,
                    cog=cog, ovr=ovr, comp=comp)

        elif idx == 2:
            if recip:
                s12_data = bandComplex  # hold temporarily
            else:
                outFile = os.path.join(base_out_dir, f"s12{ext}")
                write_rst(outFile, bandComplex, driver=driver,
                        out_dtype=gdal.GDT_CFloat32, mat=mat_tag if mat is None else mat,
                        cog=cog, ovr=ovr, comp=comp)

        elif idx == 3:
            if recip:
                s21_data = bandComplex  # hold temporarily
            else:
                outFile = os.path.join(base_out_dir, f"s21{ext}")
                write_rst(outFile, bandComplex, driver=driver,
                        out_dtype=gdal.GDT_CFloat32, mat=mat_tag if mat is None else mat,
                        cog=cog, ovr=ovr, comp=comp)

        elif idx == 4:
            outFile = os.path.join(base_out_dir, f"s22{ext}")
            write_rst(outFile, bandComplex, driver=driver,
                    out_dtype=gdal.GDT_CFloat32, mat=mat_tag if mat is None else mat,
                    cog=cog, ovr=ovr, comp=comp)

    # Handle reciprocity only once, after both s12 and s21 are read
    if recip and s12_data is not None and s21_data is not None:
        avg_s12_s21 = (s12_data + s21_data) / 2.0
        for key in ["s12", "s21"]:
            outFile = os.path.join(base_out_dir, f"{key}{ext}")
            write_rst(outFile, avg_s12_s21, driver=driver,
                    out_dtype=gdal.GDT_CFloat32, mat=mat_tag if mat is None else mat,
                    cog=cog, ovr=ovr, comp=comp)


    # except Exception as e:
    #     print(f"An error occurred: {e}")
    
    # Matrix conversion if needed
    if mat not in ['S2', 'Sxy']:
        print("Extracting multi-look elements...")
        convert_S(base_out_dir, mat=mat, azlks=azlks, rglks=rglks, cf=1,
                  fmt=fmt, out_dir=final_out_dir, cog=cog, ovr=ovr, comp=comp)

        # Clean up temp directory
        if temp_dir:
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Warning: Could not delete temporary directory {temp_dir}: {e}")
