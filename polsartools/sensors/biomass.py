import numpy as np
from osgeo import gdal
gdal.UseExceptions()
import os,tempfile,shutil
import xml.etree.ElementTree as ET
from netCDF4 import Dataset
from scipy.interpolate import LinearNDInterpolator
from polsartools.utils.utils import time_it
# from polsartools.utils.io_utils import write_T3, write_C3
from polsartools.preprocess.convert_S2 import convert_S


def resample_lut(filepath, newrows, newcols,bsc="sigma0"):
    nc_file = Dataset(filepath, mode="r")
    valid_options = ["sigma0", "gamma0"]
    if bsc =="sigma0": 
        lut = nc_file.groups["radiometry"].variables["sigmaNought"][:]
    elif bsc=="gamma0":
        lut = nc_file.groups["radiometry"].variables["gammaNought"][:]
    else:
        raise ValueError(f"Invalid bsc: '{bsc}'. Valid options are: {', '.join(valid_options)}")
    
    lut_array = np.array(lut)

    del lut
    nc_file.close()

    # Original shape
    rows0, cols0 = lut_array.shape

    # Build coordinate grid for original data
    yy, xx = np.mgrid[0:rows0, 0:cols0]
    coord = np.column_stack((xx.ravel(), yy.ravel()))
    values = lut_array.ravel()

    # Create interpolator
    interpfn = LinearNDInterpolator(coord, values)

    # New grid
    fullY, fullX = np.mgrid[0:newrows, 0:newcols]

    # Scale coordinates to original domain
    sigma_resampled = interpfn(fullX * (cols0 / newcols),
                               fullY * (rows0 / newrows))

    return sigma_resampled.astype(np.float32)

def write_rst(out_file, data,
              driver="GTiff", out_dtype=gdal.GDT_CFloat32,
              cog=False, ovr=[2,4,8,16], comp=False,
              geocode=False, ref_ds=None):

    # --- Choose driver and extension ---
    if driver == "ENVI":
        drv = gdal.GetDriverByName(driver)
        dataset = drv.Create(out_file, data.shape[1], data.shape[0], 1, out_dtype)
    else:
        drv = gdal.GetDriverByName("GTiff")
        options = ["BIGTIFF=IF_SAFER"]
        if comp:
            options += ["COMPRESS=LZW"]
        if cog:
            options += ["TILED=YES", "BLOCKXSIZE=512", "BLOCKYSIZE=512"]
        dataset = drv.Create(out_file, data.shape[1], data.shape[0], 1, out_dtype, options)

    # --- Write data ---
    band = dataset.GetRasterBand(1)
    band.WriteArray(data)
    band.SetNoDataValue(0)  # nodata set to 0

    dataset.FlushCache()

    # --- Build overviews if requested ---
    if cog and driver == "GTiff":
        dataset.BuildOverviews("NEAREST", ovr)

    # --- Copy georeferencing and metadata if ref_ds provided ---
    if ref_ds is not None:
        dataset.SetGeoTransform(ref_ds.GetGeoTransform())
        dataset.SetProjection(ref_ds.GetProjection())
        dataset.SetMetadata(ref_ds.GetMetadata())

        gcps = ref_ds.GetGCPs()
        if gcps and len(gcps) > 0:
            dataset.SetGCPs(gcps, ref_ds.GetGCPProjection())

    dataset = None  # close dataset

    # --- Optional geocoding step ---
    if geocode and ref_ds is not None:
        gcps = ref_ds.GetGCPs()
        if gcps and len(gcps) > 0:
            gdal.Warp(out_file, out_file,
                      dstSRS=ref_ds.GetGCPProjection(),
                      dstNodata=0)



def save_l1b(ref_ds, array, band_name, out_dir, common_metadata,
             geocode=False, driver="GTiff", out_dtype=gdal.GDT_Float32,
             cog=False, ovr=[2,4,8,16], comp=False):

    # --- Choose extension based on driver ---
    if driver == "ENVI":
        ext = ".bin"
    else:
        ext = ".tif"

    out_path = os.path.join(out_dir, f"{band_name}{ext}")

    # --- Create dataset with options ---
    if driver == "ENVI":
        drv = gdal.GetDriverByName(driver)
        out_ds = drv.Create(out_path,
                            ref_ds.RasterXSize,
                            ref_ds.RasterYSize,
                            1,
                            out_dtype)
    else:
        drv = gdal.GetDriverByName("GTiff")
        options = ["BIGTIFF=IF_SAFER"]
        if comp:
            options += ["COMPRESS=LZW"]
        if cog:
            options += ["TILED=YES", "BLOCKXSIZE=512", "BLOCKYSIZE=512"]
        out_ds = drv.Create(out_path,
                            ref_ds.RasterXSize,
                            ref_ds.RasterYSize,
                            1,
                            out_dtype,
                            options)

    # --- Copy georeferencing ---
    out_ds.SetGeoTransform(ref_ds.GetGeoTransform())
    out_ds.SetProjection(ref_ds.GetProjection())

    # --- Copy GCPs if present ---
    gcps = ref_ds.GetGCPs()
    if gcps and len(gcps) > 0:
        out_ds.SetGCPs(gcps, ref_ds.GetGCPProjection())

    # --- Add dataset-level metadata ---
    out_ds.SetMetadata(common_metadata)

    # --- Add band-specific metadata ---
    band = out_ds.GetRasterBand(1)
    band.WriteArray(array)
    band.SetDescription(band_name)
    band.SetMetadata({"POLARIMETRIC_INTERP": band_name})

    # --- Set nodata to 0 ---
    band.SetNoDataValue(0)

    out_ds.FlushCache()

    # --- Build overviews if requested ---
    if cog and driver == "GTiff":
        out_ds.BuildOverviews("NEAREST", ovr)

    out_ds = None  # close dataset

    # --- Optional geocoding step ---
    if geocode and gcps and len(gcps) > 0:
        gdal.Warp(out_path, out_path,
                  dstSRS=ref_ds.GetGCPProjection(),
                  dstNodata=0)




@time_it
def import_biomass_l1a(in_dir,mat='T3',
           azlks=8,rglks=2,fmt='tif',
            cog=False,ovr = [2, 4, 8, 16],comp=False,
            bsc='sigma0',
           out_dir = None,
           recip=False,
           geocode=False
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

    bsc : str, optional (default='sigma0')
        The type of radar cross-section to use for scaling. Available options:
        
        - 'sigma0' : Uses `sigmaNought` LUT for scaling.
        - 'gamma0' : Uses `gammaNought` LUT for scaling.

    out_dir : str or None, optional (default=None)
        Directory to save output files. If None, a folder named after the matrix type will be created
        in the same location as the input file.
        
    recip : bool, optional (default=False)
        If True, scattering matrix reciprocal symmetry is applied, i.e, S_HV = S_VH.
                
    """
    
    cf_dB = 39.68531599998061
    calfac_linear = np.sqrt(10 ** ((cf_dB) / 10))

    valid_full_pol = ['S2', 'C4', 'C3', 'T4', 'T3', 'C2HX', 'C2VX', 'C2HV', 'T2HV']
    # valid_dual_pol = ['Sxy', 'C2', 'T2']
    valid_matrices = valid_full_pol
    
    if mat not in valid_matrices:
        raise ValueError(f"Invalid matrix type '{mat}'. \n Supported types are:\n"
                        f"  Full-pol: {sorted(valid_full_pol)}\n")
    
    temp_dir = None
    ext = 'bin' if fmt == 'bin' else 'tif'
    driver = 'ENVI' if fmt == 'bin' else "GTiff"

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

    lut_path = os.path.join(in_dir,"annotation",biomassPath + "_lut.nc")

    # try:
    dataAbsSet = gdal.Open(biomassDataAbsPath, gdal.GA_ReadOnly)
    dataPhaseSet = gdal.Open(biomassDataPhasePath, gdal.GA_ReadOnly)

    if dataAbsSet is None or dataPhaseSet is None:
        raise ValueError("Could not open input GeoTIFFs.")

    bands = dataAbsSet.RasterCount

    print("Extracting single-look elements...")

    s12_data = None
    s21_data = None

    firstAbs = dataAbsSet.GetRasterBand(1).ReadAsArray()
    rows, cols = firstAbs.shape
    lut_g0 = resample_lut(lut_path, rows, cols, bsc)
    del firstAbs

    for idx in range(1, bands + 1):
        bandAbs = dataAbsSet.GetRasterBand(idx).ReadAsArray()
        bandPhase = dataPhaseSet.GetRasterBand(idx).ReadAsArray()

        bandReal = bandAbs * np.cos(bandPhase) #* calfac_linear
        bandImag = bandAbs * np.sin(bandPhase) #* calfac_linear

        # lut_g0 = resample_lut(lut_path, bandReal.shape[0], bandReal.shape[1],bsc)
        bandComplex = (bandReal + 1j*bandImag)*lut_g0

        del bandReal, bandImag

        if driver == "GTiff":
            ext = ".tif"
        else:
            ext = ".bin"

        if idx == 1:
            outFile = os.path.join(base_out_dir, f"s11{ext}")
            write_rst(outFile, bandComplex, driver=driver,
                    out_dtype=gdal.GDT_CFloat32,
                    cog=cog, ovr=ovr, comp=comp,geocode=geocode, ref_ds=dataAbsSet)

        elif idx == 2:
            if recip:
                s12_data = bandComplex  # hold temporarily
            else:
                outFile = os.path.join(base_out_dir, f"s12{ext}")
                write_rst(outFile, bandComplex, driver=driver,
                        out_dtype=gdal.GDT_CFloat32,
                        cog=cog, ovr=ovr, comp=comp,geocode=geocode, ref_ds=dataAbsSet)

        elif idx == 3:
            if recip:
                s21_data = bandComplex  # hold temporarily
            else:
                outFile = os.path.join(base_out_dir, f"s21{ext}")
                write_rst(outFile, bandComplex, driver=driver,
                        out_dtype=gdal.GDT_CFloat32,
                        cog=cog, ovr=ovr, comp=comp,geocode=geocode, ref_ds=dataAbsSet)

        elif idx == 4:
            outFile = os.path.join(base_out_dir, f"s22{ext}")
            write_rst(outFile, bandComplex, driver=driver,
                    out_dtype=gdal.GDT_CFloat32,
                    cog=cog, ovr=ovr, comp=comp,geocode=geocode, ref_ds=dataAbsSet)
    del lut_g0
    # Handle reciprocity only once, after both s12 and s21 are read
    if recip and s12_data is not None and s21_data is not None:
        avg_s12_s21 = (s12_data + s21_data) / 2.0
        for key in ["s12", "s21"]:
            outFile = os.path.join(base_out_dir, f"{key}{ext}")
            write_rst(outFile, avg_s12_s21, driver=driver,
                    out_dtype=gdal.GDT_CFloat32,
                    cog=cog, ovr=ovr, comp=comp,geocode=geocode, ref_ds=dataAbsSet)


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


@time_it
def import_biomass_l1b(in_dir,
            # azlks=1,rglks=1,
            fmt='tif',
            cog=False,ovr = [2, 4, 8, 16],comp=False,
            bsc='sigma0',
           out_dir = None,
           geocode=True,
        #    recip=False,
           ):

    """
    Extracts the backscatter intensity elements from a BIOMASS L2B  '_DGM__' folder and saves them into respective tif/binar files.

    Example Usage:
    --------------
    The following code will extract the intensity elements from a BIOMASS L2B  '_DGM__' folder
    
    .. code-block:: python

        import_biomass_l1b("/path/to/data")

        
    Parameters:
    -----------
    in_dir : str
        Path to the folder containing the BIOMASS L1B files.

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
        
        - 'sigma0' : Uses `sigmaNought` LUT for scaling.
        - 'gamma0' : Uses `gammaNought` LUT for scaling.

    out_dir : str or None, optional (default=None)
        Directory to save output files. If None, a folder named after the matrix type will be created
        in the same location as the input file.

    geocode : bool, default=True
        If True, performs a coarse geocoding using GCPs in the file.
                
    """
    mat='I4'
    cf_dB = 39.68531599998061
    calfac_linear = np.sqrt(10 ** ((cf_dB) / 10))
    
    temp_dir = None
    ext = 'bin' if fmt == 'bin' else 'tif'
    driver = 'ENVI' if fmt == 'bin' else "GTiff"

    # Final output directory
    if out_dir is None:
        final_out_dir = os.path.join(in_dir, mat)
    else:
        final_out_dir = os.path.join(out_dir, mat)
    os.makedirs(final_out_dir, exist_ok=True)

    # Intermediate output directory
    if mat in ['I4']:
        base_out_dir = final_out_dir
    else:
        temp_dir = tempfile.mkdtemp(prefix='temp_S2_')
        base_out_dir = temp_dir

    biomassPath = in_dir.lower()
    biomassPath = biomassPath[-80:]
    biomassPath = biomassPath[:70]
    biomassDataAbsPath = os.path.join(in_dir, "measurement", biomassPath + "_i_abs.tiff")

    lut_path = os.path.join(in_dir,"annotation",biomassPath + "_lut.nc")

    # Open dataset
    ds = gdal.Open(biomassDataAbsPath, gdal.GA_ReadOnly)
    # Extract dataset-level metadata
    common_metadata = ds.GetMetadata()
    if ds is None:
        raise("Could not open:", biomassDataAbsPath)
        
    
    # bsc = "sigma0"

    HH_shape = (ds.RasterYSize, ds.RasterXSize)
    lut_g0 = resample_lut(lut_path, HH_shape[0], HH_shape[1], bsc)
    
    HH = ds.GetRasterBand(1).ReadAsArray().astype("float32")
    HH = HH * lut_g0
    save_l1b(ds, HH, "HH", base_out_dir, common_metadata, driver=driver,cog=cog,ovr = ovr,comp=comp, geocode=geocode)
    del HH

    HV = ds.GetRasterBand(2).ReadAsArray().astype("float32")
    HV = HV * lut_g0
    save_l1b(ds, HV, "HV", base_out_dir, common_metadata, driver=driver,cog=cog,ovr = ovr,comp=comp, geocode=geocode)
    del HV
    
    VH = ds.GetRasterBand(3).ReadAsArray().astype("float32")
    VH = VH * lut_g0
    save_l1b(ds, VH, "VH", base_out_dir, common_metadata, driver=driver,cog=cog,ovr = ovr,comp=comp, geocode=geocode)
    del VH
    
    VV = ds.GetRasterBand(4).ReadAsArray().astype("float32")
    VV = VV * lut_g0
    save_l1b(ds, VV, "VV", base_out_dir, common_metadata, driver=driver,cog=cog,ovr = ovr,comp=comp, geocode=geocode)
    del VV