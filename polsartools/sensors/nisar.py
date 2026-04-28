
import numpy as np
from osgeo import gdal,osr
gdal.UseExceptions()
import os,tempfile
import tables
# from skimage.util.shape import view_as_blocks
from polsartools.utils.utils import time_it
# from polsartools.utils.io_utils import write_s2_bin_ref, write_s2_ct_ref
from polsartools.utils.h5_utils import h5_polsar, get_ml_chunk
from netCDF4 import Dataset
#%%

def rslc_meta(inFile):
    band_table = [
        ('/science/LSAR', 'L'),
        ('/science/SSAR', 'S')
    ]

    # Identify frequency band and root path
    try:
        with tables.open_file(inFile, mode="r") as h5:
            for path, band in band_table:
                if path in h5:
                    freq_band = band
                    freq_path = path
                    break
            else:
                print("Neither LSAR nor SSAR data found in the file.")
                return None

            # Read polarization list
            listOfPolarizations = None
            for base in ['RSLC', 'SLC']:
                pol_path = f'{freq_path}/{base}/swaths/frequencyA/listOfPolarizations'
                try:
                    listOfPolarizations = np.array(h5.get_node(pol_path).read()).astype(str)
                except tables.NoSuchNodeError:
                    # print(f"Polarization node missing: {pol_path}")
                    continue
                
            # pol_path = f'{freq_path}/RSLC/swaths/frequencyA/listOfPolarizations'
            # try:
            #     listOfPolarizations = np.array(h5.get_node(pol_path).read()).astype(str)
            # except tables.NoSuchNodeError:
            #     print(f"Polarization node missing: {pol_path}")
            #     listOfPolarizations = None

    except Exception:
        raise RuntimeError("Invalid .h5 file !!")

    return freq_band,listOfPolarizations

    
def get_rslc_path(h5, freq_band):
    for product_type in ['RSLC', 'SLC']:
        base_path = f'/science/{freq_band}SAR/{product_type}/swaths/frequencyA'
        if h5.__contains__(base_path):
            return base_path
        else:
            print(f"Base path not found: {base_path}")
    return None


def gslc_meta(inFile):
    band_table = [
        ('/science/LSAR', 'L'),
        ('/science/SSAR', 'S')
    ]

    # Identify frequency band and root path
    try:
        with tables.open_file(inFile, mode="r") as h5:
            for path, band in band_table:
                if path in h5:
                    freq_band = band
                    freq_path = path
                    break
            else:
                print("Neither LSAR nor SSAR data found in the file.")
                return None

            # Read polarization list
            pol_path = f'{freq_path}/GSLC/grids/frequencyA/listOfPolarizations'
            try:
                listOfPolarizations = np.array(h5.get_node(pol_path).read()).astype(str)
            except tables.NoSuchNodeError:
                print(f"Polarization node missing: {pol_path}")
                listOfPolarizations = None

    except Exception:
        raise RuntimeError("Invalid .h5 file !!")

    # Reopen to read raster metadata
    with tables.open_file(inFile, "r") as h5:
        base_grid = f'{freq_path}/GSLC/grids/frequencyA'
        projection_path = f'{freq_path}/GSLC/metadata/radarGrid/projection'

        try:
            projection = np.array(h5.get_node(projection_path).read())
            xSpacing = np.array(h5.get_node(base_grid + '/xCoordinateSpacing').read())
            ySpacing = np.array(h5.get_node(base_grid + '/yCoordinateSpacing').read())
        except tables.NoSuchNodeError as e:
            print(f"Missing expected metadata node: {e}")
            return None

    return freq_band,listOfPolarizations, xSpacing, ySpacing, int(projection)

def gcov_meta(inFile):
    band_table = [
        ('/science/LSAR', 'L'),
        ('/science/SSAR', 'S')
    ]

    # Identify frequency band and root path
    try:
        with tables.open_file(inFile, mode="r") as h5:
            for path, band in band_table:
                if path in h5:
                    freq_band = band
                    freq_path = path
                    break
            else:
                print("Neither LSAR nor SSAR data found in the file.")
                return None

            # Read polarization list
            pol_path = f'{freq_path}/GCOV/grids/frequencyA/listOfPolarizations'
            try:
                listOfPolarizations = np.array(h5.get_node(pol_path).read()).astype(str)
            except tables.NoSuchNodeError:
                print(f"Polarization node missing: {pol_path}")
                listOfPolarizations = None

    except Exception:
        raise RuntimeError("Invalid .h5 file !!")

    # Reopen to read raster metadata
    with tables.open_file(inFile, "r") as h5:
        base_grid = f'{freq_path}/GCOV/grids/frequencyA'
        projection_path = f'{freq_path}/GCOV/grids/frequencyA/projection'

        try:
            projection = np.array(h5.get_node(projection_path).read())
            xSpacing = np.array(h5.get_node(base_grid + '/xCoordinateSpacing').read())
            ySpacing = np.array(h5.get_node(base_grid + '/yCoordinateSpacing').read())
        except tables.NoSuchNodeError as e:
            print(f"Missing expected metadata node: {e}")
            return None

    return freq_band,listOfPolarizations, xSpacing, ySpacing, int(projection)

def get_geo_meta(inFile):
    """
    Combined metadata extractor for GSLC and GCOV products.
    """
    band_table = [('/science/LSAR', 'L'), ('/science/SSAR', 'S')]
    
    try:
        with tables.open_file(inFile, mode="r") as h5:
            # Detect frequency band (LSAR or SSAR)
            freq_band, freq_path = None, None
            for path, band in band_table:
                if path in h5:
                    freq_band, freq_path = band, path
                    break
            
            if not freq_band:
                print("Neither LSAR nor SSAR data found.")
                return None

            # Detect Product Type (GSLC or GCOV)
            # We check which group exists under the detected frequency path
            product_type = None
            for p_type in ["GSLC", "GCOV"]:
                if f"{freq_path}/{p_type}" in h5:
                    product_type = p_type
                    break
            
            if not product_type:
                print("Neither GSLC nor GCOV group found.")
                return None

            # 2. Set up paths based on detected product
            base_grid = f'{freq_path}/{product_type}/grids/frequencyA'
            pol_path = f'{base_grid}/listOfPolarizations'
            
            # Note: GCOV usually puts projection in the grid, GSLC in metadata/radarGrid
            if product_type == "GCOV":
                projection_path = f'{base_grid}/projection'
            else:
                projection_path = f'{freq_path}/GSLC/metadata/radarGrid/projection'

            # 3. Read the data
            try:
                # Read Polarization
                pol_node = h5.get_node(pol_path).read()
                list_of_polarizations = np.array(pol_node).astype(str)
                
                # Read Spacing and Projection
                projection = int(h5.get_node(projection_path).read())
                x_spacing = h5.get_node(f"{base_grid}/xCoordinateSpacing").read()
                y_spacing = h5.get_node(f"{base_grid}/yCoordinateSpacing").read()
                
                return freq_band, list_of_polarizations, x_spacing, y_spacing, int(projection)

            except tables.NoSuchNodeError as e:
                print(f"Missing expected metadata node: {e}")
                return None

    except Exception as e:
        print(f"Error: {e}")
        raise RuntimeError("Invalid .h5 file or structure!!")


def nisar_dp(matrix_type, inFile, inFolder, base_path, azlks, rglks, recip, max_workers,
                 start_x, start_y, xres, yres, projection, fmt, cog, ovr, comp,
                 inshape, outshape, listOfPolarizations, out_dir=None,cc=1):

    # Determine matrix type based on available polarizations
    # if matrix_type in ['C2','C2HV','C2HX','C2VX']:
        
    if matrix_type=='Sxy':
        print(f"Extracting {matrix_type} matrix elements...")

        if 'HH' in listOfPolarizations and 'HV' in listOfPolarizations:
            # matrix_type = 'C2HX'
            channels = ['HH', 'HV']
        elif 'VV' in listOfPolarizations and 'VH' in listOfPolarizations:
            # matrix_type = 'C2VX'
            channels = ['VV', 'VH']
        elif 'HH' in listOfPolarizations and 'VV' in listOfPolarizations:
            # matrix_type = 'C2HV'
            channels = ['HH', 'VV']
        elif 'RH' in listOfPolarizations and 'RV' in listOfPolarizations:
            # matrix_type = 'C2HV'
            channels = ['RH', 'RV']
        elif 'LH' in listOfPolarizations and 'LV' in listOfPolarizations:
            # matrix_type = 'C2HV'
            channels = ['LH', 'LV']
        else:
            print("No valid dual-channel polarization combination found.")
            return



        # Directory setup
        base_name = os.path.basename(inFile).split('.h5')[0]
        if out_dir is None:
            out_dir = os.path.join(inFolder, base_name, matrix_type)
        else:
            out_dir = os.path.join(out_dir, matrix_type)
        os.makedirs(out_dir, exist_ok=True)

        temp_dir = tempfile.mkdtemp(prefix=f"{matrix_type}_", dir=out_dir)
        os.makedirs(temp_dir, exist_ok=True)

        # Dataset paths
        dataset_paths = {ch: f"{base_path}/{ch}" for ch in channels}

        # Call h5_polsar
        h5_polsar(
            h5_file=inFile,
            dataset_paths=dataset_paths,
            output_dir=out_dir,
            temp_dir=temp_dir,
            azlks=azlks,
            rglks=rglks,
            matrix_type=matrix_type,
            apply_multilook=False,
            recip=recip,
            chunk_size_x=get_ml_chunk(rglks, 512),
            chunk_size_y=get_ml_chunk(azlks, 512),
            max_workers=max_workers,
            start_x=start_x, start_y=start_y,
            xres=xres, yres=yres,
            epsg=int(projection),
            fmt=fmt, cog=cog, ovr=ovr, comp=comp,
            dtype=np.complex64,
            inshape=inshape,
            outshape=outshape,
            calibration_constant=cc
        )        
    else:
        if 'HH' in listOfPolarizations and 'HV' in listOfPolarizations:
            matrix_type = 'C2HX'
            channels = ['HH', 'HV']
        elif 'VV' in listOfPolarizations and 'VH' in listOfPolarizations:
            matrix_type = 'C2VX'
            channels = ['VV', 'VH']
        elif 'HH' in listOfPolarizations and 'VV' in listOfPolarizations:
            matrix_type = 'C2HV'
            channels = ['HH', 'VV']
        elif 'RH' in listOfPolarizations and 'RV' in listOfPolarizations:
            matrix_type = 'C2R'
            channels = ['RH', 'RV']
        elif 'LH' in listOfPolarizations and 'LV' in listOfPolarizations:
            matrix_type = 'C2L'
            channels = ['LH', 'LV']
        else:
            print("No valid dual-channel polarization combination found.")
            return

        print(f"Extracting {matrix_type} matrix elements...")

        # Directory setup
        base_name = os.path.basename(inFile).split('.h5')[0]
        if out_dir is None:
            out_dir = os.path.join(inFolder, base_name, matrix_type)
        else:
            out_dir = os.path.join(out_dir, matrix_type)
        os.makedirs(out_dir, exist_ok=True)

        temp_dir = tempfile.mkdtemp(prefix=f"{matrix_type}_", dir=out_dir)
        os.makedirs(temp_dir, exist_ok=True)

        # Dataset paths
        dataset_paths = {ch: f"{base_path}/{ch}" for ch in channels}

        # Call h5_polsar
        h5_polsar(
            h5_file=inFile,
            dataset_paths=dataset_paths,
            output_dir=out_dir,
            temp_dir=temp_dir,
            azlks=azlks,
            rglks=rglks,
            matrix_type=matrix_type,
            apply_multilook=True,
            recip=recip,
            chunk_size_x=get_ml_chunk(rglks, 512),
            chunk_size_y=get_ml_chunk(azlks, 512),
            max_workers=max_workers,
            start_x=start_x, start_y=start_y,
            xres=xres, yres=yres,
            epsg=int(projection),
            fmt=fmt, cog=cog, ovr=ovr, comp=comp,
            dtype=np.float32,
            inshape=inshape,
            outshape=outshape,
            calibration_constant=cc
        )


def nisar_fp(mat, inFile, inFolder, base_path, azlks, rglks, recip, max_workers,
                   start_x, start_y, xres, yres, projection, fmt, cog, ovr, comp,
                   inshape, outshape, out_dir=None,cc=1):

    MATRIX_CONFIG = {
        'S2':   {'channels': ['HH', 'HV', 'VH', 'VV'], 'apply_multilook': False, 'dtype': np.complex64},
        'T4':   {'channels': ['HH', 'HV', 'VH', 'VV'], 'apply_multilook': True,  'dtype': np.float32},
        'T3':   {'channels': ['HH', 'HV', 'VH', 'VV'], 'apply_multilook': True,  'dtype': np.float32},
        'C4':   {'channels': ['HH', 'HV', 'VH', 'VV'], 'apply_multilook': True,  'dtype': np.float32},
        'C3':   {'channels': ['HH', 'HV', 'VH', 'VV'], 'apply_multilook': True,  'dtype': np.float32},
        'C2HV': {'channels': ['HH', 'VV'],             'apply_multilook': True,  'dtype': np.float32},
        'C2HX': {'channels': ['HH', 'HV'],             'apply_multilook': True,  'dtype': np.float32},
        'C2VX': {'channels': ['VV', 'VH'],             'apply_multilook': True,  'dtype': np.float32},
        'T2HV': {'channels': ['HH', 'VV'],             'apply_multilook': True,  'dtype': np.float32},
        'C2L':  {'channels': ['LH', 'LV'],             'apply_multilook': True,  'dtype': np.float32},
        'C2R':  {'channels': ['RH', 'RV'],             'apply_multilook': True,  'dtype': np.float32},
    }


    if mat not in MATRIX_CONFIG:
        raise ValueError(f"Unsupported matrix type: {mat}. Choose from {', '.join(MATRIX_CONFIG.keys())}")

    print(f"Extracting {mat} matrix elements...")

    # Directory setup
    base_name = os.path.basename(inFile).split('.h5')[0]
    if out_dir is None:
        out_dir = os.path.join(inFolder, base_name, mat)
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = os.path.join(out_dir, mat)
        os.makedirs(out_dir, exist_ok=True)

    # Use tempfile for temp_dir
    temp_dir = tempfile.mkdtemp(prefix=f"{mat}_", dir=out_dir)
    os.makedirs(temp_dir, exist_ok=True)

    # Dataset paths
    channels = MATRIX_CONFIG[mat]['channels']
    dataset_paths = {ch: f"{base_path}/{ch}" for ch in channels}

    # Call h5_polsar
    h5_polsar(
        h5_file=inFile,
        dataset_paths=dataset_paths,
        output_dir=out_dir,
        temp_dir=temp_dir,
        azlks=azlks,
        rglks=rglks,
        matrix_type=mat,
        apply_multilook=MATRIX_CONFIG[mat]['apply_multilook'],
        recip=recip,
        chunk_size_x=get_ml_chunk(rglks, 512),
        chunk_size_y=get_ml_chunk(azlks, 512),
        max_workers=max_workers,
        start_x=start_x, start_y=start_y,
        xres=xres, yres=yres,
        epsg=int(projection),
        fmt=fmt, cog=cog, ovr=ovr, comp=comp,
        dtype=MATRIX_CONFIG[mat]['dtype'],
        inshape=inshape,
        outshape=outshape,
        calibration_constant=cc
    )


tables.parameters.PYTABLES_SYS_ATTRS = False 
def get_geo_info(inFile, azlks, rglks):

    with tables.open_file(inFile, "r") as h5:
        try:
            # Access the 'science' group
            science_grp = h5.root.science
            
            # Detect SAR band (e.g., 'SSAR' or 'LSAR')
            sar_key = [node._v_name for node in science_grp if "SAR" in node._v_name][0]
            sar_node = h5.get_node(f"/science/{sar_key}")
            
            # Detect Product Type ('GSLC' or 'GCOV')
            product_type = [node._v_name for node in sar_node if node._v_name in ["GSLC", "GCOV"]][0]
            
            # Construct path and fetch coordinates
            grid_base = f"/science/{sar_key}/{product_type}/grids/frequencyA"
            
            xcoords = h5.get_node(f"{grid_base}/xCoordinates").read()
            ycoords = h5.get_node(f"{grid_base}/yCoordinates").read()
        
            # Calculate Shapes and Extents
            # Note: PyTables/HDF5 usually follows [y, x] (Row, Col) convention
            # inshape = [len(ycoords), len(xcoords)] 
            # outshape = [len(ycoords) // azlks, len(xcoords) // rglks]
            inshape = [len(xcoords), len(ycoords)]  
            outshape = [len(xcoords) // rglks, len(ycoords) // azlks]
            
            # Standard Geo-referencing: min X (West) and max Y (North)
            start_x = np.min(xcoords)
            start_y = np.max(ycoords)
            
            # Optional: Print detection for debugging
            # print(f"Detected: {sar_key} | {product_type}")
            
            return inshape, outshape, start_x, start_y, xcoords, ycoords

        except IndexError:
            raise KeyError(f"Could not find valid SAR band or GSLC/GCOV group in {inFile}")
        except Exception as e:
            print(f"Error processing HDF5: {e}")
            raise

@time_it 
def import_nisar_gslc(inFile, mat='T3', azlks=2, rglks=2, fmt='tif',
             cog=False,ovr = [2, 4, 8, 16],comp=False,
             out_dir=None,
             recip=False,
            max_workers=None):
    """
    Extracts the C2 matrix elements (C11, C22, and C12) from a NISAR GSLC HDF5 file 
    and saves them into respective binary files.

    Example:
    --------
    >>> import_nisar_gslc("path_to_file.h5", azlks=30, rglks=15)
    This will extract the C2 matrix elements from the dual-pol NISAR GSLC file 
    and save them in the 'C2' folder. or for full-pol 'T3'
    
    Parameters:
    -----------
    inFile : str
        The path to the NISAR GSLC HDF5 file containing the radar data.

    mat : str, optional (default = 'C2' for Dual-pol, 'T3' for Full-pol)
        Type of matrix to extract. Valid options for Full-pol: 'S2',  'C4, 'C3', 'T4', 
        'T3', 'C2HX', 'C2VX', 'C2HV','T2HV'and Dual-pol: 'Sxy','C2'.

    azlks : int, optional (default=3)
        The number of azimuth looks for multi-looking. 

    rglks : int, optional (default=3)
        The number of range looks for multi-looking. 
    
    fmt : {'tif', 'bin'}, optional (default='tif')
        Output format:
        - 'tif': GeoTIFF with georeferencing
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
        
    
    # freq_band,listOfPolarizations, xres, yres, projection = gslc_meta(inFile)
    freq_band,listOfPolarizations, xres, yres, projection = get_geo_meta(inFile)
    nchannels = len(listOfPolarizations)
    print(f"Detected {freq_band}-band polarization channels: {listOfPolarizations}")

    inshape, outshape, start_x, start_y, xcoords,ycoords = get_geo_info(inFile, azlks, rglks)
    
    # print(projection,start_x,start_y,xres,yres,inshape,outshape)
    # print(min(ycoords),max(ycoords), min(ycoords)+np.abs(yres)*(inshape[1]-1))
    
    inFolder = os.path.dirname(inFile)   
    if not inFolder:
        inFolder = "."
    
    base_path = f'/science/{freq_band}SAR/GSLC/grids/frequencyA'
    
    if nchannels==2:
        # print("Dual-Pol data detected.",mat)
        nisar_dp(mat,inFile, inFolder, base_path, azlks, rglks, recip, max_workers,
                 start_x, start_y, xres, yres, projection, fmt, cog, ovr, comp,
                 inshape, outshape, listOfPolarizations, out_dir)
        
                
    elif nchannels==4:
        nisar_fp(mat, inFile, inFolder, base_path, azlks, rglks, recip, max_workers,
        start_x, start_y, xres, yres, projection, fmt, cog, ovr, comp,
        inshape, outshape, out_dir)
        

@time_it  
def import_nisar_rslc(inFile, mat='T3', azlks=22,rglks=10, 
               fmt='tif', cog=False, ovr = [2, 4, 8, 16], comp=False,
              out_dir=None,
              recip=False,
               max_workers=None ):
    """
    Extracts the C2 (for dual-pol), S2/C3/T3 (for full-pol) matrix elements from a NISAR RSLC HDF5 file 
    and saves them into respective binary files.
    
    Example:
    --------
    >>> import_nisar_rslc("path_to_file.h5", azlks=30, rglks=15)
    This will extract the C2 (for dual-pol), S2/C3/T3 (for full-pol) matrix elements from the specified NISAR RSLC file 
    and save them in the respective folders.
    
    Parameters:
    -----------
    inFile : str
        The path to the NISAR RSLC HDF5 file containing the radar data.

    mat : str, optional (default = 'T3' or 'C2)
        Type of matrix to extract. Valid options for Full-pol: 'S2',  'C4, 'C3', 'T4', 
        'T3', 'C2HX', 'C2VX', 'C2HV','T2HV'and Dual-pol: 'Sxy','C2'.

    azlks : int, optional (default=3)
        The number of azimuth looks for multi-looking. 

    rglks : int, optional (default=3)
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


    """
    
    freq_band,listOfPolarizations = rslc_meta(inFile)
    nchannels = len(listOfPolarizations)

    print(f"Detected {freq_band}-band {listOfPolarizations} ")
    
    inFolder = os.path.dirname(inFile)   
    if not inFolder:
        inFolder = "."
        
    with tables.open_file(inFile, mode="r") as h5:
        base_path = get_rslc_path(h5, freq_band)
        h5.close()
    start_x = 0
    start_y = 0
    xres = 1
    yres = 1
    projection = 4326
    if nchannels==2:    
        nisar_dp(mat,inFile, inFolder, base_path, azlks, rglks, recip, max_workers,
            start_x, start_y, xres, yres, projection, fmt, cog, ovr, comp,
            None, None, listOfPolarizations, out_dir)   
        

    elif nchannels==4:
        nisar_fp(mat, inFile, inFolder, base_path, azlks, rglks, recip, max_workers,
        start_x, start_y, xres, yres, projection, fmt, cog, ovr, comp,
        None, None, out_dir)





def nisar_gcov(matrix_type, inFile, inFolder, base_path, azlks, rglks, max_workers,
                 start_x, start_y, xres, yres, projection, fmt, cog, ovr, comp,
                 inshape, outshape, listOfPolarizations, out_dir=None,cc=1):

    print(f"Extracting elements...")
    if len(listOfPolarizations)==2 or len(listOfPolarizations)==3:
        if 'HH' in listOfPolarizations and 'HV' in listOfPolarizations:
            # matrix_type = 'C2HX'
            channels = ['HHHH', 'HVHV']
        elif 'VV' in listOfPolarizations and 'VH' in listOfPolarizations:
            # matrix_type = 'C2VX'
            channels = ['VVVV', 'VHVH']
        elif 'HH' in listOfPolarizations and 'VV' in listOfPolarizations:
            # matrix_type = 'C2HV'
            channels = ['HHHH', 'VVVV']
        elif 'RH' in listOfPolarizations and 'RV' in listOfPolarizations:
            # matrix_type = 'C2HV'
            channels = ['RHRH', 'RVRV']
        else:
            print("No valid dual-channel polarization combination found.")
            return
    elif len(listOfPolarizations)==4:
        channels = ['HHHH', 'HVHV','VVVV','VHVH']

    # Directory setup
    base_name = os.path.basename(inFile).split('.h5')[0]
    if out_dir is None:
        out_dir = os.path.join(inFolder, base_name, matrix_type)
    else:
        out_dir = os.path.join(out_dir, matrix_type)
    os.makedirs(out_dir, exist_ok=True)

    temp_dir = tempfile.mkdtemp(prefix=f"{matrix_type}_", dir=out_dir)
    os.makedirs(temp_dir, exist_ok=True)

    # Dataset paths
    dataset_paths = {ch: f"{base_path}/{ch}" for ch in channels}
    recip = False
    # Call h5_polsar
    h5_polsar(
        h5_file=inFile,
        dataset_paths=dataset_paths,
        output_dir=out_dir,
        temp_dir=temp_dir,
        azlks=azlks,
        rglks=rglks,
        matrix_type=matrix_type,
        apply_multilook=True,
        recip=recip,
        chunk_size_x=get_ml_chunk(rglks, 512),
        chunk_size_y=get_ml_chunk(azlks, 512),
        max_workers=max_workers,
        start_x=start_x, start_y=start_y,
        xres=xres, yres=yres,
        epsg=int(projection),
        fmt=fmt, cog=cog, ovr=ovr, comp=comp,
        dtype=np.float32,
        inshape=inshape,
        outshape=outshape,
        calibration_constant=cc
    )        
    
@time_it 
def import_nisar_gcov(inFile, azlks=1, rglks=1, fmt='tif',
             cog=False,ovr = [2, 4, 8, 16],comp=False,
             out_dir=None,
            max_workers=None):
    """
    Extracts the backscatter intensity elements from a NISAR GCOV HDF5 file and saves them into respective tif/binar files.

    Example:
    --------
    >>> import_nisar_gcov("path_to_file.h5")
    This will extract the intensity elements from NISAR GCOV HDF5 file without multi-looking

    >>> import_nisar_gcov("path_to_file.h5", azlks=3, rglks=3)
    This will extract the intensity elements from NISAR GCOV HDF5 file with 3x3 multi-looking.
    
    Parameters:
    -----------
    inFile : str
        The path to the NISAR GCOV HDF5 file containing the radar data.

    azlks : int, optional (default=1)
        The number of azimuth looks for multi-looking. 

    rglks : int, optional (default=1)
        The number of range looks for multi-looking. 
    
    fmt : {'tif', 'bin'}, optional (default='tif')
        Output format:
        - 'tif': GeoTIFF with georeferencing
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

    """
      
    # freq_band,listOfPolarizations, xres, yres, projection = gcov_meta(inFile)
    freq_band,listOfPolarizations, xres, yres, projection = get_geo_meta(inFile)
    nchannels = len(listOfPolarizations)
    print(f"Detected {freq_band}-band polarization channels: {listOfPolarizations}")


    inshape, outshape, start_x, start_y, xcoords,ycoords = get_geo_info(inFile, azlks, rglks)
    
    inFolder = os.path.dirname(inFile)   
    if not inFolder:
        inFolder = "."
    
    base_path = f'/science/{freq_band}SAR/GCOV/grids/frequencyA'

    if nchannels==2:
        mat='I2'
    elif nchannels==4:
        mat='I4'
    else:
        raise('Invalid number of channels!!')
    nisar_gcov(mat,inFile, inFolder, base_path, azlks, rglks, max_workers,
                start_x, start_y, xres, yres, projection, fmt, cog, ovr, comp,
                inshape, outshape, listOfPolarizations, out_dir)
