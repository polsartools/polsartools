
import numpy as np
from osgeo import gdal,osr
gdal.UseExceptions()
import os,tempfile
import tables
from skimage.util.shape import view_as_blocks
from polsartools.utils.utils import time_it, mlook_arr
from polsartools.utils.io_utils import write_s2_bin_ref, write_s2_ct_ref
from polsartools.utils.h5_utils import h5_polsar, get_ml_chunk
from netCDF4 import Dataset
#%%

# def get_ml_chunk(multiplier, default_size):
#     # Rounds up to the next multiple of `multiplier`
#     return ((default_size + multiplier - 1) // multiplier) * multiplier

def rslc_meta(inFile):
    band_table = [
        ('/science/LSAR', 'L'),
        ('/science/SSAR', 'S')
    ]

    # Step 1: Identify frequency band and root path
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

            # Step 2: Read polarization list
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

    # Step 1: Identify frequency band and root path
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

            # Step 2: Read polarization list
            pol_path = f'{freq_path}/GSLC/grids/frequencyA/listOfPolarizations'
            try:
                listOfPolarizations = np.array(h5.get_node(pol_path).read()).astype(str)
            except tables.NoSuchNodeError:
                print(f"Polarization node missing: {pol_path}")
                listOfPolarizations = None

    except Exception:
        raise RuntimeError("Invalid .h5 file !!")

    # Step 3: Reopen to read raster metadata
    with tables.open_file(inFile, "r") as h5:
        base_grid = f'{freq_path}/GSLC/grids/frequencyA'
        projection_path = f'{freq_path}/GSLC/metadata/radarGrid/projection'

        try:
            projection = np.array(h5.get_node(projection_path).read())
            xSpacing = np.array(h5.get_node(base_grid + '/xCoordinateSpacing').read())
            ySpacing = np.array(h5.get_node(base_grid + '/yCoordinateSpacing').read())
        except tables.NoSuchNodeError as e:
            print(f"⚠️ Missing expected metadata node: {e}")
            return None

    return freq_band,listOfPolarizations, xSpacing, ySpacing, int(projection)

def gslc_dp(matrix_type, inFile, inFolder, base_path, azlks, rglks, recip, max_workers,
                 start_x, start_y, xres, yres, projection, fmt, cog, ovr, comp,
                 inshape, outshape, listOfPolarizations, out_dir=None):

    # Determine matrix type based on available polarizations
    if matrix_type in ['C2','C2HV','C2HX','C2VX']:
        if 'HH' in listOfPolarizations and 'HV' in listOfPolarizations:
            matrix_type = 'C2HX'
            channels = ['HH', 'HV']
        elif 'VV' in listOfPolarizations and 'VH' in listOfPolarizations:
            matrix_type = 'C2VX'
            channels = ['VV', 'VH']
        elif 'HH' in listOfPolarizations and 'VV' in listOfPolarizations:
            matrix_type = 'C2HV'
            channels = ['HH', 'VV']
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
            outshape=outshape
        )
    elif matrix_type=='Sxy':
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
            outshape=outshape
        )        



def gslc_fp(mat, inFile, inFolder, base_path, azlks, rglks, recip, max_workers,
                   start_x, start_y, xres, yres, projection, fmt, cog, ovr, comp,
                   inshape, outshape, out_dir=None):

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
        outshape=outshape
    )
@time_it 
def nisar_gslc(inFile, mat='C3', azlks=2, rglks=2, fmt='tif',
             cog=False,ovr = [2, 4, 8, 16],comp=False,
             out_dir=None,
             recip=False,
            max_workers=None):
    """
    Extracts the C2 matrix elements (C11, C22, and C12) from a NISAR GSLC HDF5 file 
    and saves them into respective binary files.

    Example:
    --------
    >>> nisar_gslc("path_to_file.h5", azlks=30, rglks=15)
    This will extract the C2 matrix elements from the specified NISAR GSLC file 
    and save them in the 'C2' folder.
    
    Parameters:
    -----------
    inFile : str
        The path to the NISAR GSLC HDF5 file containing the radar data.

    azlks : int, optional (default=20)
        The number of azimuth looks for multi-looking. 

    rglks : int, optional (default=10)
        The number of range looks for multi-looking. 

    Returns:
    --------
    None
        The function does not return any value. Instead, it creates a folder 
        named `C2` (if not already present) and saves the following binary files:
        
        - `C11.bin`: Contains the C11 matrix elements.
        - `C22.bin`: Contains the C22 matrix elements.
        - `C12_real.bin`: Contains the real part of the C12 matrix.
        - `C12_imag.bin`: Contains the imaginary part of the C12 matrix.
        - `config.txt`: A text file containing grid dimensions and polarimetric configuration.

    Raises:
    -------
    Exception
        If the GSLC HDF5 file is invalid or cannot be read.


    """
        
    
    freq_band,listOfPolarizations, xres, yres, projection = gslc_meta(inFile)
    nchannels = len(listOfPolarizations)
    print(f"Detected {freq_band}-band polarization channels: {listOfPolarizations}")
    
    ds = Dataset(inFile, "r")
    xcoords = ds.groups["science"].groups[f"{freq_band}SAR"].groups["GSLC"] \
                   .groups["grids"].groups["frequencyA"] \
                   .variables["xCoordinates"][:]
    ycoords = ds.groups["science"].groups[f"{freq_band}SAR"].groups["GSLC"] \
                   .groups["grids"].groups["frequencyA"] \
                   .variables["yCoordinates"][:]                
    ds=None

    inshape = [len(xcoords),len(ycoords)]
    outshape = [len(xcoords)//rglks,len(ycoords)//azlks]
    
    start_x = min(xcoords)
    start_y = max(ycoords)
    
    # print(projection,start_x,start_y,xres,yres,inshape,outshape)
    # print(min(ycoords),max(ycoords), min(ycoords)+np.abs(yres)*(inshape[1]-1))
    
    inFolder = os.path.dirname(inFile)   
    if not inFolder:
        inFolder = "."
    
    base_path = f'/science/{freq_band}SAR/GSLC/grids/frequencyA'
    
    if nchannels==2:
        # print("Dual-Pol data detected.",mat)
        gslc_dp(mat,inFile, inFolder, base_path, azlks, rglks, recip, max_workers,
                 start_x, start_y, xres, yres, projection, fmt, cog, ovr, comp,
                 inshape, outshape, listOfPolarizations, out_dir)
        # out_dir = os.path.join(inFolder,os.path.basename(inFile).split('.h5')[0],'C2')
        # temp_dir = os.path.join(inFolder,os.path.basename(inFile).split('.h5')[0],'C2','temp')
        # print("Extracting C2 matrix elements...")
        # if 'HH' in listOfPolarizations and 'HV' in listOfPolarizations:
        #     h5_polsar(
        #                 h5_file=inFile,
        #                 dataset_paths={
        #                     "HH": f"{base_path}/HH",
        #                     "HV": f"{base_path}/HV",
        #                 },
        #                 output_dir=out_dir,
        #                 temp_dir=temp_dir,
        #                 azlks=azlks,
        #                 rglks=rglks,
        #                 matrix_type = 'C2HX',
        #                 apply_multilook=True,
        #                 chunk_size_x=get_ml_chunk(rglks, 512),
        #                 chunk_size_y=get_ml_chunk(azlks, 512),
        #                 max_workers=max_workers,
        #                 start_x=start_x, start_y=start_y, xres=xres, yres=yres, epsg=projection,
        #                 fmt=fmt,cog=cog,ovr =ovr,comp=comp,
        #                 dtype = np.float32,
        #                 inshape=inshape,
        #                 outshape=outshape
        #             )
    

        # elif 'VV' in listOfPolarizations and 'VH' in listOfPolarizations:
        #     h5_polsar(
        #                 h5_file=inFile,
        #                 dataset_paths={
        #                     "VV": f"{base_path}/VV",
        #                     "VH": f"{base_path}/VH",
        #                 },
        #                 output_dir=out_dir,
        #                 temp_dir=temp_dir,
        #                 azlks=azlks,
        #                 rglks=rglks,
        #                 matrix_type = 'C2VX',
        #                 apply_multilook=True,
        #                 chunk_size_x=get_ml_chunk(rglks, 512),
        #                 chunk_size_y=get_ml_chunk(azlks, 512),
        #                 max_workers=max_workers,
        #                 start_x=start_x, start_y=start_y, xres=xres, yres=yres, epsg=projection,
        #                 fmt=fmt,cog=cog,ovr =ovr,comp=comp,
        #                 dtype = np.float32,
        #                 inshape=inshape,
        #                 outshape=outshape
        #             )

        # elif 'HH' in listOfPolarizations and 'VV' in listOfPolarizations:
        #     h5_polsar(
        #                 h5_file=inFile,
        #                 dataset_paths={
        #                     "HH": f"{base_path}/HH",
        #                     "VV": f"{base_path}/VV",
        #                 },
        #                 output_dir=out_dir,
        #                 temp_dir=temp_dir,
        #                 azlks=azlks,
        #                 rglks=rglks,
        #                 matrix_type = 'C2HV',
        #                 apply_multilook=True,
        #                 chunk_size_x=get_ml_chunk(rglks, 512),
        #                 chunk_size_y=get_ml_chunk(azlks, 512),
        #                 max_workers=max_workers,
        #                 start_x=start_x, start_y=start_y, xres=xres, yres=yres, epsg=projection,
        #                 fmt=fmt,cog=cog,ovr =ovr,comp=comp,
        #                 dtype = np.float32,
        #                 inshape=inshape,
        #                 outshape=outshape
        #             )

        # else:
        #     print("No HH, HV, VV, or VH polarizations found in the file.")

        #     return
                
    elif nchannels==4:
        gslc_fp(mat, inFile, inFolder, base_path, azlks, rglks, recip, max_workers,
        start_x, start_y, xres, yres, projection, fmt, cog, ovr, comp,
        inshape, outshape, out_dir)
        # if mat=='S2':
        #     print("Extracting S2 matrix elements...")
        #     out_dir = os.path.join(inFolder,os.path.basename(inFile).split('.h5')[0],'S2')
        #     temp_dir = os.path.join(inFolder,os.path.basename(inFile).split('.h5')[0],'S2','temp')
        #     h5_polsar(
        #         h5_file=inFile,
        #         dataset_paths={
        #             "HH": f"{base_path}/HH",
        #             "HV": f"{base_path}/HV",
        #             "VH": f"{base_path}/VH",
        #             "VV": f"{base_path}/VV",
        #         },
        #         output_dir=out_dir,
        #         temp_dir=temp_dir,
        #         azlks=azlks,
        #         rglks=rglks,
        #         matrix_type = 'S2',
        #         apply_multilook=False,
        #         chunk_size_x=get_ml_chunk(rglks, 512),
        #         chunk_size_y=get_ml_chunk(azlks, 512),
        #         max_workers=max_workers,
        #         start_x=start_x, start_y=start_y, xres=xres, yres=yres, epsg=projection,
        #         fmt=fmt,cog=cog,ovr =ovr,comp=comp,
        #         dtype = np.complex64,
        #         inshape=inshape,
        #         outshape=outshape
        #     )
            
        # elif mat=='T4':
        #     print("Extracting T4 matrix elements...")
        #     out_dir = os.path.join(inFolder,os.path.basename(inFile).split('.h5')[0],'T4')
        #     temp_dir = os.path.join(inFolder,os.path.basename(inFile).split('.h5')[0],'T4','temp')
        #     h5_polsar(
        #         h5_file=inFile,
        #         dataset_paths={
        #             "HH": f"{base_path}/HH",
        #             "HV": f"{base_path}/HV",
        #             "VH": f"{base_path}/VH",
        #             "VV": f"{base_path}/VV",
        #         },
        #         output_dir=out_dir,
        #         temp_dir=temp_dir,
        #         azlks=azlks,
        #         rglks=rglks,
        #         matrix_type = 'T4',
        #         apply_multilook=True,
        #         chunk_size_x=get_ml_chunk(rglks, 512),
        #         chunk_size_y=get_ml_chunk(azlks, 512),
        #         max_workers=max_workers,
        #         start_x=start_x, start_y=start_y, xres=xres, yres=yres, epsg=projection,
        #         fmt=fmt,cog=cog,ovr =ovr,comp=comp,
        #         dtype = np.float32,
        #         inshape=inshape,
        #         outshape=outshape
        #     )

        # elif mat=='T3':
        #     print("Extracting T3 matrix elements...")
        #     out_dir = os.path.join(inFolder,os.path.basename(inFile).split('.h5')[0],'T3')
        #     temp_dir = os.path.join(inFolder,os.path.basename(inFile).split('.h5')[0],'T3','temp')
        #     h5_polsar(
        #         h5_file=inFile,
        #         dataset_paths={
        #             "HH": f"{base_path}/HH",
        #             "HV": f"{base_path}/HV",
        #             "VH": f"{base_path}/VH",
        #             "VV": f"{base_path}/VV",
        #         },
        #         output_dir=out_dir,
        #         temp_dir=temp_dir,
        #         azlks=azlks,
        #         rglks=rglks,
        #         matrix_type = 'T3',
        #         apply_multilook=True,
        #         chunk_size_x=get_ml_chunk(rglks, 512),
        #         chunk_size_y=get_ml_chunk(azlks, 512),
        #         max_workers=max_workers,
        #         start_x=start_x, start_y=start_y, xres=xres, yres=yres, epsg=int(projection),
        #         fmt=fmt,cog=cog,ovr =ovr,comp=comp,
        #         dtype = np.float32,
        #         inshape=inshape,
        #         outshape=outshape
        #     )
        # elif mat=='C4':
        #     print("Extracting C4 matrix elements...")
        #     out_dir = os.path.join(inFolder,os.path.basename(inFile).split('.h5')[0],'C4')
        #     temp_dir = os.path.join(inFolder,os.path.basename(inFile).split('.h5')[0],'C4','temp')
        #     h5_polsar(
        #         h5_file=inFile,
        #         dataset_paths={
        #             "HH": f"{base_path}/HH",
        #             "HV": f"{base_path}/HV",
        #             "VH": f"{base_path}/VH",
        #             "VV": f"{base_path}/VV",
        #         },
        #         output_dir=out_dir,
        #         temp_dir=temp_dir,
        #         azlks=azlks,
        #         rglks=rglks,
        #         matrix_type = 'C4',
        #         apply_multilook=True,
        #         chunk_size_x=get_ml_chunk(rglks, 512),
        #         chunk_size_y=get_ml_chunk(azlks, 512),
        #         max_workers=max_workers,
        #         start_x=start_x, start_y=start_y, xres=xres, yres=yres, epsg=int(projection),
        #         fmt=fmt,cog=cog,ovr =ovr,comp=comp,
        #         dtype = np.float32,
        #         inshape=inshape,
        #         outshape=outshape            
        #     )   
        # elif mat=='C3':
        #     print("Extracting C3 matrix elements...")
        #     out_dir = os.path.join(inFolder,os.path.basename(inFile).split('.h5')[0],'C3')
        #     temp_dir = os.path.join(inFolder,os.path.basename(inFile).split('.h5')[0],'C3','temp')
        #     h5_polsar(
        #         h5_file=inFile,
        #         dataset_paths={
        #             "HH": f"{base_path}/HH",
        #             "HV": f"{base_path}/HV",
        #             "VH": f"{base_path}/VH",
        #             "VV": f"{base_path}/VV",
        #         },
        #         output_dir=out_dir,
        #         temp_dir=temp_dir,
        #         azlks=azlks,
        #         rglks=rglks,
        #         matrix_type = 'C3',
        #         apply_multilook=True,
        #         chunk_size_x=get_ml_chunk(rglks, 512),
        #         chunk_size_y=get_ml_chunk(azlks, 512),
        #         max_workers=max_workers,
        #         start_x=start_x, start_y=start_y, xres=xres, yres=yres, epsg=int(projection),
        #         fmt=fmt,      cog=cog,ovr =ovr,comp=comp,      
        #         dtype = np.float32,
        #         inshape=inshape,
        #         outshape=outshape            
        #     )
        
        # elif mat=='C2HV':
        #     print("Extracting C2HV matrix elements...")
        #     out_dir = os.path.join(inFolder,os.path.basename(inFile).split('.h5')[0],'C2HV')
        #     temp_dir = os.path.join(inFolder,os.path.basename(inFile).split('.h5')[0],'C2HV','temp')
        #     h5_polsar(
        #         h5_file=inFile,
        #         dataset_paths={
        #             "HH": f"{base_path}/HH",
        #             "VV": f"{base_path}/VV",
        #         },
        #         output_dir=out_dir, temp_dir=temp_dir,
        #         azlks=azlks, rglks=rglks, matrix_type = 'C2HV', apply_multilook=True,
        #         chunk_size_x=get_ml_chunk(rglks, 512), chunk_size_y=get_ml_chunk(azlks, 512), max_workers=max_workers,
        #         start_x=start_x, start_y=start_y, xres=xres, yres=yres, epsg=projection,
        #         fmt=fmt, cog=cog,ovr =ovr,comp=comp,dtype = np.float32, inshape=inshape,outshape=outshape            
        #     )

        # elif mat=='C2HX':
        #     print("Extracting C2HX matrix elements...")
        #     out_dir = os.path.join(inFolder,os.path.basename(inFile).split('.h5')[0],'C2HX')
        #     temp_dir = os.path.join(inFolder,os.path.basename(inFile).split('.h5')[0],'C2HX','temp')
        #     h5_polsar(
        #         h5_file=inFile,
        #         dataset_paths={
        #             "HH": f"{base_path}/HH",
        #             "HV": f"{base_path}/HV",
        #         },
        #         output_dir=out_dir, temp_dir=temp_dir,
        #         azlks=azlks, rglks=rglks, matrix_type = 'C2HX', apply_multilook=True,
        #         chunk_size_x=get_ml_chunk(rglks, 512), chunk_size_y=get_ml_chunk(azlks, 512), max_workers=max_workers,
        #         start_x=start_x, start_y=start_y, xres=xres, yres=yres, epsg=projection,
        #         fmt=fmt, cog=cog,ovr =ovr,comp=comp,dtype = np.float32, inshape=inshape,outshape=outshape            
        #     )

        # elif mat=='C2VX':
        #     print("Extracting C2VX matrix elements...")
        #     out_dir = os.path.join(inFolder,os.path.basename(inFile).split('.h5')[0],'C2VX')
        #     temp_dir = os.path.join(inFolder,os.path.basename(inFile).split('.h5')[0],'C2VX','temp')
        #     h5_polsar(
        #         h5_file=inFile,
        #         dataset_paths={
        #             "VV": f"{base_path}/VV",
        #             "VH": f"{base_path}/VH",
        #         },
        #         output_dir=out_dir, temp_dir=temp_dir,
        #         azlks=azlks, rglks=rglks, matrix_type = 'C2VX', apply_multilook=True,
        #         chunk_size_x=get_ml_chunk(rglks, 512), chunk_size_y=get_ml_chunk(azlks, 512), max_workers=max_workers,
        #         start_x=start_x, start_y=start_y, xres=xres, yres=yres, epsg=projection,
        #         fmt=fmt,cog=cog,ovr =ovr,comp=comp, dtype = np.float32, inshape=inshape,outshape=outshape            
        #     )

        # elif mat=='T2HV':
        #     print("Extracting T2HV matrix elements...")
        #     out_dir = os.path.join(inFolder,os.path.basename(inFile).split('.h5')[0],'T2HV')
        #     temp_dir = os.path.join(inFolder,os.path.basename(inFile).split('.h5')[0],'T2HV','temp')
        #     h5_polsar(
        #         h5_file=inFile,
        #         dataset_paths={
        #             "HH": f"{base_path}/HH",
        #             "VV": f"{base_path}/VV",
        #         },
        #         output_dir=out_dir, temp_dir=temp_dir,
        #         azlks=azlks, rglks=rglks, matrix_type = 'T2HV', apply_multilook=True,
        #         chunk_size_x=get_ml_chunk(rglks, 512), chunk_size_y=get_ml_chunk(azlks, 512), max_workers=max_workers,
        #         start_x=start_x, start_y=start_y, xres=xres, yres=yres, epsg=projection,
        #         fmt=fmt, cog=cog,ovr =ovr,comp=comp,dtype = np.float32, inshape=inshape,outshape=outshape            
        #     )
            
        # else:
        #     raise ValueError(f"Unsupported matrix type: {mat} please choose from S2, C4, C3, T3, T4, T2HV, C2HV, C2HX, C2VX")


@time_it  
def nisar_rslc(inFile, mat='C3', azlks=22,rglks=10, 
               fmt='tif', cog=False, ovr = [2, 4, 8, 16], comp=False,
              out_dir=None,
              recip=False,
               max_workers=None ):
    """
    Extracts the C2 (for dual-pol), S2/C3/T3 (for full-pol) matrix elements from a NISAR RSLC HDF5 file 
    and saves them into respective binary files.
    
    Example:
    --------
    >>> nisar_rslc("path_to_file.h5", azlks=30, rglks=15)
    This will extract the C2 (for dual-pol), S2/C3/T3 (for full-pol) matrix elements from the specified NISAR RSLC file 
    and save them in the respective folders.
    
    Parameters:
    -----------
    inFile : str
        The path to the NISAR RSLC HDF5 file containing the radar data.

    azlks : int, optional (default=22)
        The number of azimuth looks for multi-looking. 

    rglks : int, optional (default=10)
        The number of range looks for multi-looking. 
        
    mat : str, optional (default = C2 for dual-pol, C3 for full-pol)
        Type of matrix to extract. Valid options are 'C2', 'S2', 'C3', and 'T3'.
         

    Returns:
    --------
    None
        The function does not return any value. Instead, it creates a folder 
        named `C2/S2/C3/T3` (if not already present) and saves the following binary files:


    Raises:
    -------
    Exception
        If the RSLC HDF5 file is invalid or cannot be read.


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
        gslc_dp(mat,inFile, inFolder, base_path, azlks, rglks, recip, max_workers,
            start_x, start_y, xres, yres, projection, fmt, cog, ovr, comp,
            None, None, listOfPolarizations, out_dir)   
        # if 'HH' in listOfPolarizations and 'HV' in listOfPolarizations:
        #     print("Extracting C2HX matrix elements...")
        #     out_dir = os.path.join(inFolder,os.path.basename(inFile).split('.h5')[0],'C2HX')
        #     temp_dir = os.path.join(inFolder,os.path.basename(inFile).split('.h5')[0],'C2HX','temp')
            
        #     h5_polsar(
        #                 h5_file=inFile,
        #                 dataset_paths={
        #                     "HH": f"{base_path}/HH",
        #                     "HV": f"{base_path}/HV",
        #                 },
        #                 output_dir=out_dir,
        #                 temp_dir=temp_dir,
        #                 azlks=azlks,
        #                 rglks=rglks,
        #                 matrix_type = 'C2HX',
        #                 apply_multilook=True,
        #                 chunk_size_x=get_ml_chunk(rglks, 512),
        #                 chunk_size_y=get_ml_chunk(azlks, 512),
        #                 max_workers=max_workers,
        #                 start_x=start_x, start_y=start_y, xres=xres/rglks, yres=yres/azlks, epsg=projection,
        #                 fmt=fmt,cog=cog,ovr =ovr,comp=comp,
        #                 dtype = np.float32,
        #                 # inshape=inshape,
        #                 # outshape=outshape
        #             )
    

        # elif 'VV' in listOfPolarizations and 'VH' in listOfPolarizations:
        #     print("Extracting C2VX matrix elements...")
        #     out_dir = os.path.join(inFolder,os.path.basename(inFile).split('.h5')[0],'C2VX')
        #     temp_dir = os.path.join(inFolder,os.path.basename(inFile).split('.h5')[0],'C2VX','temp')
        #     h5_polsar(
        #                 h5_file=inFile,
        #                 dataset_paths={
        #                     "VV": f"{base_path}/VV",
        #                     "VH": f"{base_path}/VH",
        #                 },
        #                 output_dir=out_dir,
        #                 temp_dir=temp_dir,
        #                 azlks=azlks,
        #                 rglks=rglks,
        #                 matrix_type = 'C2VX',
        #                 apply_multilook=True,
        #                 chunk_size_x=get_ml_chunk(rglks, 512),
        #                 chunk_size_y=get_ml_chunk(azlks, 512),
        #                 max_workers=max_workers,
        #                 start_x=start_x, start_y=start_y, xres=xres/rglks, yres=yres/azlks, epsg=projection,
        #                 fmt=fmt,cog=cog,ovr =ovr,comp=comp,
        #                 dtype = np.float32,
        #                 # inshape=inshape,
        #                 # outshape=outshape
        #             )

        # elif 'HH' in listOfPolarizations and 'VV' in listOfPolarizations:
        #     print("Extracting C2HV matrix elements...")
        #     out_dir = os.path.join(inFolder,os.path.basename(inFile).split('.h5')[0],'C2HV')
        #     temp_dir = os.path.join(inFolder,os.path.basename(inFile).split('.h5')[0],'C2HV','temp')
        #     h5_polsar(
        #                 h5_file=inFile,
        #                 dataset_paths={
        #                     "HH": f"{base_path}/HH",
        #                     "VV": f"{base_path}/VV",
        #                 },
        #                 output_dir=out_dir,
        #                 temp_dir=temp_dir,
        #                 azlks=azlks,
        #                 rglks=rglks,
        #                 matrix_type = 'C2HV',
        #                 apply_multilook=True,
        #                 chunk_size_x=get_ml_chunk(rglks, 512),
        #                 chunk_size_y=get_ml_chunk(azlks, 512),
        #                 max_workers=max_workers,
        #                 start_x=start_x, start_y=start_y, xres=xres/rglks, yres=yres/azlks, epsg=projection,
        #                 fmt=fmt,cog=cog,ovr =ovr,comp=comp,
        #                 dtype = np.float32,
        #                 # inshape=inshape,
        #                 # outshape=outshape
        #             )

        # else:
        #     print("No HH, HV, VV, or VH polarizations found in the file.")

        #     return
        

    elif nchannels==4:
        gslc_fp(mat, inFile, inFolder, base_path, azlks, rglks, recip, max_workers,
        start_x, start_y, xres, yres, projection, fmt, cog, ovr, comp,
        None, None, out_dir)
        # if mat=='S2':
        #     print("Extracting S2 matrix elements...")
        #     out_dir = os.path.join(inFolder,os.path.basename(inFile).split('.h5')[0],'S2')
        #     temp_dir = os.path.join(inFolder,os.path.basename(inFile).split('.h5')[0],'S2','temp')
        #     h5_polsar(
        #         h5_file=inFile,
        #         dataset_paths={
        #             "HH": f"{base_path}/HH",
        #             "HV": f"{base_path}/HV",
        #             "VH": f"{base_path}/VH",
        #             "VV": f"{base_path}/VV",
        #         },
        #         output_dir=out_dir,
        #         temp_dir=temp_dir,
        #         azlks=azlks,
        #         rglks=rglks,
        #         matrix_type = 'S2',
        #         apply_multilook=False,
        #         chunk_size_x=get_ml_chunk(rglks, 512),
        #         chunk_size_y=get_ml_chunk(azlks, 512),
        #         max_workers=max_workers,
        #         start_x=start_x, start_y=start_y, xres=xres, yres=yres, epsg=projection,
        #         fmt=fmt,cog=cog,ovr =ovr,comp=comp,
        #         dtype = np.complex64,
        #         # inshape=inshape,
        #         # outshape=outshape
        #     )
            
        # elif mat=='T4':
        #     print("Extracting T4 matrix elements...")
        #     out_dir = os.path.join(inFolder,os.path.basename(inFile).split('.h5')[0],'T4')
        #     temp_dir = os.path.join(inFolder,os.path.basename(inFile).split('.h5')[0],'T4','temp')
        #     h5_polsar(
        #         h5_file=inFile,
        #         dataset_paths={
        #             "HH": f"{base_path}/HH",
        #             "HV": f"{base_path}/HV",
        #             "VH": f"{base_path}/VH",
        #             "VV": f"{base_path}/VV",
        #         },
        #         output_dir=out_dir,
        #         temp_dir=temp_dir,
        #         azlks=azlks,
        #         rglks=rglks,
        #         matrix_type = 'T4',
        #         apply_multilook=True,
        #         chunk_size_x=get_ml_chunk(rglks, 512),
        #         chunk_size_y=get_ml_chunk(azlks, 512),
        #         max_workers=max_workers,
        #         start_x=start_x, start_y=start_y, xres=xres/rglks, yres=yres/azlks, epsg=projection,
        #         fmt=fmt,cog=cog,ovr =ovr,comp=comp,
        #         dtype = np.float32,
        #         # inshape=inshape,
        #         # outshape=outshape
        #     )

        # elif mat=='T3':
        #     print("Extracting T3 matrix elements...")
        #     out_dir = os.path.join(inFolder,os.path.basename(inFile).split('.h5')[0],'T3')
        #     temp_dir = os.path.join(inFolder,os.path.basename(inFile).split('.h5')[0],'T3','temp')
        #     h5_polsar(
        #         h5_file=inFile,
        #         dataset_paths={
        #             "HH": f"{base_path}/HH",
        #             "HV": f"{base_path}/HV",
        #             "VH": f"{base_path}/VH",
        #             "VV": f"{base_path}/VV",
        #         },
        #         output_dir=out_dir,
        #         temp_dir=temp_dir,
        #         azlks=azlks,
        #         rglks=rglks,
        #         matrix_type = 'T3',
        #         apply_multilook=True,
        #         chunk_size_x=get_ml_chunk(rglks, 512),
        #         chunk_size_y=get_ml_chunk(azlks, 512),
        #         max_workers=max_workers,
        #         start_x=start_x, start_y=start_y, xres=xres/rglks, yres=yres/azlks, epsg=projection,
        #         fmt=fmt,cog=cog,ovr =ovr,comp=comp,
        #         dtype = np.float32,
        #         # inshape=inshape,
        #         # outshape=outshape
        #     )
        # elif mat=='C4':
        #     print("Extracting C4 matrix elements...")
        #     out_dir = os.path.join(inFolder,os.path.basename(inFile).split('.h5')[0],'C4')
        #     temp_dir = os.path.join(inFolder,os.path.basename(inFile).split('.h5')[0],'C4','temp')
        #     h5_polsar(
        #         h5_file=inFile,
        #         dataset_paths={
        #             "HH": f"{base_path}/HH",
        #             "HV": f"{base_path}/HV",
        #             "VH": f"{base_path}/VH",
        #             "VV": f"{base_path}/VV",
        #         },
        #         output_dir=out_dir,
        #         temp_dir=temp_dir,
        #         azlks=azlks,
        #         rglks=rglks,
        #         matrix_type = 'C4',
        #         apply_multilook=True,
        #         chunk_size_x=get_ml_chunk(rglks, 512),
        #         chunk_size_y=get_ml_chunk(azlks, 512),
        #         max_workers=max_workers,
        #         start_x=start_x, start_y=start_y, xres=xres/rglks, yres=yres/azlks, epsg=projection,
        #         fmt=fmt,cog=cog,ovr =ovr,comp=comp,
        #         dtype = np.float32,
        #         # inshape=inshape,
        #         # outshape=outshape            
        #     )   
        # elif mat=='C3':
        #     print("Extracting C3 matrix elements...")
        #     out_dir = os.path.join(inFolder,os.path.basename(inFile).split('.h5')[0],'C3')
        #     temp_dir = os.path.join(inFolder,os.path.basename(inFile).split('.h5')[0],'C3','temp')
        #     h5_polsar(
        #         h5_file=inFile,
        #         dataset_paths={
        #             "HH": f"{base_path}/HH",
        #             "HV": f"{base_path}/HV",
        #             "VH": f"{base_path}/VH",
        #             "VV": f"{base_path}/VV",
        #         },
        #         output_dir=out_dir,
        #         temp_dir=temp_dir,
        #         azlks=azlks,
        #         rglks=rglks,
        #         matrix_type = 'C3',
        #         apply_multilook=True,
        #         chunk_size_x=get_ml_chunk(rglks, 512),
        #         chunk_size_y=get_ml_chunk(azlks, 512),
        #         max_workers=max_workers,
        #         start_x=start_x, start_y=start_y, xres=xres/rglks, yres=yres/azlks, epsg=projection,
        #         fmt=fmt,   cog=cog,ovr =ovr,comp=comp,         
        #         dtype = np.float32,
        #         # inshape=inshape,
        #         # outshape=outshape            
        #     )
        
        # elif mat=='C2HV':
        #     print("Extracting C2HV matrix elements...")
        #     out_dir = os.path.join(inFolder,os.path.basename(inFile).split('.h5')[0],'C2HV')
        #     temp_dir = os.path.join(inFolder,os.path.basename(inFile).split('.h5')[0],'C2HV','temp')
        #     h5_polsar(
        #         h5_file=inFile,
        #         dataset_paths={
        #             "HH": f"{base_path}/HH",
        #             "VV": f"{base_path}/VV",
        #         },
        #         output_dir=out_dir, temp_dir=temp_dir,
        #         azlks=azlks, rglks=rglks, matrix_type = 'C2HV', apply_multilook=True,
        #         chunk_size_x=get_ml_chunk(rglks, 512), chunk_size_y=get_ml_chunk(azlks, 512), max_workers=max_workers,
        #         start_x=start_x, start_y=start_y, xres=xres/rglks, yres=yres/azlks, epsg=projection,
        #         fmt=fmt,cog=cog,ovr =ovr,comp=comp, dtype = np.float32, 
        #         # inshape=inshape,outshape=outshape            
        #     )

        # elif mat=='C2HX':
        #     print("Extracting C2HX matrix elements...")
        #     out_dir = os.path.join(inFolder,os.path.basename(inFile).split('.h5')[0],'C2HX')
        #     temp_dir = os.path.join(inFolder,os.path.basename(inFile).split('.h5')[0],'C2HX','temp')
        #     h5_polsar(
        #         h5_file=inFile,
        #         dataset_paths={
        #             "HH": f"{base_path}/HH",
        #             "HV": f"{base_path}/HV",
        #         },
        #         output_dir=out_dir, temp_dir=temp_dir,
        #         azlks=azlks, rglks=rglks, matrix_type = 'C2HX', apply_multilook=True,
        #         chunk_size_x=get_ml_chunk(rglks, 512), chunk_size_y=get_ml_chunk(azlks, 512), max_workers=max_workers,
        #         start_x=start_x, start_y=start_y, xres=xres/rglks, yres=yres/azlks, epsg=projection,
        #         fmt=fmt,cog=cog,ovr =ovr,comp=comp, dtype = np.float32, 
        #         # inshape=inshape,outshape=outshape            
        #     )

        # elif mat=='C2VX':
        #     print("Extracting C2VX matrix elements...")
        #     out_dir = os.path.join(inFolder,os.path.basename(inFile).split('.h5')[0],'C2VX')
        #     temp_dir = os.path.join(inFolder,os.path.basename(inFile).split('.h5')[0],'C2VX','temp')
        #     h5_polsar(
        #         h5_file=inFile,
        #         dataset_paths={
        #             "VV": f"{base_path}/VV",
        #             "VH": f"{base_path}/VH",
        #         },
        #         output_dir=out_dir, temp_dir=temp_dir,
        #         azlks=azlks, rglks=rglks, matrix_type = 'C2VX', apply_multilook=True,
        #         chunk_size_x=get_ml_chunk(rglks, 512), chunk_size_y=get_ml_chunk(azlks, 512), max_workers=max_workers,
        #         start_x=start_x, start_y=start_y, xres=xres/rglks, yres=yres/azlks, epsg=projection,
        #         fmt=fmt,cog=cog,ovr =ovr,comp=comp, dtype = np.float32, 
        #         # inshape=inshape,outshape=outshape            
        #     )

        # elif mat=='T2HV':
        #     print("Extracting T2HV matrix elements...")
        #     out_dir = os.path.join(inFolder,os.path.basename(inFile).split('.h5')[0],'T2HV')
        #     temp_dir = os.path.join(inFolder,os.path.basename(inFile).split('.h5')[0],'T2HV','temp')
        #     h5_polsar(
        #         h5_file=inFile,
        #         dataset_paths={
        #             "HH": f"{base_path}/HH",
        #             "VV": f"{base_path}/VV",
        #         },
        #         output_dir=out_dir, temp_dir=temp_dir,
        #         azlks=azlks, rglks=rglks, matrix_type = 'T2HV', apply_multilook=True,
        #         chunk_size_x=get_ml_chunk(rglks, 512), chunk_size_y=get_ml_chunk(azlks, 512), max_workers=max_workers,
        #         start_x=start_x, start_y=start_y, xres=xres/rglks, yres=yres/azlks, epsg=projection,
        #         fmt=fmt, cog=cog,ovr =ovr,comp=comp,dtype = np.float32, 
        #         # inshape=inshape,outshape=outshape            
        #     )
            
        # else:
        #     raise ValueError(f"Unsupported matrix type: {mat} please choose from S2, C4, C3, T3, T4, T2HV, C2HV, C2HX, C2VX")