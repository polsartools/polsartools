import numpy as np
from osgeo import gdal
gdal.UseExceptions()
import os,glob,tables
import xml.etree.ElementTree as ET
from polsartools.utils.utils import time_it
# from polsartools.utils.io_utils import write_T3, write_C3,write_C4,write_s2_bin
from polsartools.utils.geo_utils import geocode_grid, intp_grid, update_vrt, write_latlon
from polsartools.utils.proc_utils import process_chunks_parallel
from polsartools.utils.utils import conv2d,time_it

from polsartools.utils.h5_utils import h5_polsar, get_ml_chunk
from netCDF4 import Dataset
from polsartools.sensors.nisar import nisar_fp,nisar_dp

def load_asar_meta(filepath):
    data_dict = {}

    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()
            if line:  # Ensure line is not empty
                key, value = line.split("=", 1)  # Split only on first "="
                data_dict[key.strip()] = value.strip()
    return data_dict

@time_it
def convert_l1_tif_dB(infolder,  backscatter=1, complex_flag=0, fmt="tif", 
                   cog_flag=False, cog_overviews = [2, 4, 8, 16], 
                   write_flag=True, max_workers=None):
    
    l_band_files = list(sorted(glob.glob(os.path.join(infolder, "ASAR_L_JOINT_*LEVEL1_*.tif"))))
    s_band_files = list(sorted(glob.glob(os.path.join(infolder, "ASAR_S_JOINT_*LEVEL1_*.tif"))))
    inc_file = glob.glob(os.path.join(infolder, "ASAR_*LEVEL1_INC_MAP*.tif"))
    window_size=3
    # print(len(l_band_files))
    # print(len(s_band_files))
    # print(len(inc_file))
    
    nb_file = glob.glob(os.path.join(infolder, "ASAR_*LEVEL1_BAND_META*.txt"))
    
    if len(nb_file) >0:
        print("Found meta data file applying noise bias.")
        meta_dict = load_asar_meta(nb_file[0])
        l_hh_nb = meta_dict['Image_Noise_Bias_L_HH']
        l_hv_nb = meta_dict['Image_Noise_Bias_L_HV']
        l_vh_nb = meta_dict['Image_Noise_Bias_L_VH']
        l_vv_nb = meta_dict['Image_Noise_Bias_L_VV']

        s_hh_nb = meta_dict['Image_Noise_Bias_S_HH']
        s_hv_nb = meta_dict['Image_Noise_Bias_S_HV']
        s_vh_nb = meta_dict['Image_Noise_Bias_S_VH']
        s_vv_nb = meta_dict['Image_Noise_Bias_S_VV']
    else:
        print("No meta data file found. Applying noise bias of 0.")
        l_hh_nb = 0
        l_hv_nb = 0
        l_vh_nb = 0
        l_vv_nb = 0

        s_hh_nb = 0
        s_hv_nb = 0
        s_vh_nb = 0
        s_vv_nb = 0

    if backscatter==2:
        out_fix = 'g0'
    elif backscatter==3:
        out_fix = 'b0'
    else:
        out_fix = 's0' 
    
    def output_names_dB(band_files, prefix):
        out_paths = []
        for file in band_files:
            pol = file.split("_LEVEL1_")[1].split(".tif")[0]  # Extract polarization (e.g., HH, HV, VH, VV)
            out_paths.append(os.path.join(infolder, f"{prefix}_{pol}_{out_fix}_dB.{fmt}"))
        return out_paths
    def output_names_complex(band_files, prefix):
        out_paths = []
        for file in band_files:
            pol = file.split("_LEVEL1_")[1].split(".tif")[0]  # Extract polarization (e.g., HH, HV, VH, VV)           
            real_path = os.path.join(infolder, f"{prefix}_{pol}_{out_fix}_real.{fmt}")
            imag_path = os.path.join(infolder, f"{prefix}_{pol}_{out_fix}_imag.{fmt}")
        
            out_paths.extend([real_path, imag_path]) 
        return out_paths
    
    output_l_filepaths=[]
    output_s_filepaths=[]
    
    if l_band_files:
        if complex_flag:
            output_l_filepaths.extend(output_names_complex(l_band_files, "L"))
        else:
            output_l_filepaths.extend(output_names_dB(l_band_files, "L"))

    if s_band_files:
        if complex_flag:
            output_s_filepaths.extend(output_names_complex(s_band_files, "S"))
        else:
            output_s_filepaths.extend(output_names_dB(s_band_files, "S"))
    

    def copy_gcps_to_outputs(input_filepaths, output_filepaths):
        """ Copies GCPs from one input file to all output files. """
        input_filepath = input_filepaths[0]
        # Open the input file
        input_dataset = gdal.Open(input_filepath, gdal.GA_ReadOnly)
        if input_dataset is None:
            raise FileNotFoundError(f"Cannot open {input_filepath}")

        # Extract GCPs
        gcps = input_dataset.GetGCPs()
        gcp_projection = input_dataset.GetGCPProjection()

        if not gcps:
            # print("No GCPs found in the input file.")
            return

        # Apply GCPs to each output file
        for output_filepath in output_filepaths:
            output_dataset = gdal.Open(output_filepath, gdal.GA_Update)
            if output_dataset is None:
                raise FileNotFoundError(f"Cannot open {output_filepath}")

            output_dataset.SetGCPs(gcps, gcp_projection)
            output_dataset.GetRasterBand(1).SetNoDataValue(np.nan)
            output_dataset.FlushCache()  
            output_dataset = None 

        # Close the input file
        input_dataset = None

    if len(l_band_files)==4:
        print("Processing L-band data...")
        input_filepaths = l_band_files + inc_file
        # print(input_filepaths)
        process_chunks_parallel(input_filepaths, output_l_filepaths, window_size, 
                                write_flag, process_chunk_gtif,
                                *[complex_flag, backscatter, l_hh_nb, l_hv_nb, l_vh_nb, l_vv_nb],
                                bands_to_read=[2,2,2,2,1] , block_size=(512, 512), 
                                max_workers=max_workers, num_outputs=len(output_l_filepaths),
                                cog_flag=cog_flag, cog_overviews=cog_overviews,
                                post_proc=copy_gcps_to_outputs
                                )

    if len(s_band_files)==4:
        print("Processing S-band data...")
        input_filepaths = s_band_files + inc_file
        # print(input_filepaths)
        process_chunks_parallel(input_filepaths, output_s_filepaths, window_size, 
                                write_flag, process_chunk_gtif,
                                *[complex_flag, backscatter, s_hh_nb, s_hv_nb, s_vh_nb, s_vv_nb],
                                bands_to_read=[2,2,2,2,1] , block_size=(512, 512), 
                                max_workers=max_workers, num_outputs=len(output_s_filepaths),
                                cog_flag=cog_flag, cog_overviews=cog_overviews,
                                post_proc=copy_gcps_to_outputs
                                )
    
    

def process_chunk_gtif(chunks, window_size, *args):
    # kernel = np.ones((window_size,window_size),np.float32)/(window_size*window_size)
    complex_flag=int(args[-6])
    backscatter=int(args[-5])
    hh_nb=float(args[-4])
    hv_nb=float(args[-3])
    vh_nb=float(args[-2])
    vv_nb=float(args[-1])
    
    cc_dB = 42.0
    cc_linear = np.sqrt(10**(cc_dB/10.0))

    if backscatter==2 and complex_flag==1:

        hh = np.array(chunks[0])+1j*np.array(chunks[1])*np.tan(np.deg2rad(np.array(chunks[8])))/cc_linear
        hv = np.array(chunks[2])+1j*np.array(chunks[3])*np.tan(np.deg2rad(np.array(chunks[8])))/cc_linear
        vh = np.array(chunks[4])+1j*np.array(chunks[5])*np.tan(np.deg2rad(np.array(chunks[8])))/cc_linear
        vv = np.array(chunks[6])+1j*np.array(chunks[7])*np.tan(np.deg2rad(np.array(chunks[8])))/cc_linear
    elif backscatter==2 and complex_flag==0:
        hh = 10*np.log10(np.abs(np.array(chunks[0])+1j*np.array(chunks[1]))**2-hh_nb)+10*np.log10(np.tan(np.deg2rad(np.array(chunks[8]))))-cc_dB
        hv = 10*np.log10(np.abs(np.array(chunks[2])+1j*np.array(chunks[3]))**2-hv_nb)+10*np.log10(np.tan(np.deg2rad(np.array(chunks[8]))))-cc_dB
        vh = 10*np.log10(np.abs(np.array(chunks[4])+1j*np.array(chunks[5]))**2-vh_nb)+10*np.log10(np.tan(np.deg2rad(np.array(chunks[8]))))-cc_dB
        vv = 10*np.log10(np.abs(np.array(chunks[6])+1j*np.array(chunks[7]))**2-vv_nb)+10*np.log10(np.tan(np.deg2rad(np.array(chunks[8]))))-cc_dB
        
    elif backscatter==3 and complex_flag==0:
        hh = 10*np.log10(np.abs(np.array(chunks[0])+1j*np.array(chunks[1]))**2-hh_nb)-cc_dB
        hv = 10*np.log10(np.abs(np.array(chunks[2])+1j*np.array(chunks[3]))**2-hv_nb)-cc_dB
        vh = 10*np.log10(np.abs(np.array(chunks[4])+1j*np.array(chunks[5]))**2-vh_nb)-cc_dB
        vv = 10*np.log10(np.abs(np.array(chunks[6])+1j*np.array(chunks[7]))**2-vv_nb)-cc_dB
    elif backscatter==3 and complex_flag==1:
        hh = np.array(chunks[0])+1j*np.array(chunks[1])/cc_linear
        hv = np.array(chunks[2])+1j*np.array(chunks[3])/cc_linear
        vh = np.array(chunks[4])+1j*np.array(chunks[5])/cc_linear
        vv = np.array(chunks[6])+1j*np.array(chunks[7])/cc_linear
    elif complex_flag==0:
        hh = 10*np.log10(np.abs(np.array(chunks[0])+1j*np.array(chunks[1]))**2-hh_nb)+10*np.log10(np.sin(np.deg2rad(np.array(chunks[8]))))-cc_dB
        hv = 10*np.log10(np.abs(np.array(chunks[2])+1j*np.array(chunks[3]))**2-hv_nb)+10*np.log10(np.sin(np.deg2rad(np.array(chunks[8]))))-cc_dB
        vh = 10*np.log10(np.abs(np.array(chunks[4])+1j*np.array(chunks[5]))**2-vh_nb)+10*np.log10(np.sin(np.deg2rad(np.array(chunks[8]))))-cc_dB
        vv = 10*np.log10(np.abs(np.array(chunks[6])+1j*np.array(chunks[7]))**2-vv_nb)+10*np.log10(np.sin(np.deg2rad(np.array(chunks[8]))))-cc_dB
    else:
        hh = np.array(chunks[0])+1j*np.array(chunks[1])*np.sin(np.deg2rad(np.array(chunks[8])))/cc_linear
        hv = np.array(chunks[2])+1j*np.array(chunks[3])*np.sin(np.deg2rad(np.array(chunks[8])))/cc_linear
        vh = np.array(chunks[4])+1j*np.array(chunks[5])*np.sin(np.deg2rad(np.array(chunks[8])))/cc_linear
        vv = np.array(chunks[6])+1j*np.array(chunks[7])*np.sin(np.deg2rad(np.array(chunks[8])))/cc_linear

    hh[hh==0]=np.nan
    hh[hh==np.inf]=np.nan
    hh[hh==-np.inf]=np.nan
    
    hv[hv==0]=np.nan
    hv[hv==np.inf]=np.nan
    hv[hv==-np.inf]=np.nan
    
    vh[vh==0]=np.nan
    vh[vh==np.inf]=np.nan
    vh[vh==-np.inf]=np.nan
    
    vv[vv==0]=np.nan
    vv[vv==np.inf]=np.nan
    vv[vv==-np.inf]=np.nan
    
    if complex_flag:
        return np.real(hh),np.imag(hh),np.real(hv),np.imag(hv),np.real(vh),np.imag(vh),np.real(vv),np.imag(vv)
    else:
        return hh,hv,vh,vv

def get_rslc_path(h5, freq_band):
    for product_type in ['RSLC', 'SLC']:
        base_path = f'/science/{freq_band}SAR/{product_type}/swaths/frequencyA'
        if h5.__contains__(base_path):
            return base_path
        else:
            print(f"Base path not found: {base_path}")
    return None

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

@time_it
def isro_asar( inFile, mat='T3', azlks=5,rglks=5, 
               fmt='tif',
               cog=False,ovr = [2, 4, 8, 16],comp=False,
               out_dir=None,recip=False,
               max_workers=None, 
                calibration_constant = 42
              ):
    """
    Extracts PolSAR matrix elements from a ISRO ASAR RSLC HDF5 file and saves them as slant range raster files.

    This function supports both dual-polarimetric and full-polarimetric RSLC data. It performs multi-looking
    using specified azimuth and range looks, and outputs matrix elements in either GeoTIFF or binary format.
    Optional support for Cloud Optimized GeoTIFFs (COGs), and TIFF compression.

    Examples
    --------
    >>> isro_asar("path_to_file.h5", azlks=30, rglks=15)
    Extracts matrix elements with 5x5 multi-looking and saves them in the default output folder.

    >>> isro_asar("path_to_file.h5", mat='T3', fmt='tif', cog=True, comp=True)
    Extracts T3 matrix elements and saves them as compressed Cloud Optimized GeoTIFFs.

    Parameters
    ----------
    inFile : str
        Path to the NISAR RSLC HDF5 file containing dual-pol or full-pol SAR data.

    mat : str, optional (default='T3')
        Type of matrix to extract. Valid options for Full-pol: 'S2',  'C4, 'C3', 'T4', 
        'T3', 'C2HX', 'C2VX', 'C2HV','T2HV'and Dual-pol: 'Sxy','C2'.

    azlks : int, optional (default=5)
        Number of azimuth looks for multi-looking.

    rglks : int, optional (default=5)
        Number of range looks for multi-looking.

    fmt : {'tif', 'bin'}, optional (default='tif')
        Output format:
        - 'tif': GeoTIFF in slant range
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
    
    geocode_flag=False
    cc_linear = np.sqrt(10**(calibration_constant/10))
    
    
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
        None, None, listOfPolarizations, out_dir,cc_linear)   

    elif nchannels==4:
        nisar_fp(mat, inFile, inFolder, base_path, azlks, rglks, recip, max_workers,
        start_x, start_y, xres, yres, projection, fmt, cog, ovr, comp,
        None, None, out_dir,cc_linear)
        