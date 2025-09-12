import os
import numpy as np
from polsartools.utils.proc_utils import process_chunks_parallel
from polsartools.utils.utils import conv2d,time_it
from .dxp_infiles import dxpc2files
@time_it
def prvi_dp(in_dir,  win=1, fmt="tif", 
           cog=False, ovr = [2, 4, 8, 16], comp=False,
           max_workers=None,block_size=(512, 512),
           progress_callback=None,  # for QGIS plugin          
           ):
    """
        
    Computes polarimetric Radar vegetation index from the input dual-polarization (dual-pol) C2 matrix data, and writes
    the output in various formats (GeoTIFF or binary). The computation is performed in parallel for efficiency.

    Example:
    --------
    >>> prvi_dp("path_to_C2_folder", win=5, fmt="tif", cog=True)
    This will compute polarimetric Radar vegetation index from the C2 matrix in the specified folder,
    generating output in Geotiff format with Cloud Optimized GeoTIFF settings enabled.
    
    Parameters:
    -----------
    in_dir : str
        Path to the input folder containing C2 matrix data.
    win : int, optional
        Size of the processing window (default is 1).
    fmt : str, optional
        Output format of the files; can be "tif" (GeoTIFF) or "bin" (binary) (default is "tif").
    cog : bool, optional
        If True, outputs Cloud Optimized GeoTIFF (COG) (default is False).
    ovr : list of int, optional
        List of overview levels to be used for COGs (default is [2, 4, 8, 16]).
    comp : bool, optional
        If True, applies LZW compression to the output GeoTIFF files. (default is False).
    max_workers : int, optional
        Number of parallel workers for processing (default is None, which uses one less than the number of available CPU cores).
    block_size : tuple of int, optional
        Size of each processing block (default is (512, 512)), defining the spatial chunk dimensions used in parallel computation.

    Returns:
    --------
    None
        The function writes the computed polarimetric Radar vegetation index to the specified output format.

    Output Files:
    -------------
    - "prvidp.tif" or "prvidp.bin"

    """
    write_flag=True
    input_filepaths = dxpc2files(in_dir)
    output_filepaths = []

    if fmt == "bin":
        output_filepaths.append(os.path.join(in_dir, "prvidp.bin"))
    else:
        output_filepaths.append(os.path.join(in_dir, "prvidp.tif"))
       
    process_chunks_parallel(input_filepaths, list(output_filepaths), window_size=win, write_flag=write_flag,
                            processing_func=process_chunk_prvidp,block_size=block_size, max_workers=max_workers,  num_outputs=1,
                            cog=cog,ovr=ovr, comp=comp,
                            progress_callback=progress_callback
                            )


def process_chunk_prvidp(chunks, window_size,*args):
    kernel = np.ones((window_size,window_size),np.float32)/(window_size*window_size)
    c11_T1 = np.array(chunks[0])
    c12_T1 = np.array(chunks[1])+1j*np.array(chunks[2])
    c21_T1 = np.conj(c12_T1)
    c22_T1 = np.array(chunks[3])

    if window_size>1:
        c11s = conv2d(np.real(c11_T1),kernel)+1j*conv2d(np.imag(c11_T1),kernel)
        c12s = conv2d(np.real(c12_T1),kernel)+1j*conv2d(np.imag(c12_T1),kernel)
        c21s = conv2d(np.real(c21_T1),kernel)+1j*conv2d(np.imag(c21_T1),kernel)
        c22s = conv2d(np.real(c22_T1),kernel)+1j*conv2d(np.imag(c22_T1),kernel)
        
        c2_det = (c11s*c22s-c12s*c21s)
        c2_trace = c11s+c22s
        dopdp = np.real(np.sqrt(1.0-(4.0*c2_det/np.power(c2_trace,2))))
        prvidp = np.real((1-dopdp)*c22s)
    else:
        c2_det = (c11_T1*c22_T1-c12_T1*c21_T1)
        c2_trace = c11_T1+c22_T1
        dopdp = np.real(np.sqrt(1.0-(4.0*c2_det/np.power(c2_trace,2))))
        prvidp = np.real((1-dopdp)*c22_T1)  

    return prvidp.astype(np.float32)