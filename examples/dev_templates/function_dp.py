import os
import numpy as np
from polsartools.utils.proc_utils import process_chunks_parallel
from polsartools.utils.utils import conv2d,time_it
from .dxp_infiles import dxpc2files
@time_it
def function_dp(in_dir,  win=1, fmt="tif", 
             cog=False, ovr = [2, 4, 8, 16], comp=False,
             max_workers=None,block_size=(512, 512),
             progress_callback=None,  # for QGIS plugin          
                ):
    """
    
    DESCRIPTION OF THE FUNCTION

    Example:
    --------
    >>> function_dp("path_to_C2_folder", win=5, fmt="tif", cog=True)

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

    Output Files:
    -------------
    - "output.tif" or "output.bin"


    """
    write_flag=True
    input_filepaths = dxpc2files(in_dir)
    output_filepaths = []
    
    # OUTPUT FILE NAMES AND PATHS
    if fmt == "bin":
        output_filepaths.append(os.path.join(in_dir, "output.bin"))
    else:
        output_filepaths.append(os.path.join(in_dir, "output.tif"))

    process_chunks_parallel(input_filepaths, list(output_filepaths), window_size=win, write_flag=write_flag,
                            processing_func=process_chunk_func,block_size=block_size, max_workers=max_workers,  num_outputs=len(output_filepaths),
                            cog=cog,ovr=ovr, comp=comp,
                            progress_callback=progress_callback
                            )

def process_chunk_func(chunks, window_size,*args):
    kernel = np.ones((window_size,window_size),np.float32)/(window_size*window_size)
    c11_T1 = np.array(chunks[0])
    c12_T1 = np.array(chunks[1])+1j*np.array(chunks[2])
    c21_T1 = np.conj(c12_T1)
    c22_T1 = np.array(chunks[3])

    C2_stack = np.dstack((c11_T1,c12_T1,np.conj(c12_T1),c22_T1)).astype(np.complex64)

    # Filtering elements based on window size
    if window_size>1:
        C2_stack[:,:,0] = conv2d(np.real(c11_T1),kernel)+1j*conv2d(np.imag(c11_T1),kernel)
        C2_stack[:,:,1] = conv2d(np.real(c12_T1),kernel)+1j*conv2d(np.imag(c12_T1),kernel)
        C2_stack[:,:,2] = conv2d(np.real(c21_T1),kernel)+1j*conv2d(np.imag(c21_T1),kernel)
        C2_stack[:,:,3] = conv2d(np.real(c22_T1),kernel)+1j*conv2d(np.imag(c22_T1),kernel)

    data = C2_stack.reshape( C2_stack.shape[0]*C2_stack.shape[1], C2_stack.shape[2] ).reshape((-1,2,2))
   
    ############################################
    # CORE LOGIC OF THE FUNCTION
    ############################################
    out_put = data[:,0]


    return out_put.astype(np.float32)