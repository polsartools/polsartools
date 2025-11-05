import os
import numpy as np
from polsartools.utils.proc_utils import process_chunks_parallel
from polsartools.utils.utils import conv2d,time_it
from polsartools.utils.convert_matrices import T3_C3_mat
from .fp_infiles import fp_c3t3files
@time_it
def function_fp(in_dir,  win=1, fmt="tif", cog=False, 
          ovr = [2, 4, 8, 16], comp = False,
          max_workers=None,block_size=(512, 512),
            progress_callback=None  # for QGIS plugin
          ):
    """

    DESCRIPTION OF THE FUNCTION

    Examples
    --------
    >>> # Basic usage with default parameters
    >>> function_fp("/path/to/fullpol_data")
    
    >>> # Advanced usage with custom parameters
    >>> function_fp(
    ...     in_dir="/path/to/fullpol_data",
    ...     win=5,
    ...     fmt="tif",
    ...     cog=True,
    ...     block_size=(1024, 1024)
    ... )

    Parameters
    ----------
    in_dir : str
        Path to the input folder containing full-pol T3 or C3 matrix files.
    win : int, default=1
        Size of the spatial averaging window. 
    fmt : {'tif', 'bin'}, default='tif'
        Output file format:
        - 'tif': GeoTIFF format with georeferencing information
        - 'bin': Raw binary format
    cog : bool, default=False
        If True, creates Cloud Optimized GeoTIFF (COG) outputs with internal tiling
        and overviews for efficient web access.
    ovr : list[int], default=[2, 4, 8, 16]
        Overview levels for COG creation. Each number represents the
        decimation factor for that overview level.
    comp : bool, default=False
        If True, uses LZW compression for GeoTIFF outputs.
    max_workers : int | None, default=None
        Maximum number of parallel processing workers. If None, uses
        CPU count - 1 workers.
    block_size : tuple[int, int], default=(512, 512)
        Size of processing blocks (rows, cols) for parallel computation.
        Larger blocks use more memory but may be more efficient.

    Returns
    -------
    None
        Writes six output files to disk:
        1. output




    """
    write_flag=True
    input_filepaths = fp_c3t3files(in_dir)

    output_filepaths = []

    # OUTPUT FILE NAMES AND PATHS
    if fmt == "bin":
        output_filepaths.append(os.path.join(in_dir, "output.bin"))
    else:
        output_filepaths.append(os.path.join(in_dir, "output.tif"))


    process_chunks_parallel(input_filepaths, list(output_filepaths), 
                            window_size=win, write_flag=write_flag,
                        processing_func=process_chunk_func,block_size=block_size, 
                        max_workers=max_workers,  num_outputs=len(output_filepaths),
                        cog=cog,ovr=ovr,comp=comp,                        
                        progress_callback=progress_callback
                        )



def process_chunk_func(chunks, window_size, input_filepaths, *args):

    # convert C3 <-> T3 as required
    if 'T11' in input_filepaths[0] and 'T22' in input_filepaths[5] and 'T33' in input_filepaths[8]:
        t11_T1 = np.array(chunks[0])
        t12_T1 = np.array(chunks[1])+1j*np.array(chunks[2])
        t13_T1 = np.array(chunks[3])+1j*np.array(chunks[4])
        t21_T1 = np.conj(t12_T1)
        t22_T1 = np.array(chunks[5])
        t23_T1 = np.array(chunks[6])+1j*np.array(chunks[7])
        t31_T1 = np.conj(t13_T1)
        t32_T1 = np.conj(t23_T1)
        t33_T1 = np.array(chunks[8])

        T_T1 = np.array([[t11_T1, t12_T1, t13_T1], 
                     [t21_T1, t22_T1, t23_T1], 
                     [t31_T1, t32_T1, t33_T1]])
        # T_T1 = T3_C3_mat(T3)


    elif 'C11' in input_filepaths[0] and 'C22' in input_filepaths[5] and 'C33' in input_filepaths[8]:
        C11 = np.array(chunks[0])
        C12 = np.array(chunks[1])+1j*np.array(chunks[2])
        C13 = np.array(chunks[3])+1j*np.array(chunks[4])
        C21 = np.conj(C12)
        C22 = np.array(chunks[5])
        C23 = np.array(chunks[6])+1j*np.array(chunks[7])
        C31 = np.conj(C13)
        C32 = np.conj(C23)
        C33 = np.array(chunks[8])
        T_T1 = np.array([[C11, C12, C13], 
                         [C21, C22, C23], 
                         [C31, C32, C33]])

    else:
        raise ValueError("Invalid input matrices. Ensure the input is either T3 or C3 matrix foolder.")

    # Filtering elements based on window size
    if window_size>1:
        kernel = np.ones((window_size,window_size),np.float32)/(window_size*window_size)

        t11f = conv2d(T_T1[0,0,:,:],kernel)
        t12f = conv2d(np.real(T_T1[0,1,:,:]),kernel)+1j*conv2d(np.imag(T_T1[0,1,:,:]),kernel)
        t13f = conv2d(np.real(T_T1[0,2,:,:]),kernel)+1j*conv2d(np.imag(T_T1[0,2,:,:]),kernel)
        
        t21f = np.conj(t12f) 
        t22f = conv2d(T_T1[1,1,:,:],kernel)
        t23f = conv2d(np.real(T_T1[1,2,:,:]),kernel)+1j*conv2d(np.imag(T_T1[1,2,:,:]),kernel)

        t31f = np.conj(t13f) 
        t32f = np.conj(t23f) 
        t33f = conv2d(T_T1[2,2,:,:],kernel)

        T_T1 = np.array([[t11f, t12f, t13f], [t21f, t22f, t23f], [t31f, t32f, t33f]])

    _,_,rows,cols = np.shape(T_T1)
    T_T1 = T_T1.reshape(9, rows, cols)
    
    ############################################
    # CORE LOGIC OF THE FUNCTION
    ############################################
    out_put = T_T1[0,:,:]


    return out_put.astype(np.float32)