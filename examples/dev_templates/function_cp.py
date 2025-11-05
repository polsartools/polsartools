import os
import numpy as np
from polsartools.utils.proc_utils import process_chunks_parallel
from polsartools.utils.utils import conv2d,time_it
from .cp_infiles import cpc2files
@time_it
def function_cp(in_dir,   chi=45, psi=0, win=1, fmt="tif", cog=False, 
          ovr = [2, 4, 8, 16], comp=False, 
          max_workers=None,block_size=(512, 512),
          progress_callback=None  # for QGIS plugin
          ):
    """
    
    DESCRIPTION OF THE FUNCTION

    Examples
    --------
    >>> # Basic usage with default parameters
    >>> function_cp("/path/to/cp_data")
    
    >>> # Advanced usage with custom parameters
    >>> function_cp(
    ...     in_dir="/path/to/cp_data",
    ...     chi=-45,
    ...     win=5,
    ...     fmt="tif",
    ...     cog=True,
    ...     block_size=(1024, 1024)
    ... )


    Parameters
    ----------
    in_dir : str
        Path to the input folder containing compact-pol C2 matrix files.
    chi : float, default=45
        Ellipticity angle chi of the transmitted wave in degrees.
    psi : float, default=0
        Orientation angle psi of the transmitted wave in degrees.
    win : int, default=1
        Size of the spatial averaging window. Larger windows reduce speckle noise
        but decrease spatial resolution.
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
        If True, applies LZW compression to the output GeoTIFF files.
    max_workers : int | None, default=None
        Maximum number of parallel processing workers. If None, uses
        CPU count - 1 workers.
    block_size : tuple[int, int], default=(512, 512)
        Size of processing blocks (rows, cols) for parallel computation.
        Larger blocks use more memory but may be more efficient.

    Returns:
    --------
    None

    Output Files:
    -------------
    - "output.tif" or "output.bin"

    """
    write_flag=True
    input_filepaths = cpc2files(in_dir)

    output_filepaths = []

    # OUTPUT FILE NAMES AND PATHS
    if fmt == "bin":
        output_filepaths.append(os.path.join(in_dir, "output.bin"))
    else:
        output_filepaths.append(os.path.join(in_dir, "output.tif"))
        
    process_chunks_parallel(input_filepaths, list(output_filepaths), 
                            win,
                        write_flag,
                        process_chunk_mf3cc,
                        *[chi, psi],
                        block_size=block_size, 
                        max_workers=max_workers,  
                        num_outputs=len(output_filepaths),
                        cog=cog, ovr=ovr, comp=comp,
                        progress_callback=progress_callback
                        )
def process_chunk_mf3cc(chunks, window_size, *args, **kwargs):
    
    chi=args[-2]
    psi=args[-1]
    # print(chi,psi):

    kernel = np.ones((window_size,window_size),np.float32)/(window_size*window_size)
    c11_T1 = np.array(chunks[0])
    c12_T1 = np.array(chunks[1])+1j*np.array(chunks[2])
    c21_T1 = np.conj(c12_T1)
    c22_T1 = np.array(chunks[3])

    ncols,nrows = np.shape(c11_T1)

    if window_size>1:
        c11_T1 = conv2d(np.real(c11_T1),kernel)+1j*conv2d(np.imag(c11_T1),kernel)
        c12_T1 = conv2d(np.real(c12_T1),kernel)+1j*conv2d(np.imag(c12_T1),kernel)
        c21_T1 = conv2d(np.real(c21_T1),kernel)+1j*conv2d(np.imag(c21_T1),kernel)
        c22_T1 = conv2d(np.real(c22_T1),kernel)+1j*conv2d(np.imag(c22_T1),kernel)


    data = C2_stack.reshape( C2_stack.shape[0]*C2_stack.shape[1], C2_stack.shape[2] ).reshape((-1,2,2))
   
    ############################################
    # CORE LOGIC OF THE FUNCTION
    ############################################
    out_put = data[:,0]


    return out_put.astype(np.float32)