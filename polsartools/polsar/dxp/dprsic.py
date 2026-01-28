import os
import numpy as np
from polsartools.utils.proc_utils import process_chunks_parallel
from polsartools.utils.utils import conv2d,time_it,eig22
from .dxp_infiles import dxpc2files
@time_it
def dprsic(cpFile,xpFile,  win=1, fmt="tif", 
           cog=False, ovr = [2, 4, 8, 16], comp=False,
           max_workers=None,block_size=(512, 512),
           progress_callback=None,  # for QGIS plugin          
           ):
    """
    This function computes the Dual-pol Radar Surface Index (DpRSI) for GRD (only intensity no phase).


    Examples
    --------
    >>> # Basic usage with default parameters
    >>> dprsic("/path/to/copol_file.tif", "/path/to/crosspol_file.tif")

    >>> # Advanced usage with custom parameters
    >>> dprsic(
    ...     cpFile="/path/to/copol_file.tif",
    ...     xpFile="/path/to/crosspol_file.tif",
    ...     win=3,
    ...     fmt="tif",
    ...     cog=True,
    ...     block_size=(1024, 1024)
    ... )
    
    Parameters
    ----------
    cpFile : str
        Path to the co-polarized backscatter (linear) SAR raster file.
    xpFile : str
        Path to the cross-polarized backscatter (linear) SAR raster file.
    win : int, default=1
        Size of the spatial averaging window. Larger windows reduce speckle noise
        but decrease spatial resolution.
    fmt : {'tif', 'bin'}, default='tif'
        Output file format:
        - 'tif': GeoTIFF format with georeferencing information
        - 'bin': Raw binary format
    cog : bool, default=False
        If True, creates a Cloud Optimized GeoTIFF (COG) with internal tiling
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

    Returns
    -------
    None
        Results are written to disk as either 'DpRSIc.tif' or 'DpRSIc.bin'
        in the input folder.

    """
    write_flag=True
    input_filepaths = [cpFile,xpFile]
    output_filepaths = []

    if fmt == "bin":
        output_filepaths.append(os.path.join(os.path.dirname(cpFile), "DpRSIc.bin"))
    else:
        output_filepaths.append(os.path.join(os.path.dirname(cpFile), "DpRSIc.tif"))

    process_chunks_parallel(input_filepaths, list(output_filepaths), window_size=win, write_flag=write_flag,
                            processing_func=process_chunk_dprsic,block_size=block_size, max_workers=max_workers,  num_outputs=1,
                            cog=cog,ovr=ovr, comp=comp,
                            progress_callback=progress_callback
                            )
    
def process_chunk_dprsic(chunks, window_size,*args):
    kernel = np.ones((window_size,window_size),np.float32)/(window_size*window_size)
    c11 = np.array(chunks[0])
    c22 = np.array(chunks[1])


    def S_norm(S_array):
        S_5 = np.nanpercentile(S_array, 5)
        S_95 = np.nanpercentile(S_array, 95)
        S_cln = np.where(S_array > S_95, S_95, S_array)
        S_cln = np.where(S_cln < S_5, S_5, S_cln)
        S_cln_max = np.nanmax(S_cln)
        S_norm_array = np.divide(S_cln,S_cln_max) 
        
        return S_norm_array

    if window_size>1:
        c11 = conv2d(c11,kernel)
        c22 = conv2d(c22,kernel)

    s0 = np.abs(c11 + c22)
    s1 = np.abs(c11 - c22)

    C11_av_db = 10*np.log10(c11)

    prob1 = c11/(c11 + c22)
    prob2 = c22/(c11 + c22)

    ent = -prob1*np.log2(prob1) - prob2*np.log2(prob2)
    s1_s_norm = S_norm(s1) #This is S1 normalzied for DpRSI, does not include slope mask

    dprsi_con1 = (1 - ent)*np.sqrt(1 - np.square(s1_s_norm)); # For Valid pixels
    dprsi_con2 = np.sqrt(1 - np.square(s1_s_norm)); # For Noise pixels 

    NESZ = -16 ## For Sentinel-1
    dprsic = np.where(C11_av_db > NESZ, dprsi_con1, dprsi_con2) 


    return dprsic.astype(np.float32)
