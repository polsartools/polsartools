import os
import numpy as np
from polsartools.utils.proc_utils import process_chunks_parallel
from polsartools.utils.utils import conv2d,time_it,eig22
from .dxp_infiles import dxpc2files, S_norm
@time_it
def dprsi(in_dir,  win=1, fmt="tif", cog=False, 
          ovr = [2, 4, 8, 16], comp=False,
          max_workers=None,block_size=(512, 512),
          progress_callback=None,  # for QGIS plugin          
          ):
    """ This codes computes the Dual-pol Radar Surface Index (DpRSI) from C2 matrix

    Examples
    --------
    >>> # Basic usage with default parameters
    >>> dprsi("/path/to/c2_data")
    
    >>> # Advanced usage with custom parameters
    >>> dprsi(
    ...     in_dir="/path/to/c2_data",
    ...     win=3,
    ...     fmt="tif",
    ...     cog=True,
    ...     block_size=(1024, 1024)
    ... )
    
    Parameters
    ----------
    in_dir : str
        Path to the input folder containing C2 matrix files.
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
        Results are written to disk as either 'DpRSI.tif' or 'DpRSI.bin'
        in the input folder.

    """
    write_flag=True
    input_filepaths = dxpc2files(in_dir)
    output_filepaths = []

    if fmt == "bin":
        output_filepaths.append(os.path.join(in_dir, "DpRSI.bin"))
    else:
        output_filepaths.append(os.path.join(in_dir, "DpRSI.tif"))

    process_chunks_parallel(input_filepaths, list(output_filepaths), window_size=win, write_flag=write_flag,
                            processing_func=process_chunk_dprsi,block_size=block_size, max_workers=max_workers,  num_outputs=1,
                            cog=cog,ovr=ovr, comp=comp,
                            progress_callback=progress_callback
                            )
    
def process_chunk_dprsi(chunks, window_size,*args):
    kernel = np.ones((window_size,window_size),np.float32)/(window_size*window_size)
    c11_T1 = np.array(chunks[0])
    c12_T1 = np.array(chunks[1])+1j*np.array(chunks[2])
    # c21_T1 = np.conj(c12_T1)
    c22_T1 = np.array(chunks[3])

    # ##### Normalizing Stokes vector elements
    # def S_norm(S_array):
    #     S_5 = np.nanpercentile(S_array, 2)
    #     S_95 = np.nanpercentile(S_array, 98)
    #     S_cln = np.where(S_array > S_95, S_95, S_array)
    #     S_cln = np.where(S_cln < S_5, S_5, S_cln)
    #     S_cln_max = np.nanmax(S_cln)
    #     S_norm_array = np.divide(S_cln,S_cln_max) 
        
    #     return S_norm_array

    if window_size>1:
        c11s = conv2d(np.real(c11_T1),kernel)+1j*conv2d(np.imag(c11_T1),kernel)
        c12s = conv2d(np.real(c12_T1),kernel)+1j*conv2d(np.imag(c12_T1),kernel)
        # c21s = conv2d(np.real(c21_T1),kernel)+1j*conv2d(np.imag(c21_T1),kernel)
        c22s = conv2d(np.real(c22_T1),kernel)+1j*conv2d(np.imag(c22_T1),kernel)

    else:
        c11s = c11_T1
        c12s = c12_T1
        c22s = c22_T1

    C11_av_db = 10*np.log10(c11s)
    s0 = c11s + c22s
    s1 = c11s - c22s
    s2 = 2*c12s.real
    s3 = 2*c12s.imag

    ##### Calculate Entropy
    ## Here eigen values are calculated using Stokes vector elements

    tpp = np.sqrt(np.square(s1) + np.square(s2) + np.square(s3))

    lmbd1 = (s0 + tpp)/2
    lmbd2 = (s0 - tpp)/2

    prob1 = lmbd1/(lmbd1 + lmbd2)
    prob2 = lmbd2/(lmbd1 + lmbd2)

    ent = -prob1*np.log2(prob1) - prob2*np.log2(prob2)

    ##### Taking abs of Stokes vector elements
    s0 = np.abs(s0)
    s1 = np.abs(s1)
    
    s1_s_norm = S_norm(s1) #This is S1 normalzied for DpRSI, does not include slope mask

    ##### Dual-pol Radar Surface Index

    dprsi_con1 = (1 - ent)*np.sqrt(1 - np.square(s1_s_norm)); # For Valid pixels
    dprsi_con2 = np.sqrt(1 - np.square(s1_s_norm)); # For Noise pixels 

    NESZ = -16 ## For Sentinel-1
    dprsi = np.where(C11_av_db > NESZ, dprsi_con1, dprsi_con2) 

    return dprsi.astype(np.float32)
