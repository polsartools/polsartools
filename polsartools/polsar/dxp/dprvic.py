import os
import numpy as np
from polsartools.utils.proc_utils import process_chunks_parallel
from polsartools.utils.utils import conv2d,time_it,eig22
from .dxp_infiles import dxpc2files
@time_it
def dprvic(cpFile,xpFile,  win=1, fmt="tif", 
           cog=False, ovr = [2, 4, 8, 16], comp=False,
           max_workers=None,block_size=(512, 512),
           progress_callback=None,  # for QGIS plugin          
           ):
    """Compute dual-pol Radar Vegetation Index (DpRVIc) from Dual-pol GRD data.

    This function calculates the DpRVIc using co-polarized (`cpFile`) and cross-polarized (`xpFile`) SAR raster files. 
    The index is valuable for vegetation monitoring and biomass estimation. Processing is performed in parallel blocks 
    for efficiency.

    Examples
    --------
    >>> # Basic usage with default parameters
    >>> dprvic("/path/to/copol_file.tif", "/path/to/crosspol_file.tif")

    >>> # Advanced usage with custom parameters
    >>> dprvic(
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
        Results are written to disk as either 'dprvic.tif' or 'dprvic.bin'
        in the input folder.

    """
    write_flag=True
    input_filepaths = [cpFile,xpFile]
    output_filepaths = []

    if fmt == "bin":
        output_filepaths.append(os.path.join(os.path.dirname(cpFile), "dprvic.bin"))
    else:
        output_filepaths.append(os.path.join(os.path.dirname(cpFile), "dprvic.tif"))

    process_chunks_parallel(input_filepaths, list(output_filepaths), window_size=win, write_flag=write_flag,
                            processing_func=process_chunk_dprvic,block_size=block_size, max_workers=max_workers,  num_outputs=1,
                            cog=cog,ovr=ovr, comp=comp,
                            progress_callback=progress_callback
                            )
    
def process_chunk_dprvic(chunks, window_size,*args):
    kernel = np.ones((window_size,window_size),np.float32)/(window_size*window_size)
    c11 = np.array(chunks[0])
    c22 = np.array(chunks[1])


    if window_size>1:
        c11s = conv2d(c11,kernel)
        c22s = conv2d(c22,kernel)

        q = c22s/c11s
        q[q>=1]=1
        # mc = (1-q)/(1+q)
        # p1 = 1/(1+q)
        # p2 = q/(1+q)
        # Hc = -1*(p1*np.log2(p1)+p2*np.log2(p2))
        # thetac = np.arctan(((1-q)**2)/(1-q+q**2)) * (180/np.pi)
        DpRVIc = (q*(q+3)/((q+1)**2))
    else:
        q = c22/c11
        q[q>=1]=1
        # mc = (1-q)/(1+q)
        # p1 = 1/(1+q)
        # p2 = q/(1+q)
        # Hc = -1*(p1*np.log2(p1)+p2*np.log2(p2))
        # thetac = np.arctan(((1-q)**2)/(1-q+q**2)) * (180/np.pi)
        DpRVIc = (q*(q+3)/((q+1)**2))

    return DpRVIc.astype(np.float32)