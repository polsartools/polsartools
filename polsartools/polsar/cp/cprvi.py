import os
import numpy as np
from polsartools.utils.proc_utils import process_chunks_parallel
from polsartools.utils.utils import time_it
from polsartools.cprvicpp import process_chunk_cprvicpp
from .cp_infiles import cpc2files
@time_it
def cprvi(in_dir,   chi=45, psi=0, win=1, fmt="tif", cog=False, 
          ovr = [2, 4, 8, 16], comp=False, 
          max_workers=None,block_size=(512, 512),
          progress_callback=None,  # for QGIS plugin          
          ):

    """Compute compact-pol Radar Vegetation Index (CpRVI) from C2 matrix data.

    This function processes compact-polarimetric SAR data to generate the CP-RVI, which
    is useful for vegetation monitoring and biomass estimation using compact-pol SAR systems.
    The processing is done in parallel blocks for improved performance.

    Examples
    --------
    >>> # Basic usage with default parameters (right circular transmission)
    >>> cprvi("/path/to/cp_data")
    
    >>> # Custom parameters for left circular transmission
    >>> cprvi(
    ...     in_dir="/path/to/cp_data",
    ...     chi=-45,
    ...     psi=0,
    ...     win=3,
    ...     fmt="tif",
    ...     cog=True
    ... )
    
    Parameters
    ----------
    in_dir : str
        Path to the input folder containing compact-pol C2 matrix files.
    chi : float, default=45
        Ellipticity angle chi of the transmitted wave in degrees.
        For circular polarization, chi = 45° (right circular) or -45° (left circular).
    psi : float, default=0
        Orientation angle psi of the transmitted wave in degrees.
        For circular polarization, typically 0°.
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
        If True, applies LZW compression to the output GeoTIFF.
    max_workers : int | None, default=None
        Maximum number of parallel processing workers. If None, uses
        CPU count - 1 workers.
    block_size : tuple[int, int], default=(512, 512)
        Size of processing blocks (rows, cols) for parallel computation.
        Larger blocks use more memory but may be more efficient.

    Returns
    -------
    None
        Results are written to disk as either 'cprvi.tif' or 'cprvi.bin'
        in the input folder.

    """
    write_flag=True
    input_filepaths = cpc2files(in_dir)
    output_filepaths = []
    if fmt == "bin":
        output_filepaths.append(os.path.join(in_dir, "cprvi.bin"))
    else:
        output_filepaths.append(os.path.join(in_dir, "cprvi.tif"))

    process_chunks_parallel(input_filepaths, list(output_filepaths), 
                            win,
                        write_flag,
                        process_chunk_cprvi,
                        *[chi, psi],
                        block_size=block_size, 
                        max_workers=max_workers,  
                        num_outputs=len(output_filepaths),
                        cog=cog, comp=comp, ovr=ovr,
                        progress_callback=progress_callback

                        )
def process_chunk_cprvi(chunks, window_size, *args, **kwargs):
    
    chi=args[-2]
    psi=args[-1]
    # print(chi,psi)
    
    chunk_arrays = [np.array(ch) for ch in chunks]  
    # CPP function
    vi_c_raw = process_chunk_cprvicpp( chunk_arrays, window_size, chi, psi )

    return np.array(vi_c_raw, copy=True).astype(np.float32) 
    
 