import os
import numpy as np
from polsartools.utils.proc_utils import process_chunks_parallel
from polsartools.utils.utils import conv2d,time_it
from polsartools.utils.convert_matrices import T3_C3_mat
from .fp_infiles import fp_c3t3files
@time_it
def freeman_2c(in_dir,  win=1, fmt="tif", cog=False, 
          ovr = [2, 4, 8, 16], comp=False,
          max_workers=None,block_size=(512, 512),
          progress_callback=None,  # for QGIS plugin
          ):
    
    """Perform Freeman 2-Component Decomposition for full-pol SAR data.

    This function implements thetwo-component decomposition for
    full-polarimetric SAR data, decomposing the total scattered power into ground (Ps),
    and volume (Pv) scattering components.

    Examples
    --------
    >>> # Basic usage with default parameters
    >>> freeman_2c("/path/to/fullpol_data")
    
    >>> # Advanced usage with custom parameters
    >>> freeman_2c(
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
        Writes two output files to disk:
        
        1. Freeman_2c_grd: Surface scattering power component
        2. Freeman_2c_vol: Volume scattering power component
    """
    
    write_flag=True
    input_filepaths = fp_c3t3files(in_dir)

    output_filepaths = []
    if fmt == "bin":
        output_filepaths.append(os.path.join(in_dir, "Freeman_2c_grd.bin"))
        output_filepaths.append(os.path.join(in_dir, "Freeman_2c_vol.bin"))
    else:
        output_filepaths.append(os.path.join(in_dir, "Freeman_2c_grd.tif"))
        output_filepaths.append(os.path.join(in_dir, "Freeman_2c_vol.tif"))

        
    process_chunks_parallel(input_filepaths, list(output_filepaths), 
                            window_size=win, write_flag=write_flag,
                        processing_func=process_chunk_free2c,block_size=block_size, 
                        max_workers=max_workers,  num_outputs=len(output_filepaths),
                        cog=cog,ovr=ovr,comp=comp,
                        progress_callback=progress_callback
                        )

def process_chunk_free2c(chunks, window_size, input_filepaths, *args):

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

        T3 = np.array([[t11_T1, t12_T1, t13_T1], 
                     [t21_T1, t22_T1, t23_T1], 
                     [t31_T1, t32_T1, t33_T1]])
        T_T1 = T3_C3_mat(T3)


    if 'C11' in input_filepaths[0] and 'C22' in input_filepaths[5] and 'C33' in input_filepaths[8]:
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

    # print("Window size: ",window_size)
    if window_size>1:
        kernel = np.ones((window_size,window_size),np.float32)/(window_size*window_size)
        # print('Filtering with window size: ', window_size)
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
    
    C11 = np.real(T_T1[0,:,:])
    C22 = np.real(T_T1[4,:,:])
    C33 = np.real(T_T1[8,:,:])
    C13_re = np.real(T_T1[2,:,:])
    C13_im = np.imag(T_T1[2,:,:])
    
    del T_T1
    Span = C11+C22+C33
    SpanMax = np.nanmax(Span)
    eps = 1e-10
    
    
    z1 = C11 - C33
    z1 = np.where(np.abs(z1) == 0, eps, z1)

    z2r = C22 + C13_re - C11
    z2i = C13_im

    z3r = z2r / z1
    z3i = z2i / z1

    denom = z3r**2 + z3i**2 + eps
    y = -(z3i * (1.0 + 2.0 * z3r)) / denom
    x = 1.0 + (y * z3r / (z3i + eps))

    denom_fg = 1.0 - x**2 - y**2
    # denom_fg = np.where(denom_fg == 0, eps, denom_fg)
    FG = z1 / denom_fg
    FV = C11 - FG
    RHO = 1.0 - (C22/ (FV + eps))

    vol = FV * (3.0 - RHO)
    grd = FG * (1.0 + x**2 + y**2)

    grd = np.clip(grd, 0.0, SpanMax).astype(np.float32)
    vol = np.clip(vol, 0.0, SpanMax).astype(np.float32)
    
    mask1 = (grd <= eps) & (vol <= eps)
    mask2 = np.isnan(C11) & np.isnan(C22) & np.isnan(C33)
    mask3 = (np.abs(C11) <= eps) & (np.abs(C22) <= eps) & (np.abs(C33) <= eps)
    combined_mask = mask1 | mask2 | mask3

    grd[combined_mask] = np.nan
    vol[combined_mask] = np.nan
    
    
    return grd, vol