import os
import numpy as np
from polsartools.utils.proc_utils import process_chunks_parallel
from polsartools.utils.utils import conv2d,time_it
from polsartools.utils.convert_matrices import T3_C3_mat
from .fp_infiles import fp_c3t3files
@time_it
def freeman_3c(in_dir,  win=1, fmt="tif", cog=False, 
          ovr = [2, 4, 8, 16], comp=False, 
          max_workers=None,block_size=(512, 512),
          progress_callback=None,  # for QGIS plugin
          ):
    
    """Perform Freeman-Durdan 3-Component Decomposition for full-pol SAR data.

    This function implements the three-component decomposition for
    full-polarimetric SAR data, decomposing the total scattered power into surface (Ps),
    double-bounce (Pd), and volume (Pv) scattering components.

    Examples
    --------
    >>> # Basic usage with default parameters
    >>> freeman_3c("/path/to/fullpol_data")
    
    >>> # Advanced usage with custom parameters
    >>> freeman_3c(
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
        Writes three output files to disk:
        
        1. Freeman_3c_odd: Surface scattering power component
        2. Freeman_3c_dbl: Double-bounce scattering power component
        3. Freeman_3c_vol: Volume scattering power component
    """
    
    write_flag=True
    input_filepaths = fp_c3t3files(in_dir)

    output_filepaths = []
    if fmt == "bin":
        output_filepaths.append(os.path.join(in_dir, "Freeman_3c_odd.bin"))
        output_filepaths.append(os.path.join(in_dir, "Freeman_3c_dbl.bin"))
        output_filepaths.append(os.path.join(in_dir, "Freeman_3c_vol.bin"))
    else:
        output_filepaths.append(os.path.join(in_dir, "Freeman_3c_odd.tif"))
        output_filepaths.append(os.path.join(in_dir, "Freeman_3c_dbl.tif"))
        output_filepaths.append(os.path.join(in_dir, "Freeman_3c_vol.tif"))
        
    process_chunks_parallel(input_filepaths, list(output_filepaths), 
                            window_size=win, write_flag=write_flag,
                        processing_func=process_chunk_free3c,block_size=block_size, 
                        max_workers=max_workers,  num_outputs=len(output_filepaths),
                        cog=cog,ovr=ovr,comp=comp,
                        progress_callback=progress_callback
                        )

def process_chunk_free3c(chunks, window_size, input_filepaths, *args):

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


    # Copy inputs to avoid in-place corruption
    C11_ = C11.copy()
    C33_ = C33.copy()
    C13_re_ = C13_re.copy()
    FV = 3. * C22 / 2.

    # Subtract volume component
    C11_ -= FV
    C33_ -= FV
    C13_re_ -= FV / 3.

    # Initialize outputs
    FD = np.zeros_like(C11_)
    FS = np.zeros_like(C11_)
    ALP = np.zeros_like(C11_)
    BET = np.zeros_like(C11_)
    odd = np.zeros_like(C11_)
    dbl = np.zeros_like(C11_)
    vol = np.zeros_like(C11_)

    # Mask for low power (volume dominates)
    mask_eps = (C11_ <= eps) | (C33_ <= eps)
    FV_eps = 3. * (C11_[mask_eps] + C22[mask_eps] + C33_[mask_eps] + 2 * FV[mask_eps]) / 8.
    FV[mask_eps] = FV_eps  # Update FV only for masked pixels

    # Data conditioning
    rtemp = C13_re_**2 + C13_im**2
    mask_non_realizable = rtemp > (C11_ * C33_)
    scale = np.sqrt((C11_ * C33_) / rtemp)
    scale[~mask_non_realizable] = 1.0  # Avoid modifying valid pixels
    C13_re_ *= scale
    C13_im *= scale

    # Bounce classification
    mask_odd = (C13_re_ >= 0.) & ~mask_eps
    mask_even = (C13_re_ < 0.) & ~mask_eps

    # Odd Bounce
    FD[mask_odd] = (C11_[mask_odd] * C33_[mask_odd] - C13_re_[mask_odd]**2 - C13_im[mask_odd]**2) / \
                (C11_[mask_odd] + C33_[mask_odd] + 2 * C13_re_[mask_odd])
    FS[mask_odd] = C33_[mask_odd] - FD[mask_odd]
    ALP[mask_odd] = -1.
    BET[mask_odd] = np.sqrt((FD[mask_odd] + C13_re_[mask_odd])**2 + C13_im[mask_odd]**2) / FS[mask_odd]

    # Even Bounce
    FS[mask_even] = (C11_[mask_even] * C33_[mask_even] - C13_re_[mask_even]**2 - C13_im[mask_even]**2) / \
                    (C11_[mask_even] + C33_[mask_even] - 2 * C13_re_[mask_even])
    FD[mask_even] = C33_[mask_even] - FS[mask_even]
    BET[mask_even] = 1.
    FD_safe = np.where(FD[mask_even] <= eps, eps, FD[mask_even])
    ALP[mask_even] = np.sqrt((FS[mask_even] - C13_re_[mask_even])**2 + C13_im[mask_even]**2) / FD_safe


    odd = FS * (1 + BET**2)
    dbl = FD * (1 + ALP**2)
    vol = 8. * FV / 3.
    odd = np.clip(odd, 0, SpanMax)
    dbl = np.clip(dbl, 0, SpanMax)
    vol = np.clip(vol, 0, SpanMax)
    
    zero_mask = (odd == 0) & (dbl == 0) & (vol == 0)
    odd = odd.astype(np.float32)
    dbl = dbl.astype(np.float32)
    vol = vol.astype(np.float32)

    odd[zero_mask] = np.nan
    dbl[zero_mask] = np.nan
    vol[zero_mask] = np.nan
    
    return odd, dbl, vol