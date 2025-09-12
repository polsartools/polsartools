import os
import numpy as np
from polsartools.utils.proc_utils import process_chunks_parallel
from polsartools.utils.utils import conv2d,time_it
from polsartools.utils.convert_matrices import T3_C3_mat
from .fp_infiles import fp_c3t3files
@time_it
def nned_fp(in_dir,  win=1, fmt="tif", cog=False, 
          ovr = [2, 4, 8, 16], comp=False,
          max_workers=None,block_size=(512, 512),
          progress_callback=None,  # for QGIS plugin
          ):
    """Perform Non-Negative Eigenvalue Decomposition (NNED) for full-pol SAR data.

    This function implements the Non-Negative Eigenvalue Decomposition for
    full-polarimetric SAR data, ensuring physically meaningful decomposition results
    by constraining eigenvalues to be non-negative. The method decomposes the total
    power into odd-bounce (surface), double-bounce, and volume scattering components.

    Examples
    --------
    >>> # Basic usage with default parameters
    >>> nned_fp("/path/to/fullpol_data")
    
    >>> # Advanced usage with custom parameters
    >>> nned_fp(
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
        
        1. NNED_odd: Odd-bounce (surface) scattering component
        2. NNED_dbl: Double-bounce scattering component
        3. NNED_vol: Volume scattering component

    Notes
    -----
    The NNED method ensures physical realizability by:
    1. Maintaining non-negative eigenvalues
    2. Preserving total power conservation
    3. Ensuring positive semi-definite coherency matrices


    """
    write_flag=True
    input_filepaths = fp_c3t3files(in_dir)

    output_filepaths = []
    if fmt == "bin":
        output_filepaths.append(os.path.join(in_dir, "NNED_odd.bin"))
        output_filepaths.append(os.path.join(in_dir, "NNED_dbl.bin"))
        output_filepaths.append(os.path.join(in_dir, "NNED_vol.bin"))
    else:
        output_filepaths.append(os.path.join(in_dir, "NNED_odd.tif"))
        output_filepaths.append(os.path.join(in_dir, "NNED_dbl.tif"))
        output_filepaths.append(os.path.join(in_dir, "NNED_vol.tif"))
        
    process_chunks_parallel(input_filepaths, list(output_filepaths), 
                            window_size=win, write_flag=write_flag,
                        processing_func=process_chunk_nnedfp,block_size=block_size, 
                        max_workers=max_workers,  num_outputs=len(output_filepaths),
                        cog=cog, ovr=ovr, comp=comp,
                        progress_callback=progress_callback
                        )

def process_chunk_nnedfp(chunks, window_size, input_filepaths, *args):

    # additional_arg1 = args[0] if len(args) > 0 else None
    # additional_arg2 = args[1] if len(args) > 1 else None

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
    
    # Indices for vectorized access
    i, j = np.indices((rows, cols))

    SpanMax = -float('inf')  
    SpanMin = float('inf')  

    Span_data = np.real(T_T1[0,:,:]) + np.real(T_T1[4,:,:]) + np.real(T_T1[8,:,:])

    # Find the maximum and minimum values in one step
    SpanMax = np.nanmax(Span_data)
    SpanMin = np.nanmin(Span_data)

    # Ensure SpanMin does not go below eps
    SpanMin = np.nanmax([SpanMin, 1e-6])

    # Set all values in Span_data below eps to eps
    Span_data[Span_data < 1e-6] = 1e-6

    epsilon = np.real(T_T1[0,:,:])
    rho_re = np.real(T_T1[2,:,:])
    rho_im = np.imag(T_T1[2,:,:])
    nhu = np.real(T_T1[4,:,:])
    gamma = np.real(T_T1[8,:,:])
    
    # Pre-calculate epsilon_veg, rho_re_veg, rho_im_veg, nhu_veg, gamma_veg
    fv = 3. * nhu / 2.
    epsilon_veg = fv
    rho_re_veg = fv / 3.
    rho_im_veg = 0.
    nhu_veg = 2. * fv / 3.
    gamma_veg = fv
    
    # Calculate z, a, b
    z = epsilon * gamma_veg + epsilon_veg * gamma - 2. * rho_re * rho_re_veg
    a = epsilon_veg * gamma_veg - rho_re_veg * rho_re_veg
    b = epsilon * gamma - rho_re * rho_re - rho_im * rho_im

    # Calculate x1
    x1 = nhu / nhu_veg

    # Handle the case where a == 0 using a mask
    mask_a_zero = a == 0.
    x2 = np.where(mask_a_zero, b / z, (z - np.sqrt(z * z - 4. * a * b)) / (2. * a))
    xmax = np.minimum(x1, x2)
    
    # make sure the array is writable
    epsilon.setflags(write=True)  
    rho_re.setflags(write=True)
    rho_im.setflags(write=True)
    nhu.setflags(write=True)
    gamma.setflags(write=True)

    # Update epsilon, rho_re, rho_im, nhu, gamma
    epsilon -= xmax * epsilon_veg
    rho_re -= xmax * rho_re_veg
    rho_im -= xmax * rho_im_veg
    nhu -= xmax * nhu_veg
    gamma -= xmax * gamma_veg
    

    # Calculate delta, lambda1, lambda2
    delta = (epsilon - gamma) ** 2 + 4. * (rho_re * rho_re + rho_im * rho_im)
    lambda1 = 0.5 * (epsilon + gamma + np.sqrt(delta))
    lambda2 = 0.5 * (epsilon + gamma - np.sqrt(delta))

    # Calculate OMEGA1, OMEGA2
    OMEGA1 = lambda1 * (gamma - epsilon + np.sqrt(delta)) ** 2
    OMEGA1 /= ( (gamma - epsilon + np.sqrt(delta)) ** 2 + 4. * (rho_re * rho_re + rho_im * rho_im))

    OMEGA2 = lambda2 * (gamma - epsilon - np.sqrt(delta)) ** 2
    OMEGA2 /= ((gamma - epsilon - np.sqrt(delta)) ** 2 + 4. * (rho_re * rho_re + rho_im * rho_im))

    # Calculate hh1_re, hh1_im, hh2_re, hh2_im
    hh1_re = 2. * rho_re / (gamma - epsilon + np.sqrt(delta))
    hh1_im = 2. * rho_im / (gamma - epsilon + np.sqrt(delta))

    hh2_re = 2. * rho_re / (gamma - epsilon - np.sqrt(delta))
    hh2_im = 2. * rho_im / (gamma - epsilon - np.sqrt(delta))

    # Define A0A0 and B0pB
    A0A0 = (hh1_re + 1.) ** 2 + hh1_im ** 2
    B0pB = (hh1_re - 1.) ** 2 + hh1_im ** 2
    
    # Apply conditions for ALPre, ALPim, OMEGAodd, BETre, BETim, OMEGAdbl
    mask_A0A0_greater = A0A0 > B0pB
    ALPre = np.where(mask_A0A0_greater, hh1_re, hh2_re)
    ALPim = np.where(mask_A0A0_greater, hh1_im, hh2_im)
    OMEGAodd = np.where(mask_A0A0_greater, OMEGA1, OMEGA2)

    BETre = np.where(mask_A0A0_greater, hh2_re, hh1_re)
    BETim = np.where(mask_A0A0_greater, hh2_im, hh1_im)
    OMEGAdbl = np.where(mask_A0A0_greater, OMEGA2, OMEGA1)

    # Calculate NNED_odd, NNED_dbl, and NNED_vol
    NNED_odd = OMEGAodd * (1 + ALPre**2 + ALPim**2)
    NNED_dbl = OMEGAdbl * (1 + BETre**2 + BETim**2)
    NNED_vol = xmax * (epsilon_veg + nhu_veg + gamma_veg)

    # Apply limits to M_odd, M_dbl, M_vol
    NNED_odd = np.clip(NNED_odd, 0., SpanMax)
    NNED_dbl = np.clip(NNED_dbl, 0., SpanMax)
    NNED_vol = np.clip(NNED_vol, 0., SpanMax)
    
    return NNED_odd.astype(np.float32),NNED_dbl.astype(np.float32),NNED_vol.astype(np.float32)