import os
import numpy as np
from polsartools.utils.proc_utils import process_chunks_parallel
from polsartools.utils.utils import conv2d,time_it
from polsartools.utils.convert_matrices import C3_T3_mat
from .fp_infiles import fp_c3t3files
@time_it
def mf4cf(in_dir,  win=1, fmt="tif", cog=False, 
          ovr = [2, 4, 8, 16], comp=False,
          max_workers=None,block_size=(512, 512),
          progress_callback=None,  # for QGIS plugin
          ):
    """Perform Model-Free 4-Component Decomposition for full-pol SAR data.

    This function implements an advanced model-free four-component decomposition for
    full-polarimetric SAR data. It decomposes the total scattered power into surface (Ps),
    double-bounce (Pd), volume (Pv), and helix (Pc) scattering components, along with
    scattering type (Theta_FP) and helicity (Tau_FP) parameters.


    Examples
    --------
    >>> # Basic usage with default parameters
    >>> mf4cf("/path/to/fullpol_data")
    
    >>> # Advanced usage with custom parameters
    >>> mf4cf(
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
        Writes six output files to disk:
        
        1. Ps_mf4cf: Surface scattering power component
        2. Pd_mf4cf: Double-bounce scattering power component
        3. Pv_mf4cf: Volume scattering power component
        4. Pc_mf4cf: Helix scattering power component
        5. Theta_FP_mf4cf: Scattering type parameter
        6. Tau_FP_mf4cf: Helicity parameter

    """
    write_flag=True
    input_filepaths = fp_c3t3files(in_dir)

    output_filepaths = []
    if fmt == "bin":
        output_filepaths.append(os.path.join(in_dir, "Ps_mf4cf.bin"))
        output_filepaths.append(os.path.join(in_dir, "Pd_mf4cf.bin"))
        output_filepaths.append(os.path.join(in_dir, "Pv_mf4cf.bin"))
        output_filepaths.append(os.path.join(in_dir, "Pc_mf4cf.bin"))
        output_filepaths.append(os.path.join(in_dir, "Theta_FP_mf4cf.bin"))
        output_filepaths.append(os.path.join(in_dir, "Tau_FP_mf4cf.bin"))
        
    else:
        output_filepaths.append(os.path.join(in_dir, "Ps_mf4cf.tif"))
        output_filepaths.append(os.path.join(in_dir, "Pd_mf4cf.tif"))
        output_filepaths.append(os.path.join(in_dir, "Pv_mf4cf.tif"))
        output_filepaths.append(os.path.join(in_dir, "Pc_mf4cf.tif"))
        output_filepaths.append(os.path.join(in_dir, "Theta_FP_mf4cf.tif"))
        output_filepaths.append(os.path.join(in_dir, "Tau_FP_mf4cf.tif"))
    

    process_chunks_parallel(input_filepaths, list(output_filepaths), 
                            window_size=win, write_flag=write_flag,
                        processing_func=process_chunk_mf4cf,block_size=block_size, 
                        max_workers=max_workers,  num_outputs=len(output_filepaths),
                        cog=cog, ovr=ovr, comp=comp,
                        progress_callback=progress_callback
                        )

def process_chunk_mf4cf(chunks, window_size, input_filepaths,*args):

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
        C3 = np.array([[C11, C12, C13], 
                         [C21, C22, C23], 
                         [C31, C32, C33]])

        T_T1 = C3_T3_mat(C3)


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

    # det_T3 = t11s*(t22s*t33s-t23s*t32s)-t12s*(t21s*t33s-t23s*t31s)+t13s*(t21s*t32s-t22s*t31s)
    # trace_T3 = t11s + t22s + t33s
    # m1 = np.real(np.sqrt(1-(27*(det_T3/(trace_T3**3)))))

    reshaped_arr = T_T1.reshape(3, 3, -1).transpose(2, 0, 1)
    det_T3 = np.linalg.det(reshaped_arr)
    # del reshaped_arr
    det_T3 = det_T3.reshape(T_T1.shape[2], T_T1.shape[3])

    s0_f = T_T1[0,0,:,:] + T_T1[1,1,:,:] + T_T1[2,2,:,:] #trace_T3
    dop_f = np.real(np.sqrt(1-(27*(det_T3/(s0_f**3)))))

    k11_f = (T_T1[0,0,:,:] + T_T1[1,1,:,:] + T_T1[2,2,:,:])/2
    k44_f = (-T_T1[0,0,:,:] + T_T1[1,1,:,:] + T_T1[2,2,:,:])/2
    k14_f = np.imag(T_T1[1,2,:,:])

    
    # s0_f = trace_T3
    # dop_f = m1

    val1 = (4*dop_f*k11_f*k44_f)/(k44_f**2 - (1 + 4*dop_f**2)*k11_f**2)
    val2 = np.abs(k14_f)/(k11_f)
    


    theta_f = np.real(np.arctan(val1)) # separation for surface and dbl
    tau_f = np.real(np.arctan(val2)) # separation for helix
    # thet = np.rad2deg(thet)
    theta_FP = np.rad2deg(theta_f).astype(np.float32)
    tau_FP = np.rad2deg(tau_f).astype(np.float32)

    pc_f = (dop_f*s0_f*(np.sin(2*tau_f))).astype(np.float32)
    pv_f = ((1-dop_f)*s0_f).astype(np.float32)
    res_pow = s0_f - (pc_f + pv_f)
    ps_f = ((res_pow/2)*(1+np.sin((2*theta_f)))).astype(np.float32)
    pd_f = ((res_pow/2)*(1-np.sin((2*theta_f)))).astype(np.float32)

    return ps_f.astype(np.float32), pd_f.astype(np.float32), pv_f.astype(np.float32),pc_f.astype(np.float32),theta_FP.astype(np.float32),tau_FP.astype(np.float32)