import os
import numpy as np
from polsartools.utils.proc_utils import process_chunks_parallel
from polsartools.utils.utils import conv2d,time_it
from polsartools.utils.convert_matrices import T3_C3_mat, C3_T3_mat
from .fp_infiles import fp_c3t3files
from polsartools.yam4cpp import process_chunk_yam4cpp


@time_it
def yam4c_fp(in_dir,  model="", win=1, fmt="tif", cog=False, 
                    ovr = [2, 4, 8, 16], comp=False, 
                    max_workers=None,block_size=(512, 512),
                    progress_callback=None,  # for QGIS plugin
                        ):
    
    """Perform Yamaguchi 4-Component Decomposition for full-pol SAR data.

    This function implements the Yamaguchi 4-component decomposition with three
    different model options: original (Y4O), rotation-corrected (Y4R), and
    extended volume scattering model (Y4S). The decomposition separates the total
    power into surface, double-bounce, volume, and helix scattering components.

    Examples
    --------
    >>> # Original Yamaguchi decomposition
    >>> yam4cfp("/path/to/fullpol_data")
    
    >>> # Rotation-corrected decomposition
    >>> yam4c_fp(
    ...     in_dir="/path/to/fullpol_data",
    ...     model="y4cr",
    ...     win=5,
    ...     fmt="tif",
    ...     cog=True
    ... )
    
    >>> # Extended volume model decomposition
    >>> yam4c_fp(
    ...     in_dir="/path/to/fullpol_data",
    ...     model="y4cs",
    ...     win=5
    ... )

    Parameters
    ----------
    in_dir : str
        Path to the input folder containing full-pol T3 or C3 matrix files.
    model : {'', 'y4cr', 'y4cs'}, default=''
        Decomposition model to use:
        - '': Original Yamaguchi 4-component (Y4O)
        - 'y4cr': Rotation-corrected Yamaguchi (Y4R)
        - 'y4cs': Extended volume scattering model (Y4S)
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
        Writes four output files to disk for the selected model:
        1. {prefix}_odd: Surface scattering power
        2. {prefix}_dbl: Double-bounce scattering power
        3. {prefix}_vol: Volume scattering power
        4. {prefix}_hlx: Helix scattering power
        where prefix is 'Yam4co', 'Yam4cr', or 'Yam4csr' based on model choice.


    """
    write_flag=True
    input_filepaths = fp_c3t3files(in_dir)

    output_filepaths = []
    if fmt == "bin":
        if not model or model=="y4co":
            output_filepaths.append(os.path.join(in_dir, "Yam4co_odd.bin"))
            output_filepaths.append(os.path.join(in_dir, "Yam4co_dbl.bin"))
            output_filepaths.append(os.path.join(in_dir, "Yam4co_vol.bin"))
            output_filepaths.append(os.path.join(in_dir, "Yam4co_hlx.bin"))
        elif model=="y4cr":
            output_filepaths.append(os.path.join(in_dir, "Yam4cr_odd.bin"))
            output_filepaths.append(os.path.join(in_dir, "Yam4cr_dbl.bin"))
            output_filepaths.append(os.path.join(in_dir, "Yam4cr_vol.bin"))
            output_filepaths.append(os.path.join(in_dir, "Yam4cr_hlx.bin"))
        elif model=="y4cs":
            output_filepaths.append(os.path.join(in_dir, "Yam4csr_odd.bin"))
            output_filepaths.append(os.path.join(in_dir, "Yam4csr_dbl.bin"))
            output_filepaths.append(os.path.join(in_dir, "Yam4csr_vol.bin"))
            output_filepaths.append(os.path.join(in_dir, "Yam4csr_hlx.bin"))
        else:
            raise(f"Invalid model!! \n model type argument must be either '' for default (y4co) or Y4R or S4R")
    
    else:
        if not model or model=="y4co":
            output_filepaths.append(os.path.join(in_dir, "Yam4co_odd.tif"))
            output_filepaths.append(os.path.join(in_dir, "Yam4co_dbl.tif"))
            output_filepaths.append(os.path.join(in_dir, "Yam4co_vol.tif"))
            output_filepaths.append(os.path.join(in_dir, "Yam4co_hlx.tif"))
        elif model=="y4cr":
            output_filepaths.append(os.path.join(in_dir, "Yam4cr_odd.tif"))
            output_filepaths.append(os.path.join(in_dir, "Yam4cr_dbl.tif"))
            output_filepaths.append(os.path.join(in_dir, "Yam4cr_vol.tif"))
            output_filepaths.append(os.path.join(in_dir, "Yam4cr_hlx.tif"))
        elif model=="y4cs":
            output_filepaths.append(os.path.join(in_dir, "Yam4csr_odd.tif"))
            output_filepaths.append(os.path.join(in_dir, "Yam4csr_dbl.tif"))
            output_filepaths.append(os.path.join(in_dir, "Yam4csr_vol.tif"))
            output_filepaths.append(os.path.join(in_dir, "Yam4csr_hlx.tif"))
        else:
            raise(f"Invalid model!! \n model type argument must be either '' for default (y4co) or Y4R or S4R")
            
    
    process_chunks_parallel(input_filepaths, list(output_filepaths), 
                    win, write_flag,
                    process_chunk_yam4cfp,
                    *[model],
                    block_size=block_size, 
                    max_workers=max_workers,  num_outputs=len(output_filepaths),
                    cog=cog, ovr=ovr, comp=comp,
                    progress_callback=progress_callback
                    )

def process_chunk_yam4cfp(chunks, window_size, input_filepaths,  *args, **kwargs):
    model = args[-1]
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

        T_T1 = np.array([[t11_T1, t12_T1, t13_T1], 
                     [t21_T1, t22_T1, t23_T1], 
                     [t31_T1, t32_T1, t33_T1]])
        C3 = T3_C3_mat(T_T1)
        span = C3[0,0].real+C3[1,1].real+C3[2,2].real
        del C3


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
        span = C3[0,0].real+C3[1,1].real+C3[2,2].real
        T_T1 = C3_T3_mat(C3)
        del C3


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
    
    SpanMax = np.nanmax(span)
    SpanMin = np.nanmin(span)
    eps = 1e-6
    SpanMin = np.nanmax([SpanMin, eps])

    
    T_T1 = T_T1.reshape(9, rows, cols)

    chunk_arrays = [T_T1[0,:,:],T_T1[1,:,:],T_T1[2,:,:],
                    T_T1[3,:,:],T_T1[4,:,:],T_T1[5,:,:],T_T1[6,:,:],T_T1[7,:,:],T_T1[8,:,:]]  
    vi_c_raw = process_chunk_yam4cpp(chunk_arrays, window_size, model,SpanMin, SpanMax)

    proc_chunks=[]
    for chunk in vi_c_raw:
        filt_data = np.array(chunk)
        # filt_data[filt_data == 0] = np.nan
        proc_chunks.append(filt_data)
    
    # print(np.shape(proc_chunks))
    return proc_chunks[0].astype(np.float32), \
            proc_chunks[1].astype(np.float32), \
            proc_chunks[2].astype(np.float32), \
            proc_chunks[3].astype(np.float32), 

