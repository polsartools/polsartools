import os
import numpy as np
from functools import partial
from polsartools.utils.proc_utils import process_chunks_parallel
from polsartools.utils.utils import conv2d,time_it
from polsartools.utils.convert_matrices import C3_T3_mat
from polsartools.polsar.fp.fp_infiles import fp_c3t3files

@time_it
def div_fp(in_dir,win=1, fmt="tif", cog=False, 
          ovr = [2, 4, 8, 16], comp=False,
          max_workers=None,block_size=(512, 512),
          progress_callback=None,  # for QGIS plugin
          ):
    
    """ Compute multiple diversity indices from full-polarimetric SAR data.

    This function processes full-polarimetric SAR data stored as T3 or C3 covariance/scattering matrix files,
    calculating various diversity measures based on the eigenvalue decomposition of the scattering matrices.
    The computed indices characterize the scattering complexity and heterogeneity of the scene.

    The diversity indices computed include:
    - Shannon entropy
    - Perplexity
    - Simpson index
    - Inverse Simpson index
    - Gini index
    - Rényi entropies (orders 2, 3, and 4)

    The function processes the input data in spatial chunks with optional parallelization and writes the results 
    as individual files to the specified input directory.

    Examples
    --------
    >>> # Basic usage with default parameters
    >>> div_fp("/path/to/fullpol_data")
    
    >>> # Advanced usage with custom parameters
    >>> div_fp(
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
        Size of the spatial averaging window. Larger windows improve DoP estimation
        accuracy but decrease spatial resolution.
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
        Writes eight output files to disk in the format specified by the `fmt` parameter (GeoTIFF by default):
        
        1. shannon.tif / shannon.bin: Shannon entropy index of scattering diversity.
        2. perplex.tif / perplex.bin: Perplexity index (effective number of scattering components).
        3. simpson.tif / simpson.bin: Simpson diversity index.
        4. simpson_inv.tif / simpson_inv.bin: Inverse Simpson index.
        5. gini.tif / gini.bin: Gini index (measure of inequality among scattering components).
        6. reyni_2.tif / reyni_2.bin: Rényi entropy of order 2.
        7. reyni_3.tif / reyni_3.bin: Rényi entropy of order 3.
        8. reyni_4.tif / reyni_4.bin: Rényi entropy of order 4.
    """

    write_flag=True
    input_filepaths = fp_c3t3files(in_dir)
    
    metrics = ["shannon", "perplex", "simpson", "simpson_inv", "gini", "reyni_2", "reyni_3", "reyni_4"]
    output_filepaths = []
    if fmt == "bin":
        output_filepaths = [os.path.join(in_dir, f"{m}.bin") for m in metrics]
    else:
        output_filepaths = [os.path.join(in_dir, f"{m}.tif") for m in metrics]
    
    first_file=input_filepaths[0].upper()
  
    if "T11" in first_file:
        matrix_type = "T3"
    elif "C11" in first_file:
        matrix_type = "C3"
    else:
            raise ValueError("Could not determine matrix type (C3 or T3) from  input files.")
            
    
    process_chunks_parallel(input_filepaths, list(output_filepaths), 
                        window_size=win,write_flag=write_flag,
                        processing_func=partial(process_chunk_diversity, matrix_type=matrix_type), 
                        block_size=block_size, 
                        max_workers=max_workers,  
                        num_outputs=len(output_filepaths),
                        cog=cog, comp=comp, ovr=ovr,
                        progress_callback=progress_callback
                        

                        )
    

def process_chunk_diversity(chunks,window_size,*args,matrix_type=None):

    
    if matrix_type=="T3":
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


    elif matrix_type=="C3" :
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

    Npp = 3 

    eps=1e-10
    h_id, w_id = 2, 3 
    height, width = T_T1.shape[h_id], T_T1.shape[w_id]

    reshaped_arr = T_T1.reshape(Npp, Npp, -1).transpose(2, 0, 1)
    eigenvalues = np.linalg.eigvalsh(reshaped_arr)

    sorted_eigenvalues = np.sort(eigenvalues, axis=1)[:, ::-1]
    sorted_eigenvalues = sorted_eigenvalues.reshape(height, width, Npp)
    sorted_eigenvalues = np.maximum(sorted_eigenvalues, eps)
    span = np.sum(sorted_eigenvalues, axis=-1, keepdims=True)

    p = sorted_eigenvalues / (span + eps)
    p = np.clip(p, eps, 1.0)

    shannon = -np.sum((p * np.log(p)) / np.log(Npp), axis=-1)
    perplex = np.exp(shannon * np.log(Npp))
    
    simpson = np.sum(p * p, axis=-1)
    simpson_safe = np.clip(simpson, eps, 1.0) 
    simpson_inv = 1.0 / simpson_safe
    gini = 1.0 - simpson

    
    reyni_2 = -np.log(np.sum(p**2, axis=-1)) / np.log(Npp)
    reyni_3 = -np.log(np.sum(p**3, axis=-1)) / (2.0 * np.log(Npp))
    reyni_4 = -np.log(np.sum(p**4, axis=-1)) / (3.0 * np.log(Npp))

    return shannon.astype(np.float32),perplex.astype(np.float32),simpson.astype(np.float32),simpson_inv.astype(np.float32),gini.astype(np.float32),reyni_2.astype(np.float32),reyni_3.astype(np.float32),reyni_4.astype(np.float32)
