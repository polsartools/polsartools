import os
import numpy as np
from polsartools.utils.proc_utils import process_chunks_parallel
from polsartools.utils.utils import conv2d,time_it
from .dxp_infiles import dxpc2files
@time_it
def h_alpha_dp(in_dir,  win=1, fmt="tif", 
             cog=False, ovr = [2, 4, 8, 16], comp=False,
             max_workers=None,block_size=(512, 512),
             progress_callback=None,  # for QGIS plugin          
                ):
    """
    
    Computes Entropy and alpha parameters from the input dual-polarization (dual-pol) C2 matrix data, and writes
    the output in various formats (GeoTIFF or binary). The computation is performed in parallel for efficiency.

    Example:
    --------
    >>> h_alpha_dp("path_to_C2_folder", win=5, fmt="tif", cog=True)
    This will compute Entropy and alpha parameters from the C2 matrix in the specified folder,
    generating output in Geotiff format with Cloud Optimized GeoTIFF settings enabled.
    
    Parameters:
    -----------
    in_dir : str
        Path to the input folder containing C2 matrix data.
    win : int, optional
        Size of the processing window (default is 1).
    fmt : str, optional
        Output format of the files; can be "tif" (GeoTIFF) or "bin" (binary) (default is "tif").
    cog : bool, optional
        If True, outputs Cloud Optimized GeoTIFF (COG) (default is False).
    ovr : list of int, optional
        List of overview levels to be used for COGs (default is [2, 4, 8, 16]).
    comp : bool, optional
        If True, applies LZW compression to the output GeoTIFF files. (default is False).
    max_workers : int, optional
        Number of parallel workers for processing (default is None, which uses one less than the number of available CPU cores).
    block_size : tuple of int, optional
        Size of each processing block (default is (512, 512)), defining the spatial chunk dimensions used in parallel computation.
   
    Returns:
    --------
    None
        The function writes the computed entropy, alpha parameters to the specified output format.

    Output Files:
    -------------
    - "Hdp.tif" or "Hdp.bin"
    - "alphadp.tif" or "alphadp.bin"

    """
    write_flag=True
    input_filepaths = dxpc2files(in_dir)
    output_filepaths = []
    if fmt == "bin":
        output_filepaths.append(os.path.join(in_dir, "Hdp.bin"))
        output_filepaths.append(os.path.join(in_dir, "alphadp.bin"))
        output_filepaths.append(os.path.join(in_dir, "e1_norm.bin"))
        output_filepaths.append(os.path.join(in_dir, "e2_norm.bin"))
        
    else:
        output_filepaths.append(os.path.join(in_dir, "Hdp.tif"))
        output_filepaths.append(os.path.join(in_dir, "alphadp.tif"))
        output_filepaths.append(os.path.join(in_dir, "e1_norm.tif"))
        output_filepaths.append(os.path.join(in_dir, "e2_norm.tif"))

    process_chunks_parallel(input_filepaths, list(output_filepaths), window_size=win, write_flag=write_flag,
                            processing_func=process_chunk_halphadp,block_size=block_size, max_workers=max_workers,  num_outputs=len(output_filepaths),
                            cog=cog,ovr=ovr, comp=comp,
                            progress_callback=progress_callback
                            )
def process_chunk_halphadp(chunks, window_size,*args):
    kernel = np.ones((window_size,window_size),np.float32)/(window_size*window_size)
    c11_T1 = np.array(chunks[0])
    c12_T1 = np.array(chunks[1])+1j*np.array(chunks[2])
    c21_T1 = np.conj(c12_T1)
    c22_T1 = np.array(chunks[3])

    # C2_stack = np.zeros((np.shape(c11_T1)[0],np.shape(c11_T1)[1],4))
    C2_stack = np.dstack((c11_T1,c12_T1,np.conj(c12_T1),c22_T1)).astype(np.complex64)

    if window_size>1:
        C2_stack[:,:,0] = conv2d(np.real(c11_T1),kernel)+1j*conv2d(np.imag(c11_T1),kernel)
        C2_stack[:,:,1] = conv2d(np.real(c12_T1),kernel)+1j*conv2d(np.imag(c12_T1),kernel)
        C2_stack[:,:,2] = conv2d(np.real(c21_T1),kernel)+1j*conv2d(np.imag(c21_T1),kernel)
        C2_stack[:,:,3] = conv2d(np.real(c22_T1),kernel)+1j*conv2d(np.imag(c22_T1),kernel)

    data = C2_stack.reshape( C2_stack.shape[0]*C2_stack.shape[1], C2_stack.shape[2] ).reshape((-1,2,2))
    # infinity, nan handling
    data = np.nan_to_num(data, nan=0.0, posinf=0, neginf=0)
    # data = np.nan_to_num(data, nan=np.nan, posinf=np.nan, neginf=np.nan)
    
    evals, evecs = np.linalg.eig(data)
    
    
    evals[:,0][evals[:,0] <0] = 0
    evals[:,1][evals[:,1] >1] = 1
    
    eval_norm1 = (evals[:,1])/(evals[:,0] + evals[:,1])
    eval_norm1[eval_norm1<0]=0
    eval_norm1[eval_norm1>1]=1
    
    
    eval_norm2 = (evals[:,0])/(evals[:,0] + evals[:,1])
    eval_norm2[eval_norm2<0]=0
    eval_norm2[eval_norm2>1]=1
    
    # # %Alpha 1
    eig_vec_r1 = np.real(evecs[:,0,1])
    eig_vec_c1 = np.imag(evecs[:,0,1])
    alpha1 = np.arccos(np.sqrt(eig_vec_r1*eig_vec_r1 + eig_vec_c1*eig_vec_c1))*180/np.pi
    
    # # %Alpha 2
    eig_vec_r2 = np.real(evecs[:,0,0])
    eig_vec_c2 = np.imag(evecs[:,0,0])
    alpha2 = np.arccos(np.sqrt(eig_vec_r2*eig_vec_r2 + eig_vec_c2*eig_vec_c2))*180/np.pi
    
    # # %Cloude Alpha
    alpha_ = (eval_norm1*alpha1 + eval_norm2*alpha2)
    alpha_ = alpha_.reshape(C2_stack.shape[0],C2_stack.shape[1])
    # # %Entropy
    H = - eval_norm1*np.log10(eval_norm1)/np.log10(2) - eval_norm2*np.log10(eval_norm2)/np.log10(2)
    H = H.reshape(C2_stack.shape[0],C2_stack.shape[1])

    # alpha1 = alpha1.reshape(C2_stack.shape[0],C2_stack.shape[1])
    # alpha2 = alpha2.reshape(C2_stack.shape[0],C2_stack.shape[1])

    # print(np.nanmean(H),np.nanmean(alpha_))
    eval_norm1 = np.real(eval_norm1.reshape(C2_stack.shape[0],C2_stack.shape[1]))
    eval_norm2 = np.real(eval_norm2.reshape(C2_stack.shape[0],C2_stack.shape[1]))

    return np.real(H).astype(np.float32),np.real(alpha_).astype(np.float32),eval_norm1.astype(np.float32),eval_norm2.astype(np.float32)