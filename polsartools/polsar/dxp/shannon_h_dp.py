import os
import numpy as np
from polsartools.utils.proc_utils import process_chunks_parallel
from polsartools.utils.utils import conv2d,time_it
from .dxp_infiles import dxpc2files

"""
normlized shannon entropy parameters are not agreeing with polsarpro
others are fine

"""
@time_it
def shannon_h_dp(in_dir,  win=1, fmt="tif", 
                 cog=False, ovr = [2, 4, 8, 16], 
                 comp=False, max_workers=None,block_size=(512, 512),
                 progress_callback=None,  # for QGIS plugin          
                 ):
    """
    
    Computes Shannon entropy parameter, total entropy, SE, intensity (SEI) and polarimetry (SEP) from the input dual-polarization (dual-pol) C2 matrix data, and writes
    the output in various formats (GeoTIFF or binary). The computation is performed in parallel for efficiency.

    Example:
    --------
    >>> shannon_h_dp("path_to_C2_folder", win=5, fmt="tif", cog=True)
    This will compute Shannon entropy parameters from the C2 matrix in the specified folder,
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
        The function writes the computed entropy parameters to the specified output format.

    Output Files:
    -------------
    - "H_Shannon.tif" or "H_Shannon.bin"
    - "HI_Shannon.tif" or "HI_Shannon.bin"
    - "HP_Shannon.tif" or "HP_Shannon.bin"


    """
    write_flag=True
    input_filepaths = dxpc2files(in_dir)
    
    output_filepaths = []
    if fmt == "bin":
        output_filepaths.append(os.path.join(in_dir, "H_Shannon.bin"))
        output_filepaths.append(os.path.join(in_dir, "HI_Shannon.bin"))
        output_filepaths.append(os.path.join(in_dir, "HP_Shannon.bin"))
    else:
        output_filepaths.append(os.path.join(in_dir, "H_Shannon.tif"))
        output_filepaths.append(os.path.join(in_dir, "HI_Shannon.tif"))
        output_filepaths.append(os.path.join(in_dir, "HP_Shannon.tif"))
        
    process_chunks_parallel(input_filepaths, list(output_filepaths), window_size=win, write_flag=write_flag,
                            processing_func=process_chunk_shannondp,block_size=block_size, max_workers=max_workers,  num_outputs=3,
                            cog=cog,ovr=ovr, comp=comp,
                            progress_callback=progress_callback
                            )

def process_chunk_shannondp(chunks, window_size,*args):
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
    rows, cols,_ = C2_stack.shape
    
    # infinity, nan handling
    data = np.nan_to_num(data, nan=0.0, posinf=0, neginf=0)
    # data = np.nan_to_num(data, nan=np.nan, posinf=np.nan, neginf=np.nan)
    evals, evecs = np.linalg.eig(data)
    
    
    evals[:,0][evals[:,0] <0] = 0
    evals[:,1][evals[:,1] >1] = 1
  
    eps  = 1e-8
    D = evals[:,0]*evals[:,1]
    I = evals[:,0]+evals[:,1]
    
    # Barakat degree of polarization
    DoP = np.ones(rows*cols).astype(np.float32) - 4* D / (I*I + eps)

    HSP = np.zeros(rows*cols).astype(np.float32)
    # HSI = np.zeros(rows*cols).astype(np.float32)
    # HS = np.zeros(rows*cols).astype(np.float32)

    condition = (np.ones(rows*cols) - DoP) < eps
    HSP = np.where(condition, 0, np.log(np.abs(np.ones(rows*cols) - DoP)))
    HSP[np.isinf(HSP)] = np.nan
    HSP[HSP==0] = np.nan
    
    with np.errstate(divide='ignore', invalid='ignore'):
        HSI = 2 * np.log(np.exp(1) * np.pi * I / 2)
        HSI[np.isinf(HSI)] = np.nan
        HSI[HSI==0] = np.nan
    
    HS = np.nansum(np.dstack((HSP, HSI)), 2)

    """ Normalization will not not work as expected if we are processing individual blocks of data. 
    Therefore we will normalize the whole image at the end.
    """
    # HSP_norm = (HSP - np.nanmin(HSP)) / (np.nanmax(HSP) - np.nanmin(HSP))
    # HSI_norm = (HSI - np.nanmin(HSI)) / (np.nanmax(HSI) - np.nanmin(HSI))
    # HS_norm = (HS - np.nanmin(HS)) / (np.nanmax(HS) - np.nanmin(HS))

    

    return np.real(HS).reshape(rows,cols).astype(np.float32),   np.real(HSI).reshape(rows,cols).astype(np.float32),  np.real(HSP).reshape(rows,cols).astype(np.float32) 