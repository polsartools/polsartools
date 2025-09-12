import os
import numpy as np
from polsartools.utils.proc_utils import process_chunks_parallel
from polsartools.utils.utils import conv2d,time_it
from .dcp_infiles import dcpt2files
@time_it
def mf3cd(in_dir,  win=1, fmt="tif", 
          cog=False, ovr = [2, 4, 8, 16], comp=False,
          max_workers=None,block_size=(512, 512),
          progress_callback=None,  # for QGIS plugin          
          ):
    """
    
    Computes decomposed powers, and a scattering type parameter from the input dual-co-polarization T2 matrix data, and writes
    the output in various formats (GeoTIFF or binary). The computation is performed in parallel for efficiency.

    Example:
    --------
    >>> mf3cd("path_to_T2_folder", win=5, fmt="tif", cog=True)
    This will compute decomposed powers from the dual-co-pol T2 matrix in the specified folder,
    generating output in Geotiff format with Cloud Optimized GeoTIFF settings enabled.
    
    Parameters:
    -----------
    in_dir : str
        Path to the input folder containing T2 matrix data.
    win : int, optional
        Size of the processing window (default is 1).
    fmt : str, optional
        Output format of the files; can be "tif" (GeoTIFF) or "bin" (binary) (default is "tif").
    cog : bool, optional
        If True, outputs Cloud Optimized GeoTIFF (COG) (default is False).
    ovr : list of int, optional
        List of overview levels to be used for COGs (default is [2, 4, 8, 16]).
    comp : bool, optional
        If True, applies LZW compression to GeoTIFF outputs (default is False).
    max_workers : int, optional
        Number of parallel workers for processing (default is None, which uses one less than the number of available CPU cores).
    block_size : tuple of int, optional
        Size of each processing block (default is (512, 512)), defining the spatial chunk dimensions used in parallel computation.
   
    Returns:
    --------
    None
        The function writes the computed decomposed powers to the specified output format.

    Output Files:
    -------------
    - "Ps_mf3cd.tif" or "Ps_mf3cd.bin"
    - "Pd_mf3cd.tif" or "Pd_mf3cd.bin"
    - "Pv_mf3cd.tif" or "Pv_mf3cd.bin"
    - "Theta_DP_mf3cd.tif" or "Theta_DP_mf3cd.bin"


    """
    write_flag=True
    input_filepaths = dcpt2files(in_dir)
    output_filepaths = []
    if fmt == "bin":
        output_filepaths.append(os.path.join(in_dir, "Ps_mf3cd.bin"))
        output_filepaths.append(os.path.join(in_dir, "Pd_mf3cd.bin"))
        output_filepaths.append(os.path.join(in_dir, "Pv_mf3cd.bin"))
        output_filepaths.append(os.path.join(in_dir, "Theta_DP_mf3cd.bin"))
    else:
        output_filepaths.append(os.path.join(in_dir, "Ps_mf3cd.tif"))
        output_filepaths.append(os.path.join(in_dir, "Pd_mf3cd.tif"))
        output_filepaths.append(os.path.join(in_dir, "Pv_mf3cd.tif"))
        output_filepaths.append(os.path.join(in_dir, "Theta_DP_mf3cd.tif"))

    process_chunks_parallel(input_filepaths, list(output_filepaths), 
                            window_size=win, write_flag=write_flag,
                            processing_func=process_chunk_mf3cd,
                            block_size=block_size, max_workers=max_workers,  
                            num_outputs=len(output_filepaths),
                            cog=cog, ovr=ovr, comp=comp,
                            progress_callback=progress_callback
                            )

def process_chunk_mf3cd(chunks, window_size,*args):
    kernel = np.ones((window_size,window_size),np.float32)/(window_size*window_size)
    
    t11_T1 = np.array(chunks[0])
    t12_T1 = np.array(chunks[1])+1j*np.array(chunks[2])
    t21_T1 = np.conj(t12_T1)
    t22_T1 = np.array(chunks[3])


    if window_size>1:
        t11s = conv2d(np.real(t11_T1),kernel)+1j*conv2d(np.imag(t11_T1),kernel)
        t12s = conv2d(np.real(t12_T1),kernel)+1j*conv2d(np.imag(t12_T1),kernel)
        t21s = np.conj(t12_T1)
        t22s = conv2d(np.real(t22_T1),kernel)+1j*conv2d(np.imag(t22_T1),kernel)
    
    else:
        t11s = t11_T1
        t12s = t12_T1
        t21s = t21_T1
        t22s = t22_T1
    
    det_T2 = t11s*t22s-t12s*t21s
    trace_T2 = t11s + t22s

    m1 = np.real(np.sqrt(1-(4*(det_T2/(trace_T2**2)))))
    h = (t11s - t22s)
    g = t22s
    span = t11s + t22s

    
    val = (m1*span*h)/(t11s*g+m1**2*span**2);
    thet = np.real(np.arctan(val))
    # thet = np.rad2deg(thet)
    theta_DP = np.rad2deg(thet)

    
    Ps_DP = (((m1*(span)*(1+np.sin(2*thet))/2)))
    Pd_DP = (((m1*(span)*(1-np.sin(2*thet))/2)))
    Pv_DP = (span*(1-m1))


    return np.real(Ps_DP),np.real(Pd_DP),np.real(Pv_DP),np.real(theta_DP)