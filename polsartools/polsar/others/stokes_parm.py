import os
import numpy as np
from polsartools.utils.proc_utils import process_chunks_parallel
from polsartools.utils.utils import conv2d,time_it


@time_it
def stokes_parm(SxxFile,SxyFile,  win=3, fmt="tif", cog=False, 
          ovr = [2, 4, 8, 16], comp=False,
          max_workers=None,block_size=(512, 512),
          progress_callback=None,  # for QGIS plugin
          ):

    """

    This function computes the Stokes parameters and child parameters from Single look Scattering Matrix elements.

    Examples
    --------
    >>> # Basic usage with default parameters
    >>> stokes_parm("/path/to/s11.bin", "/path/to/s21.bin")
    
    >>> # Advanced usage with custom parameters
    >>> stokes_parm(
    ...     "/path/to/s11.bin",
    ...     "/path/to/s21.bin",
    ...     win=5,
    ...     fmt="tif",
    ...     cog=True,
    ...     block_size=(1024, 1024)
    ... )
    
    Parameters
    ----------
    SxxFile : str
        Path to the Sxx file.
    SxyFile : str
        Path to the Sxy file.
    win : int, default=3
        Size of the spatial averaging window. 
    fmt : {'tif', 'bin'}, default='tif'
        Output file format:
        - 'tif': GeoTIFF format 
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
    
    Writes stkokes and child parameters to disk:
    
        - stokes_g0 (.bin or .tif)
        - stokes_g1 (.bin or .tif)
        - stokes_g2 (.bin or .tif)
        - stokes_g3 (.bin or .tif)
        - stokes_e1 (.bin or .tif)
        - stokes_e2 (.bin or .tif)
        - stokes_e1norm (.bin or .tif)
        - stokes_e2norm (.bin or .tif)
        - stokes_phi (.bin or .tif)
        - stokes_tau (.bin or .tif)
        - stokes_x_poincare (.bin or .tif)
        - stokes_y_poincare (.bin or .tif)
        - stokes_H (.bin or .tif)
        - stokes_A (.bin or .tif)
        - stokes_contrast (.bin or .tif)
        - stokes_DoL (.bin or .tif)
        - stokes_DoCP (.bin or .tif)
        - stokes_LPR (.bin or .tif)
        - stokes_CPR (.bin or .tif)

    """
    write_flag=True
    input_filepaths = [SxxFile,SxyFile]
    infolder = os.path.dirname(SxxFile)
    
    output_filepaths = []
    if fmt == "bin":
        output_filepaths.append(os.path.join(infolder, "stokes_g0.bin"))
        output_filepaths.append(os.path.join(infolder, "stokes_g1.bin"))
        output_filepaths.append(os.path.join(infolder, "stokes_g2.bin"))
        output_filepaths.append(os.path.join(infolder, "stokes_g3.bin"))
        
        output_filepaths.append(os.path.join(infolder, "stokes_e1.bin"))
        output_filepaths.append(os.path.join(infolder, "stokes_e2.bin"))
        output_filepaths.append(os.path.join(infolder, "stokes_e1norm.bin"))
        output_filepaths.append(os.path.join(infolder, "stokes_e2norm.bin"))
        
        output_filepaths.append(os.path.join(infolder, "stokes_phi.bin"))
        output_filepaths.append(os.path.join(infolder, "stokes_tau.bin"))
        output_filepaths.append(os.path.join(infolder, "stokes_x_poincare.bin"))
        output_filepaths.append(os.path.join(infolder, "stokes_y_poincare.bin"))
        
        output_filepaths.append(os.path.join(infolder, "stokes_H.bin"))
        output_filepaths.append(os.path.join(infolder, "stokes_A.bin"))
        output_filepaths.append(os.path.join(infolder, "stokes_contrast.bin"))
        
        output_filepaths.append(os.path.join(infolder, "stokes_DoLP.bin"))
        output_filepaths.append(os.path.join(infolder, "stokes_DoCP.bin"))
        output_filepaths.append(os.path.join(infolder, "stokes_LPR.bin"))
        output_filepaths.append(os.path.join(infolder, "stokes_CPR.bin"))        
        
    else:
        output_filepaths.append(os.path.join(infolder, "stokes_g0.tif"))
        output_filepaths.append(os.path.join(infolder, "stokes_g1.tif"))
        output_filepaths.append(os.path.join(infolder, "stokes_g2.tif"))
        output_filepaths.append(os.path.join(infolder, "stokes_g3.tif"))
        
        output_filepaths.append(os.path.join(infolder, "stokes_e1.tif"))
        output_filepaths.append(os.path.join(infolder, "stokes_e2.tif"))
        output_filepaths.append(os.path.join(infolder, "stokes_e1norm.tif"))
        output_filepaths.append(os.path.join(infolder, "stokes_e2norm.tif"))
        
        output_filepaths.append(os.path.join(infolder, "stokes_phi.tif"))
        output_filepaths.append(os.path.join(infolder, "stokes_tau.tif"))
        output_filepaths.append(os.path.join(infolder, "stokes_x_poincare.tif"))
        output_filepaths.append(os.path.join(infolder, "stokes_y_poincare.tif"))
        
        output_filepaths.append(os.path.join(infolder, "stokes_H.tif"))
        output_filepaths.append(os.path.join(infolder, "stokes_A.tif"))
        output_filepaths.append(os.path.join(infolder, "stokes_contrast.tif"))
        
        output_filepaths.append(os.path.join(infolder, "stokes_DoLP.tif"))
        output_filepaths.append(os.path.join(infolder, "stokes_DoCP.tif"))
        output_filepaths.append(os.path.join(infolder, "stokes_LPR.tif"))
        output_filepaths.append(os.path.join(infolder, "stokes_CPR.tif"))   
        
        
        
        
    process_chunks_parallel(input_filepaths, list(output_filepaths), 
                            window_size=win, write_flag=write_flag,
                        processing_func=process_chunk_stokes,block_size=block_size, 
                        max_workers=max_workers,  num_outputs=len(output_filepaths),
                        cog=cog,                         ovr=ovr,                         comp=comp,
                        progress_callback=progress_callback
                        )

def process_chunk_stokes(chunks, window_size,input_filepaths,*args):
    
    kernel = np.ones((window_size,window_size),np.float32)/(window_size*window_size)

    s11 = chunks[0]
    s21 = chunks[1]
    
    
    C11 = (np.abs(s11)**2).astype(np.float32)
    C22 = (np.abs(s21)**2).astype(np.float32)

    C11 = conv2d(C11,kernel)
    C22 = conv2d(C22,kernel)

    C12 = (s11*np.conjugate(s21)).astype(np.complex64)
    C12 = conv2d(C12.real,kernel)+1j*conv2d(C12.imag,kernel)


    g0 = C11+C22
    g1 = C11-C22
    g2 = 2*C12.real
    g3 = -2*C12.imag

    e1 = 0.5*(g0+np.sqrt(g1*g1+g2*g2+g3*g3))
    e2 = 0.5*(g0-np.sqrt(g1*g1+g2*g2+g3*g3))
    e1norm = e1/(e1+e2)
    e2norm = e2/(e1+e2)

    phi = 0.5*np.arctan2(g2,g1)
    tau = 0.5*np.arcsin(g3/(np.sqrt(g1*g1+g2*g2+g3*g3)))

    pb = np.arccos(np.cos(phi)*np.cos(2*tau))
    pc = np.sqrt(2.)*np.sin(pb/2.)
    pd = np.arcsin(np.sin(2*tau)/np.sin(pb))

    check_phi = np.where(phi < 0, -1.0, 1.0)
    Xp = 2*pc*np.cos(pd)*check_phi
    Yp = pc*np.sin(pd)
    
    H = -(e1norm*np.log(e1norm)+e2norm*np.log(e2norm))/np.log(2)
    A = (e1norm-e2norm)/(e1norm+e2norm)
    Co = g1/g0
    
    DoLP = np.sqrt(g1*g1+g2*g2)/(g0)
    DoCP = g3/g0
    LPR = (g0-g1)/(g0+g1)
    CPR = (g0-g3)/(g0+g3)
    

    return g0.astype(np.float32),g1.astype(np.float32),g2.astype(np.float32),g3.astype(np.float32),\
        e1.astype(np.float32),e2.astype(np.float32),e1norm.astype(np.float32),e2norm.astype(np.float32),\
            (phi*180/np.pi).astype(np.float32),(tau*180/np.pi).astype(np.float32),Xp.astype(np.float32),Yp.astype(np.float32),\
                H.astype(np.float32),A.astype(np.float32),Co.astype(np.float32),\
                    DoLP.astype(np.float32),DoCP.astype(np.float32),LPR.astype(np.float32),CPR.astype(np.float32)