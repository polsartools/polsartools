import os
import numpy as np
from polsartools.utils.proc_utils import process_chunks_parallel
from polsartools.utils.utils import conv2d,time_it
from .cp_infiles import cpc2files
@time_it
def mf3cc(in_dir,   chi=45, psi=0, win=1, fmt="tif", cog=False, 
          ovr = [2, 4, 8, 16], comp=False, 
          max_workers=None,block_size=(512, 512),
          progress_callback=None  # for QGIS plugin
          ):
    """Perform Model-Free 3-Component Decomposition for compact-pol SAR data.

    This function implements the model-free three-component decomposition for
    compact-polarimetric SAR data, decomposing the total backscattered power into
    surface (Ps), double-bounce (Pd), and volume (Pv) scattering components, along
    with the scattering-type parameter (Theta_CP).

    Examples
    --------
    >>> # Basic usage with default parameters
    >>> mf3cc("/path/to/cp_data")
    
    >>> # Advanced usage with custom parameters
    >>> mf3cc(
    ...     in_dir="/path/to/cp_data",
    ...     chi=-45,
    ...     win=5,
    ...     fmt="tif",
    ...     cog=True,
    ...     block_size=(1024, 1024)
    ... )


    Parameters
    ----------
    in_dir : str
        Path to the input folder containing compact-pol C2 matrix files.
    chi : float, default=45
        Ellipticity angle chi of the transmitted wave in degrees.
        For circular polarization, chi = 45° (right circular) or -45° (left circular).
    psi : float, default=0
        Orientation angle psi of the transmitted wave in degrees.
        For circular polarization, typically 0°.
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
        Writes four output files to disk:
        1. Ps_mf3cc: Surface scattering power component
        2. Pd_mf3cc: Double-bounce scattering power component
        3. Pv_mf3cc: Volume scattering power component
        4. Theta_CP_mf3cc: Scattering-type parameter

    """
    write_flag=True
    input_filepaths = cpc2files(in_dir)

    output_filepaths = []
    if fmt == "bin":
        output_filepaths.append(os.path.join(in_dir, "Ps_mf3cc.bin"))
        output_filepaths.append(os.path.join(in_dir, "Pd_mf3cc.bin"))
        output_filepaths.append(os.path.join(in_dir, "Pv_mf3cc.bin"))
        output_filepaths.append(os.path.join(in_dir, "Theta_CP_mf3cc.bin"))
    else:
        output_filepaths.append(os.path.join(in_dir, "Ps_mf3cc.tif"))
        output_filepaths.append(os.path.join(in_dir, "Pd_mf3cc.tif"))
        output_filepaths.append(os.path.join(in_dir, "Pv_mf3cc.tif"))
        output_filepaths.append(os.path.join(in_dir, "Theta_CP_mf3cc.tif"))
        
    process_chunks_parallel(input_filepaths, list(output_filepaths), 
                            win,
                        write_flag,
                        process_chunk_mf3cc,
                        *[chi, psi],
                        block_size=block_size, 
                        max_workers=max_workers,  
                        num_outputs=len(output_filepaths),
                        cog=cog, ovr=ovr, comp=comp,
                        progress_callback=progress_callback
                        )
def process_chunk_mf3cc(chunks, window_size, *args, **kwargs):
    
    chi=args[-2]
    psi=args[-1]
    # print(chi,psi):

    kernel = np.ones((window_size,window_size),np.float32)/(window_size*window_size)
    c11_T1 = np.array(chunks[0])
    c12_T1 = np.array(chunks[1])+1j*np.array(chunks[2])
    c21_T1 = np.conj(c12_T1)
    c22_T1 = np.array(chunks[3])

    ncols,nrows = np.shape(c11_T1)

    if window_size>1:
        c11_T1 = conv2d(np.real(c11_T1),kernel)+1j*conv2d(np.imag(c11_T1),kernel)
        c12_T1 = conv2d(np.real(c12_T1),kernel)+1j*conv2d(np.imag(c12_T1),kernel)
        c21_T1 = conv2d(np.real(c21_T1),kernel)+1j*conv2d(np.imag(c21_T1),kernel)
        c22_T1 = conv2d(np.real(c22_T1),kernel)+1j*conv2d(np.imag(c22_T1),kernel)
    
    c2_det = (c11_T1*c22_T1-c12_T1*c21_T1)
    c2_trace = c11_T1+c22_T1
    # t2_span = t11s*t22s
    m1 = np.real(np.sqrt(1.0-(4.0*c2_det/np.power(c2_trace,2))))

    # Compute Stokes parameters
    s0 = c11_T1 + c22_T1
    s1 = c11_T1 - c22_T1
    s2 = np.real(c12_T1 + c21_T1)
    s3 = np.where(chi >= 0, 1j * (c12_T1 - c21_T1), -1j * (c12_T1 - c21_T1))
    s3 = np.real(s3)

    SC = ((s0)-(s3))/2;
    OC = ((s0)+(s3))/2;

    h = (OC-SC)
    # span = c11s + c22s

    val = ((m1*s0*h))/((SC*OC + (m1**2)*(s0**2)))
    thet = np.real(np.arctan(val))
    theta_CP = np.rad2deg(thet)

    Ps_CP= (((m1*(c2_trace)*(1.0+np.sin(2*thet))/2)))
    Pd_CP= (((m1*(c2_trace)*(1.0-np.sin(2*thet))/2)))
    Pv_CP= (c2_trace*(1.0-m1))
    

    return Ps_CP.astype(np.float32), Pd_CP.astype(np.float32), Pv_CP.astype(np.float32), theta_CP.astype(np.float32)