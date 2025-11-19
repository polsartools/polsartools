import os
import numpy as np
from polsartools.utils.proc_utils import process_chunks_parallel
from polsartools.utils.utils import conv2d,time_it
from .cp_infiles import cpc2files
@time_it
def s_omega(in_dir,   chi=45, psi=0, win=1, fmt="tif", cog=False, 
          ovr = [2, 4, 8, 16], comp=False, 
          max_workers=None,block_size=(512, 512),
          progress_callback=None,  # for QGIS plugin
          ):
    """Perform Modified/Improved S-Omega Decomposition for compact-pol SAR data.

    This function implements an enhanced version of the S-Omega decomposition
    technique for compact-polarimetric SAR data. It decomposes the total
    backscattered power into three components: surface scattering (Ps),
    double-bounce scattering (Pd), and volume scattering (Pv), with improvements
    over the traditional S-Omega method.

    Examples
    --------
    >>> # Basic usage with default parameters
    >>> s_omega("/path/to/cp_data")
    
    >>> # Advanced usage with custom parameters
    >>> s_omega(
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
        Writes three output files to disk:
        1. Ps_miSOmega: Surface scattering power component
        2. Pd_miSOmega: Double-bounce scattering power component
        3. Pv_miSOmega: Volume scattering power component
    """
    write_flag=True
    input_filepaths = cpc2files(in_dir)

    output_filepaths = []
    if fmt == "bin":
        output_filepaths.append(os.path.join(in_dir, "Ps_miSOmega.bin"))
        output_filepaths.append(os.path.join(in_dir, "Pd_miSOmega.bin"))
        output_filepaths.append(os.path.join(in_dir, "Pv_miSOmega.bin"))
    else:
        output_filepaths.append(os.path.join(in_dir, "Ps_miSOmega.tif"))
        output_filepaths.append(os.path.join(in_dir, "Pd_miSOmega.tif"))
        output_filepaths.append(os.path.join(in_dir, "Pv_miSOmega.tif"))
        


    process_chunks_parallel(input_filepaths, list(output_filepaths), 
                            win,
                        write_flag,
                        process_chunk_misomega,
                        *[chi, psi],
                        block_size=block_size, 
                        max_workers=max_workers,  
                        num_outputs=len(output_filepaths),
                        cog=cog, ovr=ovr, comp=comp,
                        progress_callback=progress_callback
                        )
    
def process_chunk_misomega(chunks, window_size, *args, **kwargs):
    
    chi=args[-2]
    psi=args[-1]
    # print(chi,psi)

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
        
    # Compute Stokes parameters
    s0 = np.abs(c11_T1 + c22_T1)
    s1 = np.abs(c11_T1 - c22_T1)
    s2 = np.abs(c12_T1 + c21_T1)
    s3 = np.where(chi >= 0, 1j * (c12_T1 - c21_T1), -1j * (c12_T1 - c21_T1))
    s3 = np.real(s3)

    ## Stokes child parameters
    SC = ((s0)-(s3))/2;
    OC = ((s0)+(s3))/2;
    
    CPR = np.divide(SC,OC)  ##SC/OC
    # CPR = SC/OC
    ##scattered fields    
    dop= np.sqrt(np.power(s1,2) + np.power(s2,2) + np.power(s3,2))/(s0)
    Psi = 0.5*((180/np.pi)*np.arctan2(s2,s1))
    DOCP = (-s3)/(dop*s0);
    Chi = 0.5*((180/np.pi)*np.arcsin(DOCP))
    ##---------------------------------
    
    ##---------------------------------
    # Calculating Omega from S-Omega decomposition        
    x1 = np.cos(2*chi*np.pi/180)*np.cos(2*psi*np.pi/180)*np.cos(2*Chi*np.pi/180)*np.cos(2*Psi*np.pi/180)
    x2 = np.cos(2*chi*np.pi/180)*np.sin(2*psi*np.pi/180)*np.cos(2*Chi*np.pi/180)*np.sin(2*Psi*np.pi/180)
    x3 = np.abs(np.sin(2*chi*np.pi/180)*np.sin(2*Chi*np.pi/180))
    Prec =  dop*(1 + x1 + x2 + x3)
    Prec1 = (1 - dop) + dop*(1 + x1 + x2 + x3)
    omega = (Prec/Prec1)
       
    
    # ## Improved S-Omega (i-SOmega powers
    ind_g1 = (CPR>1).astype(int)
    s_new_g1 = omega*(1 - omega)*OC
    db_new_g1 = omega*s0 - omega*(1 - omega)*OC   ##depolarized of OC x polarized of SC

    ind_l1 = (CPR<1).astype(int)
    s_new_l1 = omega*s0 - omega*(1 - omega)*SC
    db_new_l1 = omega*(1 - omega)*SC   ##depolarized of OC x polarized of SC

    ind_e1 = (CPR==1).astype(int)
    s_new_e1 = omega*OC
    db_new_e1 = omega*SC
                     
    
    surface_new = s_new_g1*ind_g1+s_new_l1*ind_l1+s_new_e1*ind_e1
    double_bounce_new = db_new_g1*ind_g1+db_new_l1*ind_l1+db_new_e1*ind_e1

    diffused_new = (1 - omega)*s0; ##diffused scattering
    
    surface_new[surface_new==0] = np.nan
    double_bounce_new[double_bounce_new==0] = np.nan
    diffused_new[diffused_new==0] = np.nan
    

    return surface_new.astype(np.float32), double_bounce_new.astype(np.float32), diffused_new.astype(np.float32)