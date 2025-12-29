import os
import numpy as np
from polsartools.utils.proc_utils import process_chunks_parallel
from polsartools.utils.utils import conv2d,time_it,eig22
from .dxp_infiles import dxpc2files
@time_it
def powers_dp_grd(cpFile,xpFile, method=1, win=1, fmt="tif", 
           cog=False, ovr = [2, 4, 8, 16], comp=False,
           max_workers=None,block_size=(512, 512),
           progress_callback=None,  # for QGIS plugin          
           ):
    """
    This function computes the scattering power components for GRD (only intensity no phase) dual-pol SAR data (decomposition/factorization based approach)


    Examples
    --------
    >>> # Basic usage with default parameters
    >>> powers_dp_grd("/path/to/copol_file.tif", "/path/to/crosspol_file.tif")

    >>> # Advanced usage with custom parameters
    >>> powers_dp_grd(
    ...     cpFile="/path/to/copol_file.tif",
    ...     xpFile="/path/to/crosspol_file.tif",
    ...     method=2,    
    ...     win=3,
    ...     fmt="tif",
    ...     cog=True,
    ...     block_size=(1024, 1024)
    ... )
    
    Parameters
    ----------
    cpFile : str
        Path to the co-polarized backscatter (linear) SAR raster file.
    xpFile : str
        Path to the cross-polarized backscatter (linear) SAR raster file.
    method : int
        1: Decomposition based powers
        2: Factorisation based powers
    win : int, default=1
        Size of the spatial averaging window. Larger windows reduce speckle noise
        but decrease spatial resolution.
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


    """
    write_flag=True
    input_filepaths = [cpFile,xpFile]
    output_filepaths = []

    if fmt == "bin":
        if method==1:
            output_filepaths.append(os.path.join(os.path.dirname(cpFile), "Alpha_dp_grd.bin"))
            output_filepaths.append(os.path.join(os.path.dirname(cpFile), "Pdl_dcmp_grd.bin"))
            output_filepaths.append(os.path.join(os.path.dirname(cpFile), "Psl_dcmp_grd.bin"))
            output_filepaths.append(os.path.join(os.path.dirname(cpFile), "Pu_dcmp_grd.bin"))
        else:
            output_filepaths.append(os.path.join(os.path.dirname(cpFile), "Pdl_fact_grd.bin"))
            output_filepaths.append(os.path.join(os.path.dirname(cpFile), "Psl_fact_grd.bin"))
            output_filepaths.append(os.path.join(os.path.dirname(cpFile), "Pr_fact_grd.bin"))
    else:
        if method==1:
            output_filepaths.append(os.path.join(os.path.dirname(cpFile), "Alpha_dp_grd.tif"))
            output_filepaths.append(os.path.join(os.path.dirname(cpFile), "Pdl_dcmp_grd.tif"))
            output_filepaths.append(os.path.join(os.path.dirname(cpFile), "Psl_dcmp_grd.tif"))
            output_filepaths.append(os.path.join(os.path.dirname(cpFile), "Pu_dcmp_grd.tif"))
        else:
            output_filepaths.append(os.path.join(os.path.dirname(cpFile), "Pdl_fact_grd.tif"))
            output_filepaths.append(os.path.join(os.path.dirname(cpFile), "Psl_fact_grd.tif"))
            output_filepaths.append(os.path.join(os.path.dirname(cpFile), "Pr_fact_grd.tif"))

    process_chunks_parallel(input_filepaths, list(output_filepaths), win, write_flag, 
                            process_chunk_dp_powers,
                            *[method],
                            block_size=block_size, max_workers=max_workers,  num_outputs=len(output_filepaths),
                            cog=cog,ovr=ovr, comp=comp,
                            progress_callback=progress_callback
                            )
    
def process_chunk_dp_powers(chunks, window_size,*args, **kwargs):
    method = int(args[-1])
    kernel = np.ones((window_size,window_size),np.float32)/(window_size*window_size)
    c11 = np.array(chunks[0])
    c22 = np.array(chunks[1])


    def S_norm(S_array):
        S_5 = np.percentile(S_array, 5)
        S_95 = np.percentile(S_array, 95)
        S_cln = np.where(S_array > S_95, S_95, S_array)
        S_cln = np.where(S_cln < S_5, S_5, S_cln)
        S_cln_max = np.max(S_cln)
        S_norm_array = np.divide(S_cln,S_cln_max) 
        
        return S_norm_array

    if window_size>1:
        c11 = conv2d(c11,kernel)
        c22 = conv2d(c22,kernel)

    s0 = np.abs(c11 + c22)
    s1 = np.abs(c11 - c22)

    C11_av_db = 10*np.log10(c11)

    prob1 = c11/(c11 + c22)
    prob2 = c22/(c11 + c22)

    ent = -prob1*np.log2(prob1) - prob2*np.log2(prob2)
    s1_s_norm = S_norm(s1) #This is S1 normalzied for DpRSI, does not include slope mask
    C11_norm = S_norm(c11)
    C22_norm = S_norm(c22)


    dop = (c11 - c22)/(c11 + c22)
    dop = np.abs(dop)
    beta = prob1

    ##### Power Calculation

    dprbi = np.sqrt(np.square(C11_norm) + np.square(C22_norm))/np.sqrt(2)
    dprbi = dprbi*s1_s_norm

    dprsi_con1 = (1 - ent)*np.sqrt(1 - np.square(s1_s_norm)) # For Valid pixels
    dprsi_con2 = np.sqrt(1 - np.square(s1_s_norm)) # For Noise pixels 

    NESZ = -16 ## For Sentinel-1
    dprsi = np.where(C11_av_db > NESZ, dprsi_con1, dprsi_con2) 

    shp = np.shape(dprbi)

    if method==1:

        alpha1 = np.arctan2(dprbi, 1 - dprbi)
        alpha1 = np.degrees(alpha1)

        alpha2 = np.arctan2(1-dprsi, dprsi)
        alpha2 = np.degrees(alpha2)

        alpha_dp = (alpha1 + alpha2)/2; #Dual-pol target characteristic parameter proposed in Verma et al. 2024

        ## Alpha as geomteric factor
        alpha_dp_rad = np.radians(2*alpha_dp) 

        cos_a = np.cos(alpha_dp_rad)

        ## Power components for valid pixels (VV > NESZ)
        Pu_v = (1 - dop)*s0 
        Pd_v = (1/2)*dop*s0*(1 - cos_a)
        Ps_v = (1/2)*dop*s0*(1 + cos_a)

        ## Power components for noise pixels (VV < NESZ)
        Pu_n = (1 - beta)*s0 
        Pd_n = (1/2)*beta*s0*(1 - cos_a)
        Ps_n = (1/2)*beta*s0*(1 + cos_a)

        ## Dual-pol scattering power
        Pu = np.where(C11_av_db > NESZ, Pu_v, Pu_n) # Unpolized power
        Pd = np.where(C11_av_db > NESZ, Pd_v, Pd_n) # "Dihedral-like" power
        Ps = np.where(C11_av_db > NESZ, Ps_v, Ps_n) # "Surface-like" power
    

        return alpha_dp.astype(np.float32), Pd.astype(np.float32), Ps.astype(np.float32), Pu.astype(np.float32)
    else:
        dprbi_flt = dprbi.flatten()
        dprsi_flt = dprsi.flatten()
        shp_flt = np.shape(dprbi_flt)


        indices_vec = np.array([dprsi_flt, dprbi_flt]).transpose()
        indices_vec_sort = np.array([[max(row), min(row)] for row in indices_vec])

        y1 = indices_vec_sort[:,0] #First dominant
        y2 = (1 - indices_vec_sort[:,0])*indices_vec_sort[:,1] #Second dominant

        residue = 1 - (y1 + y2)

        dprsi_dom = np.where(dprsi_flt > dprbi_flt)[0] #Keeps the tuple where dprsi is dominant
        dprbi_dom = np.where(dprsi_flt < dprbi_flt)[0] #Keeps the tuple where dprbi is dominant

        ## Surface-like power component
        Ps = np.zeros(shp_flt)
        ##dprsi_dom and dprbi_dom are not dprsi and dprbi values, they just indicate tuples (pixel) for which they are greater
        Ps[dprsi_dom] = y1[dprsi_dom] #In these tuples dprsi was dominant, hence taking y1 (first dominant)
        Ps[dprbi_dom] = y2[dprbi_dom] #In these tuples dprbi was dominant, hence taking y2 (Second domiant)

        Ps = Ps.reshape(shp[0],shp[1])
        Ps = np.multiply(s0,Ps)

        ## Dihedral-like power component
        Pd = np.zeros(shp_flt)
        Pd[dprbi_dom] = y1[dprbi_dom]
        Pd[dprsi_dom] = y2[dprsi_dom]

        Pd = Pd.reshape(shp[0],shp[1])
        Pd = np.multiply(s0,Pd)

        ## Residue (diffused) power component

        Pr = residue.reshape(shp[0],shp[1])
        Pr = np.multiply(s0,Pr)

        return Pd.astype(np.float32), Ps.astype(np.float32), Pr.astype(np.float32)
