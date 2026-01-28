import os
import numpy as np
from polsartools.utils.proc_utils import process_chunks_parallel
from polsartools.utils.utils import conv2d,time_it,eig22
from .dxp_infiles import dxpc2files
@time_it
def powers_dp(in_dir, method=1, win=1, fmt="tif", cog=False, 
          ovr = [2, 4, 8, 16], comp=False,
          max_workers=None,block_size=(512, 512),
          progress_callback=None,  # for QGIS plugin          
          ):
    """ This function computes the scattering power components for dual-pol SAR C2 matrix data (decomposition/factorization based approach)

    Examples
    --------
    >>> # Basic usage with default parameters
    >>> powers_dp("/path/to/c2_data")
    
    >>> # Advanced usage with custom parameters
    >>> powers_dp(
    ...     in_dir="/path/to/c2_data",
    ...     method=2,   
    ...     win=3,
    ...     fmt="tif",
    ...     cog=True,
    ...     block_size=(1024, 1024)
    ... )
    
    Parameters
    ----------
    in_dir : str
        Path to the input folder containing C2 matrix files.
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
    input_filepaths = dxpc2files(in_dir)
    output_filepaths = []


    if fmt == "bin":
        if method==1:
            output_filepaths.append(os.path.join(in_dir, "Alpha_dp.bin"))
            output_filepaths.append(os.path.join(in_dir, "Pdl_dcmp.bin"))
            output_filepaths.append(os.path.join(in_dir, "Psl_dcmp.bin"))
            output_filepaths.append(os.path.join(in_dir, "Pu_dcmp.bin"))
        else:
            output_filepaths.append(os.path.join(in_dir, "Pdl_fact.bin"))
            output_filepaths.append(os.path.join(in_dir, "Psl_fact.bin"))
            output_filepaths.append(os.path.join(in_dir, "Pr_fact.bin"))
    else:
        if method==1:
            output_filepaths.append(os.path.join(in_dir, "Alpha_dp.tif"))
            output_filepaths.append(os.path.join(in_dir, "Pdl_dcmp.tif"))
            output_filepaths.append(os.path.join(in_dir, "Psl_dcmp.tif"))
            output_filepaths.append(os.path.join(in_dir, "Pu_dcmp.tif"))
        else:
            output_filepaths.append(os.path.join(in_dir, "Pdl_fact.tif"))
            output_filepaths.append(os.path.join(in_dir, "Psl_fact.tif"))
            output_filepaths.append(os.path.join(in_dir, "Pr_fact.tif"))



    process_chunks_parallel(input_filepaths, list(output_filepaths), win, write_flag,
                            process_chunk_dp_pow,
                            *[method],
                            block_size=block_size, max_workers=max_workers,  num_outputs=len(output_filepaths),
                            cog=cog,ovr=ovr, comp=comp,
                            progress_callback=progress_callback
                            )
    
def process_chunk_dp_pow(chunks, window_size,*args):
    method = int(args[-1])
    kernel = np.ones((window_size,window_size),np.float32)/(window_size*window_size)
    c11_T1 = np.array(chunks[0])
    c12_T1 = np.array(chunks[1])+1j*np.array(chunks[2])
    # c21_T1 = np.conj(c12_T1)
    c22_T1 = np.array(chunks[3])

    ##### Normalizing Stokes vector elements
    def S_norm(S_array):
        S_5 = np.nanpercentile(S_array, 2)
        S_95 = np.nanpercentile(S_array, 98)
        S_cln = np.where(S_array > S_95, S_95, S_array)
        S_cln = np.where(S_cln < S_5, S_5, S_cln)
        S_cln_max = np.nanmax(S_cln)
        S_norm_array = np.divide(S_cln,S_cln_max) 
        
        return S_norm_array

    if window_size>1:
        c11s = conv2d(np.real(c11_T1),kernel)
        c12s = conv2d(np.real(c12_T1),kernel)+1j*conv2d(np.imag(c12_T1),kernel)
        # c21s = conv2d(np.real(c21_T1),kernel)+1j*conv2d(np.imag(c21_T1),kernel)
        c22s = conv2d(np.real(c22_T1),kernel)

    else:
        c11s = c11_T1
        c12s = c12_T1
        c22s = c22_T1

    C11_av_db = 10*np.log10(c11s)
    s0 = c11s + c22s
    s1 = c11s - c22s
    s2 = 2*c12s.real
    s3 = 2*c12s.imag


    ##### Calculate Entropy
    ## Here eigen values are calculated using Stokes vector elements

    tpp = np.sqrt(np.square(s1) + np.square(s2) + np.square(s3))

    lmbd1 = (s0 + tpp)/2
    lmbd2 = (s0 - tpp)/2

    prob1 = lmbd1/(lmbd1 + lmbd2)
    prob2 = lmbd2/(lmbd1 + lmbd2)

    ent = -prob1*np.log2(prob1) - prob2*np.log2(prob2)
    dop = (lmbd1 - lmbd2)/(lmbd1 + lmbd2)
    beta = lmbd1/(lmbd1 + lmbd2)

    ##### Taking abs of Stokes vector elements
    s0 = np.abs(s0)
    s1 = np.abs(s1)
    s2 = np.abs(s2)
    s3 = np.abs(s3)

    s1_s_norm = S_norm(s1) #This is S1 normalzied for DpRSI, does not include slope mask
    s1_norm = S_norm(s1)
    s2_norm = S_norm(s2)
    s3_norm = S_norm(s3)
    
    if method==1:
        ##### Power Calculation

        dprbi = np.sqrt(np.square(s1_norm) + np.square(s2_norm) + np.square(s3_norm))/np.sqrt(3)

        dprsi_con1 = (1 - ent)*np.sqrt(1 - np.square(s1_s_norm)) # For Valid pixels
        dprsi_con2 = np.sqrt(1 - np.square(s1_s_norm)) # For Noise pixels 

        NESZ = -16 ## For Sentinel-1
        dprsi = np.where(C11_av_db > NESZ, dprsi_con1, dprsi_con2) 

        shp = np.shape(dprbi)


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

        ##### Power Calculation

        dprbi = np.sqrt(np.square(s1_norm) + np.square(s2_norm) + np.square(s3_norm))/np.sqrt(3)

        dprsi_con1 = (1 - ent)*np.sqrt(1 - np.square(s1_s_norm)) # For Valid pixels
        dprsi_con2 = np.sqrt(1 - np.square(s1_s_norm)) # For Noise pixels 

        NESZ = -16 ## For Sentinel-1
        dprsi = np.where(C11_av_db > NESZ, dprsi_con1, dprsi_con2) 

        shp = np.shape(dprbi)

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
