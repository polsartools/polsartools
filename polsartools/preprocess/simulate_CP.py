import os
import numpy as np
from polsartools.utils.proc_utils import process_chunks_parallel
from polsartools.utils.utils import conv2d,time_it
from polsartools.utils.convert_matrices import T3_C3_mat, C3_T3_mat
from .fp_infiles import fp_c3t3files

@time_it
def simulate_CP(in_dir,  chi=45,psi=0, win=1, fmt="tif", cog=False, 
                    ovr = [2, 4, 8, 16], comp=False, 
                    max_workers=None,block_size=(512, 512),
                    progress_callback=None,  # for QGIS plugin
                        ):
    """
    This function simulates Compact polarimetric C2 matrix (RHV, LHV, pi/4 etc) from full polarimetric C3/T3 matrix.

    Examples
    --------
    >>> # Basic usage with default LC
    >>> simulate_CP("/path/to/C3")

    >>> # With chi, psi and COG GeoTIFF output
    >>> simulate_CP("/path/to/C3", chi=-45, psi=0, fmt="tif", cog=True)

    Parameters
    ----------
    in_dir : str
        Path to the input folder containing a supported polarimetric matrix.
    chi : float, default=45
        Ellipticity angle chi of the transmitted wave in degrees.
        For circular polarization, chi = 45° (right circular) or -45° (left circular).
    psi : float, default=0
        Orientation angle psi of the transmitted wave in degrees.
        For circular polarization, typically 0°.
    fmt : {'tif', 'bin'}, default='tif'
        Output format:
        - 'tif': Cloud-optimized GeoTIFF (if cog_flag is True)
        - 'bin': Raw binary format
    cog : bool, default=False
        Enable Cloud Optimized GeoTIFF output with internal overviews and tiling.
    ovr : list[int], default=[2, 4, 8, 16]
        Overview levels for pyramid generation (used with COGs).
    comp : bool, default=False
        If True, uses LZW compression for GeoTIFF outputs.
    max_workers : int | None, default=None
        Maximum number of parallel worker threads (defaults to all available CPUs).
    block_size : tuple[int, int], default=(512, 512)
        Size of processing blocks for chunked and parallel execution.

    Returns
    -------
    None
        The simulated CP C2 matrix elements

    """

    write_flag=True
    input_filepaths = fp_c3t3files(in_dir)

    output_filepaths = []

    os.makedirs(os.path.join(in_dir, "C2CP"), exist_ok=True)
    if fmt == "bin":

        output_filepaths.append(os.path.join(in_dir, "C2CP","C11.bin"))
        output_filepaths.append(os.path.join(in_dir, "C2CP","C12_real.bin"))
        output_filepaths.append(os.path.join(in_dir, "C2CP","C12_imag.bin"))
        output_filepaths.append(os.path.join(in_dir, "C2CP","C22.bin"))    
    else:

        output_filepaths.append(os.path.join(in_dir, "C2CP","C11.tif"))
        output_filepaths.append(os.path.join(in_dir, "C2CP","C12_real.tif"))
        output_filepaths.append(os.path.join(in_dir, "C2CP","C12_imag.tif"))
        output_filepaths.append(os.path.join(in_dir, "C2CP","C22.tif"))

    process_chunks_parallel(input_filepaths, list(output_filepaths), 
                    win, write_flag,
                    process_chunk_sim_cp,
                    *[chi,psi],
                    block_size=block_size, 
                    max_workers=max_workers,  num_outputs=len(output_filepaths),
                    cog=cog, ovr=ovr, comp=comp,
                    progress_callback=progress_callback
                    )


# def process_chunk_yam4cfp(chunks, window_size, input_filepaths, model,*args):
def process_chunk_sim_cp(chunks, window_size, input_filepaths,  *args, **kwargs):
    chi=args[-2]
    psi=args[-1]


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

        T3 = np.array([[t11_T1, t12_T1, t13_T1], 
                     [t21_T1, t22_T1, t23_T1], 
                     [t31_T1, t32_T1, t33_T1]])
        T_T1 = T3_C3_mat(T3)



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
        T_T1 = np.array([[C11, C12, C13], 
                         [C21, C22, C23], 
                         [C31, C32, C33]])



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
    
    psi = psi*np.pi/180
    chi = chi*np.pi/180
    
    CP11 = 0.5*((1+np.cos(2*psi)*np.cos(2*chi))*T_T1[0,0,:,:]+
                0.5*(1-np.cos(2*psi)*np.cos(2*chi))*T_T1[1,1,:,:]+
                (1/np.sqrt(2))*(np.sin(2*psi)*np.cos(2*chi))*(T_T1[0,1,:,:]+np.conj(T_T1[0,1,:,:]))+
                    (1j/np.sqrt(2))*np.sin(2*chi)*(T_T1[0,1,:,:]-np.conj(T_T1[0,1,:,:])) 
            )

    CP12 = 0.5*((1/np.sqrt(2))*(1+np.cos(2*psi)*np.cos(2*chi))*T_T1[0,1,:,:]+
                (1/np.sqrt(2))*(1-np.cos(2*psi)*np.cos(2*chi))*T_T1[1,2,:,:]+
                (np.sin(2*psi)*np.cos(2*chi))*(T_T1[0,2,:,:]+0.5*T_T1[1,1,:,:])+
                1j*np.sin(2*chi)*(T_T1[0,2,:,:]-0.5*T_T1[1,1,:,:])
                )

    CP22 = 0.5*(0.5*(1+np.cos(2*psi)*np.cos(2*chi))*T_T1[1,1,:,:]+
                    (1-np.cos(2*psi)*np.cos(2*chi))*T_T1[2,2,:,:]+
                (1/np.sqrt(2))*(np.sin(2*psi)*np.cos(2*chi))*(T_T1[1,2,:,:]+np.conj(T_T1[1,2,:,:]))+
                    (1j/np.sqrt(2))*np.sin(2*chi)*(T_T1[1,2,:,:]-np.conj(T_T1[1,2,:,:])) 
                )



    return np.real(CP11).astype(np.float32), np.real(CP12).astype(np.float32), \
        np.imag(CP12).astype(np.float32), np.real(CP22).astype(np.float32)