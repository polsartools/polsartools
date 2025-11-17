import os
import numpy as np
from polsartools.utils.proc_utils import process_chunks_parallel
from polsartools.utils.utils import conv2d,time_it
from polsartools.utils.convert_matrices import T3_C3_mat
from .fp_infiles import fp_c3t3files
@time_it
def h_a_alpha_fp(in_dir,  win=1, fmt="tif", cog=False, 
          ovr = [2, 4, 8, 16], comp = False,
          max_workers=None,block_size=(512, 512),
            progress_callback=None  # for QGIS plugin
          ):
    """Perform H/A//α (Entropy/Anisotropy/Alpha) decomposition for full-pol SAR data.

    This function implements the Cloude-Pottier decomposition, computing entropy (H),
    alpha angle (α), anisotropy (A), and normalized eigenvalues from full-polarimetric
    SAR coherency (T3) or covariance (C3) matrices. This decomposition is fundamental
    for understanding scattering mechanisms in polarimetric SAR data.

    Examples
    --------
    >>> # Basic usage with default parameters
    >>> h_a_alpha_fp("/path/to/fullpol_data")
    
    >>> # Advanced usage with custom parameters
    >>> h_a_alpha_fp(
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
        Size of the spatial averaging window. Larger windows improve eigenvalue/eigenvector
        estimation but decrease spatial resolution.
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
        If True, uses LZW compression for GeoTIFF outputs.
    max_workers : int | None, default=None
        Maximum number of parallel processing workers. If None, uses
        CPU count - 1 workers.
    block_size : tuple[int, int], default=(512, 512)
        Size of processing blocks (rows, cols) for parallel computation.
        Larger blocks use more memory but may be more efficient.

    Returns
    -------
    None
        Writes six output files to disk:
        1. H_fp: Entropy (H) [0-1]
        2. alpha_fp: Alpha angle (α) [0°-90°]
        3. anisotropy_fp: Anisotropy (A) [0-1]
        4. e1_norm: Normalized first eigenvalue
        5. e2_norm: Normalized second eigenvalue
        6. e3_norm: Normalized third eigenvalue

    Notes
    -----
    The H/α/A decomposition provides three main parameters:

    1. Entropy (H):
       - Range: [0, 1]
       - H = 0: Single scattering mechanism
       - H = 1: Random mixture of scattering mechanisms
       - Formula: H = -∑(pᵢ log₃(pᵢ)), where pᵢ are normalized eigenvalues

    2. Alpha angle (α):
       - Range: [0°, 90°]
       - α ≈ 0°: Surface scattering
       - α ≈ 45°: Volume scattering
       - α ≈ 90°: Double-bounce scattering
       - Formula: α = ∑(pᵢαᵢ), where αᵢ are individual alpha angles

    3. Anisotropy (A):
       - Range: [0, 1]
       - Measures relative importance of secondary mechanisms
       - A = (λ₂ - λ₃)/(λ₂ + λ₃), where λᵢ are eigenvalues


    References
    ----------
        - Cloude, S.R. and Pottier, E., 2002. An entropy based classification scheme for land applications of polarimetric SAR. IEEE transactions on geoscience and remote sensing, 35(1), pp.68-78.
    """
    write_flag=True
    input_filepaths = fp_c3t3files(in_dir)

    output_filepaths = []
    if fmt == "bin":
        output_filepaths.append(os.path.join(in_dir, "H_fp.bin"))
        output_filepaths.append(os.path.join(in_dir, "alpha_fp.bin"))
        output_filepaths.append(os.path.join(in_dir, "anisotropy_fp.bin"))
        output_filepaths.append(os.path.join(in_dir, "e1_norm.bin"))
        output_filepaths.append(os.path.join(in_dir, "e2_norm.bin"))
        output_filepaths.append(os.path.join(in_dir, "e3_norm.bin"))

    else:
        output_filepaths.append(os.path.join(in_dir, "H_fp.tif"))
        output_filepaths.append(os.path.join(in_dir, "alpha_fp.tif"))
        output_filepaths.append(os.path.join(in_dir, "anisotropy_fp.tif"))
        output_filepaths.append(os.path.join(in_dir, "e1_norm.tif"))
        output_filepaths.append(os.path.join(in_dir, "e2_norm.tif"))
        output_filepaths.append(os.path.join(in_dir, "e3_norm.tif"))
    

    process_chunks_parallel(input_filepaths, list(output_filepaths), 
                            window_size=win, write_flag=write_flag,
                        processing_func=process_chunk_halphafp,block_size=block_size, 
                        max_workers=max_workers,  num_outputs=len(output_filepaths),
                        cog=cog,ovr=ovr,comp=comp,                        
                        progress_callback=progress_callback
                        )



def process_chunk_halphafp(chunks, window_size, input_filepaths, *args):

    # additional_arg1 = args[0] if len(args) > 0 else None
    # additional_arg2 = args[1] if len(args) > 1 else None

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

        T_T1 = np.array([[t11_T1, t12_T1, t13_T1], 
                     [t21_T1, t22_T1, t23_T1], 
                     [t31_T1, t32_T1, t33_T1]])
        # T_T1 = T3_C3_mat(T3)


    elif 'C11' in input_filepaths[0] and 'C22' in input_filepaths[5] and 'C33' in input_filepaths[8]:
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

    else:
        raise ValueError("Invalid input matrices. Ensure the input is either T3 or C3 matrix foolder.")


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
    T_T1 = T_T1.reshape(9, rows, cols)
    
    # Indices for vectorized access
    i, j = np.indices((rows, cols))
    
    T_T1 = np.dstack((T_T1[0,:,:],T_T1[1,:,:],T_T1[2,:,:],
                      T_T1[3,:,:],T_T1[4,:,:],T_T1[5,:,:],T_T1[6,:,:],T_T1[7,:,:],T_T1[8,:,:]))
    
    data = T_T1.reshape( T_T1.shape[0]*T_T1.shape[1], T_T1.shape[2]).reshape((-1,3,3))
    
    # infinity, nan handling
    data = np.nan_to_num(data, nan=0.0, posinf=0, neginf=0)
    # data = np.nan_to_num(data, nan=np.nan, posinf=np.nan, neginf=np.nan)

    evals_, evecs_ = np.linalg.eig(data.reshape(-1, 3, 3))

    # Sort eigenvalues for each pixel in descending order; 
    sorted_indices = np.argsort(evals_, axis=-1)[:, ::-1] 
    
    # Reorder eigenvalues and eigenvectors based on sorted indices
    evals = np.take_along_axis(evals_, sorted_indices, axis=-1)  # Reorder eigenvalues
    
    # To reorder the eigenvectors, we use  indexing
    # Use `sorted_indices` to index along the second axis (the 3 components of the eigenvector)
    evecs = np.array([evecs_[i, :, sorted_indices[i]] for i in range(evecs_.shape[0])])
    
    # print('Eigen!')
    # Not sure if this is required; if enabled entropy values are different between polsarpro and this code
    # evals[:,0][evals[:,0] <0] = 0
    # evals[:,1][evals[:,1] <0] = 0
    # evals[:,2][evals[:,2] <0] = 0
    # evals[:,0][evals[:,0] >1] = 1
    # evals[:,1][evals[:,1] >1] = 1
    # evals[:,2][evals[:,2] >1] = 1
    
    eval_norm1 = (evals[:,0])/(evals[:,0] + evals[:,1]+ evals[:,2])
    eval_norm1[eval_norm1<0]=0
    # eval_norm1[eval_norm1>1]=1
    
    
    eval_norm2 = (evals[:,1])/(evals[:,0] + evals[:,1]+ evals[:,2])
    eval_norm2[eval_norm2<0]=0
    # eval_norm2[eval_norm2>1]=1
    
    
    eval_norm3 = (evals[:,2])/(evals[:,0] + evals[:,1]+evals[:,2])
    eval_norm3[eval_norm3<0]=0
    # eval_norm3[eval_norm3>1]=1
      
    
    # # %Alpha 1
    eig_vec_r1 = np.real(evecs[:,0,0])
    eig_vec_c1 = np.imag(evecs[:,0,0])
    alpha1 = np.arccos(np.sqrt(eig_vec_r1*eig_vec_r1 + eig_vec_c1*eig_vec_c1))*180/np.pi
    
    # # %Alpha 2
    eig_vec_r2 = np.real(evecs[:,0,1])
    eig_vec_c2 = np.imag(evecs[:,0,1])
    alpha2 = np.arccos(np.sqrt(eig_vec_r2*eig_vec_r2 + eig_vec_c2*eig_vec_c2))*180/np.pi
    
    # # %Alpha 3
    eig_vec_r3 = np.real(evecs[:,0,2])
    eig_vec_c3 = np.imag(evecs[:,0,2])
    alpha3 = np.arccos(np.sqrt(eig_vec_r3*eig_vec_r3 + eig_vec_c3*eig_vec_c3))*180/np.pi
        
    # # %Cloude Alpha
    alpha_ = (eval_norm1*alpha1 + eval_norm2*alpha2+ eval_norm3*alpha3)
    alpha_ = np.real(alpha_.reshape(rows,cols))
    
    # print('Alpha!')
    
    # # %Entropy
    H = - eval_norm1*np.log10(eval_norm1)/np.log10(3) - eval_norm2*np.log10(eval_norm2)/np.log10(3) - eval_norm3*np.log10(eval_norm3)/np.log10(3)
    H = np.real(H.reshape(rows,cols))

    # alpha1 = alpha1.reshape(rows,cols)
    # alpha2 = alpha2.reshape(rows,cols)
    # alpha3 = alpha3.reshape(rows,cols)
    
    ## POLARIMETRIC SCATTERING ANISOTROPY (A)
    Anisotropy = (eval_norm2-eval_norm3)/(eval_norm2+eval_norm3)
    Anisotropy = np.real(Anisotropy.reshape(rows,cols))
    eval_norm1 = np.real(eval_norm1.reshape(rows,cols))
    eval_norm2 = np.real(eval_norm2.reshape(rows,cols))
    eval_norm3 = np.real(eval_norm3.reshape(rows,cols))
    
    
    return H.astype(np.float32),alpha_.astype(np.float32),Anisotropy.astype(np.float32),\
        eval_norm1.astype(np.float32),eval_norm2.astype(np.float32),eval_norm3.astype(np.float32)