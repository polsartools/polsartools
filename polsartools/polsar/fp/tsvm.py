import os
import numpy as np
from polsartools.utils.proc_utils import process_chunks_parallel
from polsartools.utils.utils import conv2d,time_it
from polsartools.utils.convert_matrices import C3_T3_mat
from .fp_infiles import fp_c3t3files
@time_it
def tsvm(in_dir,  win=1, fmt="tif", cog=False, 
          ovr = [2, 4, 8, 16], comp=False, 
          max_workers=None,block_size=(512, 512),
          progress_callback=None,  # for QGIS plugin
          ):
    """Perform Touzi Decomposition for full-pol SAR data.


    Examples
    --------
    >>> # Basic usage with default parameters
    >>> tsvm("/path/to/fullpol_data")
    
    >>> # Advanced usage with custom parameters
    >>> tsvm(
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
        Writes the follwing files to disk:
        
        1. TSVM_alpha1
        2. TSVM_alpha2
        3. TSVM_alpha3 
        4. TSVM_phi1 
        5. TSVM_phi2 
        6. TSVM_phi3 
        7. TSVM_tau1 
        8. TSVM_tau2 
        9. TSVM_tau3 
        10. TSVM_psi1 
        11. TSVM_psi2 
        12. TSVM_psi3
        13. TSVM_alphas
        14. TSVM_phis
        15. TSVM_taus
        16. TSVM_psis

    """
    write_flag=True
    input_filepaths = fp_c3t3files(in_dir)

    output_filepaths = []
    if fmt == "bin":
        output_filepaths.append(os.path.join(in_dir, "TSVM_alpha1.bin"))
        output_filepaths.append(os.path.join(in_dir, "TSVM_alpha2.bin"))
        output_filepaths.append(os.path.join(in_dir, "TSVM_alpha3.bin"))
        output_filepaths.append(os.path.join(in_dir, "TSVM_phi1.bin"))
        output_filepaths.append(os.path.join(in_dir, "TSVM_phi2.bin"))
        output_filepaths.append(os.path.join(in_dir, "TSVM_phi3.bin"))
        output_filepaths.append(os.path.join(in_dir, "TSVM_tau1.bin"))
        output_filepaths.append(os.path.join(in_dir, "TSVM_tau2.bin"))
        output_filepaths.append(os.path.join(in_dir, "TSVM_tau3.bin"))
        output_filepaths.append(os.path.join(in_dir, "TSVM_psi1.bin"))
        output_filepaths.append(os.path.join(in_dir, "TSVM_psi2.bin"))
        output_filepaths.append(os.path.join(in_dir, "TSVM_psi3.bin"))
        
        output_filepaths.append(os.path.join(in_dir, "TSVM_alphas.bin"))
        output_filepaths.append(os.path.join(in_dir, "TSVM_phis.bin"))
        output_filepaths.append(os.path.join(in_dir, "TSVM_taus.bin"))
        output_filepaths.append(os.path.join(in_dir, "TSVM_psis.bin"))
        
        
        
    else:
        output_filepaths.append(os.path.join(in_dir, "TSVM_alpha1.tif"))
        output_filepaths.append(os.path.join(in_dir, "TSVM_alpha2.tif"))
        output_filepaths.append(os.path.join(in_dir, "TSVM_alpha3.tif"))
        output_filepaths.append(os.path.join(in_dir, "TSVM_phi1.tif"))
        output_filepaths.append(os.path.join(in_dir, "TSVM_phi2.tif"))
        output_filepaths.append(os.path.join(in_dir, "TSVM_phi3.tif"))
        output_filepaths.append(os.path.join(in_dir, "TSVM_tau1.tif"))
        output_filepaths.append(os.path.join(in_dir, "TSVM_tau2.tif"))
        output_filepaths.append(os.path.join(in_dir, "TSVM_tau3.tif"))
        output_filepaths.append(os.path.join(in_dir, "TSVM_psi1.tif"))
        output_filepaths.append(os.path.join(in_dir, "TSVM_psi2.tif"))
        output_filepaths.append(os.path.join(in_dir, "TSVM_psi3.tif"))
        
        output_filepaths.append(os.path.join(in_dir, "TSVM_alphas.tif"))        
        output_filepaths.append(os.path.join(in_dir, "TSVM_phis.tif"))        
        output_filepaths.append(os.path.join(in_dir, "TSVM_taus.tif"))        
        output_filepaths.append(os.path.join(in_dir, "TSVM_psis.tif"))

            
    process_chunks_parallel(input_filepaths, list(output_filepaths), 
                            window_size=win, write_flag=write_flag,
                        processing_func=process_chunk_tsvm,block_size=block_size, 
                        max_workers=max_workers,  num_outputs=len(output_filepaths),
                        cog=cog, ovr=ovr, comp=comp,
                        progress_callback=progress_callback
                        )

def process_chunk_tsvm(chunks, window_size, input_filepaths,*args):

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
        C3 = np.array([[C11, C12, C13], 
                         [C21, C22, C23], 
                         [C31, C32, C33]])

        T_T1 = C3_T3_mat(C3)


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

    pi = np.pi
    nx, ny = T_T1.shape[2], T_T1.shape[3]
    N = nx * ny
    eps = 1e-8
    
    # Reshape M to (N, 3, 3) for batch processing
    M_flat = T_T1.transpose(2, 3, 0, 1).reshape(N, 3, 3)

    # Eigen decomposition
    lambda_vals, V = np.linalg.eig(M_flat)  # V: (N, 3, 3), lambda_vals: (N, 3)

    # Sort eigenvalues and corresponding eigenvectors
    idx = np.argsort(lambda_vals.real, axis=1)[:, ::-1]  # descending order
    lambda_sorted = np.take_along_axis(lambda_vals.real, idx, axis=1)
    V_sorted = np.array([V[i][:, idx[i]] for i in range(N)])  # shape: (N, 3, 3)

    # Clip negative eigenvalues
    lambda_sorted = np.clip(lambda_sorted, 0, None)

    # Normalize eigenvectors: remove global phase
    phase = np.arctan2(V_sorted[:, 0, :].imag, eps + V_sorted[:, 0, :].real)
    V_sorted *= np.exp(-1j * phase[:, np.newaxis, :])

    # Compute psi
    psi = 0.5 * np.arctan2(V_sorted[:, 2, :].real, eps + V_sorted[:, 1, :].real)

    # Rotate V[1] and V[2] by 2*psi manually
    cos2psi = np.cos(2 * psi)
    sin2psi = np.sin(2 * psi)
    V1r, V1i = V_sorted[:, 1, :].real, V_sorted[:, 1, :].imag
    V2r, V2i = V_sorted[:, 2, :].real, V_sorted[:, 2, :].imag

    V1r_new = V1r * cos2psi + V2r * sin2psi
    V1i_new = V1i * cos2psi + V2i * sin2psi
    V2r_new = -V1r * sin2psi + V2r * cos2psi
    V2i_new = -V1i * sin2psi + V2i * cos2psi

    V_sorted[:, 1, :] = V1r_new + 1j * V1i_new
    V_sorted[:, 2, :] = V2r_new + 1j * V2i_new

    # Compute tau
    tau = 0.5 * np.arctan2(-V_sorted[:, 2, :].imag, eps + V_sorted[:, 0, :].real)

    # Compute phi
    phi = np.arctan2(V_sorted[:, 1, :].imag, eps + V_sorted[:, 1, :].real)

    # Rotate V[0] and V[2] by 2*tau manually
    cos2tau = np.cos(2 * tau)
    sin2tau = np.sin(2 * tau)
    V0r, V0i = V_sorted[:, 0, :].real, V_sorted[:, 0, :].imag
    V2r, V2i = V_sorted[:, 2, :].real, V_sorted[:, 2, :].imag

    V0r_new = V0r * cos2tau - V2i * sin2tau
    V0i_new = V0i * cos2tau + V2r * sin2tau
    V2r_new = -V0i * sin2tau + V2r * cos2tau
    V2i_new = V0r * sin2tau + V2i * cos2tau

    V_sorted[:, 0, :] = V0r_new + 1j * V0i_new
    V_sorted[:, 2, :] = V2r_new + 1j * V2i_new

    # Compute alpha
    alpha = np.arccos(V_sorted[:, 0, :].real)

    # Flip signs if psi out of bounds
    flip_mask = (psi < -pi / 4) | (psi > pi / 4)
    tau[flip_mask] *= -1
    phi[flip_mask] *= -1

    # Compute probabilities
    total_lambda = eps + np.sum(lambda_sorted, axis=1, keepdims=True)
    p = np.clip(lambda_sorted / total_lambda, 0, 1)

    # Weighted mean angles
    alpha_mean = np.sum(alpha * p, axis=1)
    phi_mean = np.sum(phi * p, axis=1)
    tau_mean = np.sum(tau * p, axis=1)
    psi_mean = np.sum(psi * p, axis=1)

    # Reshape outputs back to (3, nx, ny) and (nx, ny)
    shape = (nx, ny)
    to_deg = 180. / pi
    
    
    alphas = (alpha.T.reshape((3,) + shape) * to_deg).astype(np.float32)
    phis = (phi.T.reshape((3,) + shape) * to_deg).astype(np.float32)
    taus = (tau.T.reshape((3,) + shape) * to_deg).astype(np.float32)
    psis = (psi.T.reshape((3,) + shape) * to_deg).astype(np.float32)
    alpha_mean = (alpha_mean.reshape(shape) * to_deg).astype(np.float32)
    phi_mean = (phi_mean.reshape(shape) * to_deg).astype(np.float32)
    tau_mean = (tau_mean.reshape(shape) * to_deg).astype(np.float32)
    psi_mean = (psi_mean.reshape(shape) * to_deg).astype(np.float32)
    
    return  alphas[0], alphas[1], alphas[2], \
            phis[0], phis[1], phis[2], \
            taus[0], taus[1], taus[2], \
            psis[0], psis[1], psis[2], \
            alpha_mean, phi_mean, tau_mean, psi_mean
    
    # return {
    #     "alpha": alpha.T.reshape((3,) + shape) * to_deg,
    #     "phi": phi.T.reshape((3,) + shape) * to_deg,
    #     "tau": tau.T.reshape((3,) + shape) * to_deg,
    #     "psi": psi.T.reshape((3,) + shape) * to_deg,
    #     "alpha_mean": alpha_mean.reshape(shape) * to_deg,
    #     "phi_mean": phi_mean.reshape(shape) * to_deg,
    #     "tau_mean": tau_mean.reshape(shape) * to_deg,
    #     "psi_mean": psi_mean.reshape(shape) * to_deg,
    #     "p": p.T.reshape((3,) + shape)
    # }