import numpy as np

def process_chunk_refined_lee(chunks, window_size=3, *args ):

    Nlook = 1
    if window_size % 2 == 0:
        window_size = window_size + 1
    # print('before pad',np.shape(chunks[0]))
    for i in range(len(chunks)):
        pad_top_left = window_size // 2 
        pad_bottom_right = window_size // 2 +1
        
        # Pad the array
        chunks[i] = np.pad(chunks[i], 
                            ((pad_top_left, pad_bottom_right), 
                            (pad_top_left, pad_bottom_right)), 
                            mode='constant', constant_values=0)



    # print("after pad",np.shape(chunks[0]))
    if len(chunks)==16:
        t11_T1 = np.array(chunks[0])
        t12_T1 = np.array(chunks[1])+1j*np.array(chunks[2])
        t13_T1 = np.array(chunks[3])+1j*np.array(chunks[4])
        t14_T1 = np.array(chunks[5])+1j*np.array(chunks[6])

        t21_T1 = np.conj(t12_T1)
        t22_T1 = np.array(chunks[7])
        t23_T1 = np.array(chunks[8])+1j*np.array(chunks[9])
        t24_T1 = np.array(chunks[10])+1j*np.array(chunks[11])
        
        t31_T1 = np.conj(t13_T1)
        t32_T1 = np.conj(t23_T1)
        t33_T1 = np.array(chunks[12])
        t34_T1 = np.array(chunks[13])+1j*np.array(chunks[14])

        t41_T1 = np.conj(t14_T1)
        t42_T1 = np.conj(t24_T1)
        t43_T1 = np.conj(t34_T1)
        t44_T1 = np.array(chunks[15])

        M_in = np.dstack((t11_T1,t12_T1,t13_T1,t14_T1,
                            t21_T1,t22_T1,t23_T1,t24_T1,
                            t31_T1,t32_T1,t33_T1,t34_T1,
                            t41_T1,t42_T1,t43_T1,t44_T1,
                            ))
        PolTypeOut = "C4"

    # print("after pad",np.shape(chunks[0]))
    if len(chunks)==9:
        t11_T1 = np.array(chunks[0])
        t12_T1 = np.array(chunks[1])+1j*np.array(chunks[2])
        t13_T1 = np.array(chunks[3])+1j*np.array(chunks[4])
        t21_T1 = np.conj(t12_T1)
        t22_T1 = np.array(chunks[5])
        t23_T1 = np.array(chunks[6])+1j*np.array(chunks[7])
        t31_T1 = np.conj(t13_T1)
        t32_T1 = np.conj(t23_T1)
        t33_T1 = np.array(chunks[8])

        M_in = np.dstack((t11_T1,t12_T1,t13_T1,
                            t21_T1,t22_T1,t23_T1,
                            t31_T1,t32_T1,t33_T1))
        PolTypeOut = "C3"
    if len(chunks)==4:
        t11_T1 = np.array(chunks[0])
        t12_T1 = np.array(chunks[1])+1j*np.array(chunks[2])
        t21_T1 = np.conj(t12_T1)
        t22_T1 = np.array(chunks[3])

        M_in = np.dstack((t11_T1,t12_T1,
                            t21_T1,t22_T1
                            ))
        PolTypeOut = "C2"
    

    M_in = np.transpose(M_in,axes=[2,0,1])
    # print("M_in shape",M_in.shape)
    NpolarOut, Nlig_padded, Ncol_padded = M_in.shape
    Nlig = Nlig_padded - window_size
    Ncol = Ncol_padded - window_size
    window_sizeM1S2 = (window_size - 1) // 2
    window_sizeL = window_sizeC = window_size

    # Initialize Valid to ones (assuming no mask)
    Valid = np.ones((Nlig_padded, Ncol_padded), dtype=np.float32)

    # Speckle variance given by the input data number of looks
    sigma2 = 1.0 / Nlook

    # Gradient window calculation parameters
    window_params = {
        3: (1, 1),
        5: (3, 1),
        7: (3, 2),
        9: (5, 2),
        11: (5, 3),
        13: (5, 4),
        15: (7, 4),
        17: (7, 5),
        19: (7, 6),
        21: (9, 6),
        23: (9, 7),
        25: (9, 8),
        27: (11, 8),
        29: (11, 9),
        31: (11, 10),
    }

    if window_size in window_params:
        Nwindow_size, Deplct = window_params[window_size]
    else:
        raise ValueError("The window width window_size must be set to 3 to 31")

    # Create Mask
    Mask = make_Mask(window_size)

    # Compute span based on PolTypeOut
    span = np.zeros((Nlig_padded, Ncol_padded), dtype=np.float32)
    if PolTypeOut in ["C2", "C2pp1", "C2pp2", "C2pp3", "T2", "T2pp1", "T2pp2", "T2pp3"]:
        span = M_in[0] + M_in[3]
    elif PolTypeOut in ["C3", "T3"]:
        span = M_in[0] + M_in[4] + M_in[8]
    elif PolTypeOut in ["C4", "T4"]:
        span = M_in[0] + M_in[6] + M_in[11] + M_in[15]
    else:
        raise ValueError("Unsupported PolTypeOut")

    # Compute coefficients
    coeff, Nmax = make_Coeff(sigma2, Deplct, Nwindow_size, window_sizeM1S2, Nlig, Ncol, span, Mask)

    # Initialize M_out
    M_out = np.zeros((NpolarOut, Nlig, Ncol), dtype=np.complex64)

    # Filtering element per element
    for lig in range(Nlig):
        for col in range(Ncol):
            if Valid[window_sizeM1S2 + lig, window_sizeM1S2 + col] == 1.0:
                for Np in range(NpolarOut):
                    mean = 0.0
                    Npoints = 0.0
                    for k in range(-window_sizeM1S2, window_sizeM1S2 + 1):
                        for l in range(-window_sizeM1S2, window_sizeM1S2 + 1):
                            mask_value = Mask[Nmax[lig, col], window_sizeM1S2 + k, window_sizeM1S2 + l]
                            if mask_value == 1.0:
                                idx_k = window_sizeM1S2 + lig + k
                                idx_l = window_sizeM1S2 + col + l
                                mean += M_in[Np, idx_k, idx_l]
                                Npoints += 1.0
                    mean /= Npoints
                    center_pixel = M_in[Np, window_sizeM1S2 + lig, window_sizeM1S2 + col]
                    M_out[Np, lig, col] = mean + coeff[lig, col] * (center_pixel - mean)
    
    filtered_chunks = []

    if len(chunks)==16:
            filtered_chunks.append(np.real(M_out[0,:,:]))
            filtered_chunks.append(np.real(M_out[1,:,:]))
            filtered_chunks.append(np.imag(M_out[1,:,:]))
            filtered_chunks.append(np.real(M_out[2,:,:]))
            filtered_chunks.append(np.imag(M_out[2,:,:]))
            filtered_chunks.append(np.real(M_out[3,:,:]))
            filtered_chunks.append(np.imag(M_out[3,:,:]))

            filtered_chunks.append(np.real(M_out[5,:,:]))
            filtered_chunks.append(np.real(M_out[6,:,:]))
            filtered_chunks.append(np.imag(M_out[6,:,:]))
            filtered_chunks.append(np.real(M_out[7,:,:]))
            filtered_chunks.append(np.imag(M_out[7,:,:]))

            filtered_chunks.append(np.real(M_out[10,:,:]))
            filtered_chunks.append(np.real(M_out[11,:,:]))
            filtered_chunks.append(np.imag(M_out[11,:,:]))

            filtered_chunks.append(np.real(M_out[15,:,:]))


    if len(chunks)==9:
            filtered_chunks.append(np.real(M_out[0,:,:]))
            filtered_chunks.append(np.real(M_out[1,:,:]))
            filtered_chunks.append(np.imag(M_out[1,:,:]))
            filtered_chunks.append(np.real(M_out[2,:,:]))
            filtered_chunks.append(np.imag(M_out[2,:,:]))
            filtered_chunks.append(np.real(M_out[4,:,:]))
            filtered_chunks.append(np.real(M_out[5,:,:]))
            filtered_chunks.append(np.imag(M_out[5,:,:]))
            filtered_chunks.append(np.real(M_out[8,:,:]))


    if len(chunks)==4:
            filtered_chunks.append(np.real(M_out[0,:,:]))
            filtered_chunks.append(np.real(M_out[1,:,:]))
            filtered_chunks.append(np.imag(M_out[1,:,:]))
            filtered_chunks.append(np.real(M_out[3,:,:]))

    # M_out = np.transpose(M_out,axes=[1,2,0])
    # print("M_out shape:", np.shape(filtered_chunks))
    
    
    
    return filtered_chunks



def make_Mask(window_size):
    Mask = np.zeros((8, window_size, window_size), dtype=np.float32)

    Nmax = 0
    for k in range(window_size):
        for l in range((window_size - 1) // 2, window_size):
            Mask[Nmax, k, l] = 1.0

    Nmax = 4
    for k in range(window_size):
        for l in range(0, 1 + (window_size - 1) // 2):
            Mask[Nmax, k, l] = 1.0

    Nmax = 1
    for k in range(window_size):
        for l in range(k, window_size):
            Mask[Nmax, k, l] = 1.0

    Nmax = 5
    for k in range(window_size):
        for l in range(0, k + 1):
            Mask[Nmax, k, l] = 1.0

    Nmax = 2
    for k in range(0, 1 + (window_size - 1) // 2):
        for l in range(window_size):
            Mask[Nmax, k, l] = 1.0

    Nmax = 6
    for k in range((window_size - 1) // 2, window_size):
        for l in range(window_size):
            Mask[Nmax, k, l] = 1.0

    Nmax = 3
    for k in range(window_size):
        for l in range(0, window_size - k):
            Mask[Nmax, k, l] = 1.0

    Nmax = 7
    for k in range(window_size):
        for l in range(window_size - 1 - k, window_size):
            Mask[Nmax, k, l] = 1.0

    return Mask

def make_Coeff(sigma2, Deplct, Nwindow_size, window_sizeM1S2, Sub_Nlig, Sub_Ncol, span, Mask):
    coeff = np.zeros((Sub_Nlig, Sub_Ncol), dtype=np.complex64)
    Nmax = np.zeros((Sub_Nlig, Sub_Ncol), dtype=np.int32)
    
    for lig in range(Sub_Nlig):
        for col in range(Sub_Ncol):
            # 3x3 average SPAN sub-window calculation for directional gradient determination
            subwin = np.zeros((3, 3), dtype=np.complex64)
            for k in range(3):
                for l in range(3):
                    sum_subwin = 0.0
                    for kk in range(Nwindow_size):
                        for ll in range(Nwindow_size):
                            idx_k = k * Deplct + kk + lig
                            idx_l = l * Deplct + ll + col
                            sum_subwin += span[idx_k, idx_l] / (Nwindow_size * Nwindow_size)
                    subwin[k, l] = sum_subwin

            # Directional gradient computation
            Dist = np.zeros(4, dtype=np.complex64)
            Dist[0] = -subwin[0, 0] + subwin[0, 2] - subwin[1, 0] + subwin[1, 2] - subwin[2, 0] + subwin[2, 2]
            Dist[1] =  subwin[0, 1] + subwin[0, 2] - subwin[1, 0] + subwin[1, 2] - subwin[2, 0] - subwin[2, 1]
            Dist[2] =  subwin[0, 0] + subwin[0, 1] + subwin[0, 2] - subwin[2, 0] - subwin[2, 1] - subwin[2, 2]
            Dist[3] =  subwin[0, 0] + subwin[0, 1] + subwin[1, 0] - subwin[1, 2] - subwin[2, 1] - subwin[2, 2]

            # Choice of a directional mask according to the maximum gradient
            MaxDist = -np.inf
            Nmax_lig_col = 0
            for k in range(4):
                if MaxDist < abs(Dist[k]):
                    MaxDist = abs(Dist[k])
                    Nmax_lig_col = k
            if Dist[Nmax_lig_col] > 0.0:
                Nmax_lig_col += 4
            Nmax[lig, col] = Nmax_lig_col

            # Within window statistics
            m_span = 0.0
            m_span2 = 0.0
            Npoints = 0.0
            for k in range(-window_sizeM1S2, window_sizeM1S2 + 1):
                for l in range(-window_sizeM1S2, window_sizeM1S2 + 1):
                    mask_value = Mask[Nmax_lig_col, window_sizeM1S2 + k, window_sizeM1S2 + l]
                    if mask_value == 1.0:
                        idx_k = window_sizeM1S2 + k + lig
                        idx_l = window_sizeM1S2 + l + col
                        s = span[idx_k, idx_l]
                        m_span += s
                        m_span2 += s * s
                        Npoints += 1.0
            m_span /= Npoints
            m_span2 /= Npoints

            # SPAN variation coefficient cv_span
            v_span = m_span2 - m_span * m_span  # Var(x) = E(x^2) - E(x)^2
            cv_span = np.sqrt(abs(v_span)) / (1e-8 + m_span)

            # Linear filter coefficient
            coeff_lig_col = (cv_span * cv_span - sigma2) / (cv_span * cv_span * (1 + sigma2) + 1e-8)
            if coeff_lig_col < 0.0:
                coeff_lig_col = 0.0
            coeff[lig, col] = coeff_lig_col

    return coeff, Nmax
