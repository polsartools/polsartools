
import numpy as np
from polsartools.utils.plot_utils import pol_sign, pol_sign2d, poincare_plot,poincare_sphere
def prepare_data(S2):
    
    A0 = (1/4)*np.abs(S2[0,0]+S2[1,1])**2
    B0 = (1/4)*np.abs(S2[0,0]-S2[1,1])**2 + np.abs(S2[0,1])**2
    B_psi = (1/4)*np.abs(S2[0,0]-S2[1,1])**2 - np.abs(S2[0,1])**2
    # C_psi = (1/2)*np.abs(S2[0,0]-S2[1,1])**2 
    # di-hedral fix convert real part of vv to positive
    C_psi = (1/2) * np.abs(S2[0,0] - (np.real(S2[1,1]) * (1 if np.real(S2[1,1]) >= 0 else -1) + 1j * np.imag(S2[1,1])))**2
    D_psi = np.imag(S2[0,0]*np.conjugate(S2[1,1]))
    E_psi = np.real(np.conjugate(S2[0,1])*(S2[0,0]-S2[1,1]))
    F_psi = np.imag(np.conjugate(S2[0,1])*(S2[0,0]-S2[1,1]))
    G_psi = np.imag(np.conjugate(S2[0,1])*(S2[0,0]+S2[1,1]))
    H_psi = np.real(np.conjugate(S2[0,1])*(S2[0,0]+S2[1,1]))
    
    K_mat = np.array([
        [A0+B0,C_psi,H_psi,F_psi],
        [C_psi,A0+B_psi,E_psi,G_psi],
        [H_psi,E_psi,A0-B_psi,D_psi],
        [F_psi,G_psi,D_psi,-A0+B0]
        ]) 
    XX_POL = []
    cp_sign = np.zeros((181,91))
    xp_sign = np.zeros((181,91))    
    x, y = np.mgrid[-90: 91 : 1, -45 : 46 : 1]
    
    xx = 0
    for tilt in np.arange(-90,91,1):
        yy=0
        for elep in np.arange(-45,46,1):
            psi = tilt*np.pi/180
            tau = elep*np.pi/180
            
            S_co = np.array([1,
                             np.cos(2*psi)*np.cos(2*tau),
                             np.sin(2*psi)*np.cos(2*tau), 
                             np.sin(2*tau)
                             ])
            stc = np.conjugate(S_co.T)
            temp_pow_c = np.matmul(stc,np.matmul(K_mat,S_co))
            S_cross = np.array([1,
                                -np.cos(2*psi)*np.cos(2*tau),
                                -np.sin(2*psi)*np.cos(2*tau), 
                                -np.sin(2*tau)
                                ])
            stx = np.conjugate(S_cross.T)
            temp_pow_x = np.matmul(stx,np.matmul(K_mat,S_co))
            
            
            XX_POL.append([tilt,elep,temp_pow_c,temp_pow_x])
            cp_sign[xx,yy] = temp_pow_c
            xp_sign[xx,yy] = temp_pow_x
            
            yy=yy+1  
            # print(xx,yy)
        xx=xx+1
    # cp_sign[cp_sign==0]=np.nan
    # xp_sign[xp_sign==0]=np.nan
    
    cp_sign = cp_sign/np.nanmax(cp_sign)
    xp_sign = xp_sign/np.nanmax(xp_sign)
    
    return cp_sign, xp_sign

def prepare_dataT3(T3):
    
    # Define angular grids
    phi_deg = np.linspace(-90, 90, 181)    
    tau_deg = np.linspace(-45, 45, 91)       
    phi = np.deg2rad(phi_deg)
    tau = np.deg2rad(tau_deg)
    
    # Create output arrays
    cp_sign = np.zeros((len(phi), len(tau)))
    xp_sign = np.zeros((len(phi), len(tau)))
    
    
    T11     = T3[0, 0].real
    T12_re  = T3[0, 1].real
    T12_im  = T3[0, 1].imag
    T13_re  = T3[0, 2].real
    T13_im  = T3[0, 2].imag
    T22     = T3[1, 1].real
    T23_re  = T3[1, 2].real
    T23_im  = T3[1, 2].imag
    T33     = T3[2, 2].real
    
    
    for i_phi, Phi in enumerate(phi):
        cos2P, sin2P, sin4P, cos4P = np.cos(2*Phi), np.sin(2*Phi), np.sin(4*Phi), np.cos(4*Phi)
    
        # Phi rotation
        T12_re_phi =  T12_re * cos2P + T13_re * sin2P
        # T12_im_phi =  T12_im * cos2P + T13_im * sin2P
        # T13_re_phi = -T12_re * sin2P + T13_re * cos2P
        T13_im_phi = -T12_im * sin2P + T13_im * cos2P
        T22_phi = T22 * cos2P**2 + T23_re * sin4P + T33 * sin2P**2
        # T23_re_phi = 0.5 * (T33 - T22) * sin4P + T23_re * cos4P
        T23_im_phi = T23_im
        T33_phi = T22 * sin2P**2 - T23_re * sin4P + T33 * cos2P**2
    
        for j_tau, Tau in enumerate(tau):
            cos2T, sin2T, sin4T, cos4T = np.cos(2*Tau), np.sin(2*Tau), np.sin(4*Tau), np.cos(4*Tau)
    
            # Tau rotation
            T311 = T11 * cos2T**2 + T13_im_phi * sin4T + T33_phi * sin2T**2
            T312_re = T12_re_phi * cos2T + T23_im_phi * sin2T
            T322 = T22_phi
            T333 = T11 * sin2T**2 - T13_im_phi * sin4T + T33_phi * cos2T**2
    
            # Output values
            cp_sign[i_phi, j_tau] = (T311 + 2 * T312_re + T322) / 2.0
            xp_sign[i_phi, j_tau] = T333 / 2.0
            
    
    cp_sign = cp_sign/np.nanmax(cp_sign)
    xp_sign = xp_sign/np.nanmax(xp_sign)
    return cp_sign, xp_sign


def signature_fp(mat=None, title='',pname='',cmap='jet',plotType = 1,
    fig=None, axes=None, start_index=0):
    """
    Generates and visualizes polarimetric signatures from a 2x2 scattering matrix (S2) or a 3x3 coherency matrix (T3).
    
    Examples
    --------
    >>> mat = np.array([[1, 0], [0, 1]]) # Trihedral
    >>> signature_fp(mat, title='Trihedral', plotType=1)

    Parameters
    ----------
    mat : np.ndarray (2x2 or 3x3) or None
        Input matrix. If shape is (2, 2), treated as S2. If (3, 3), treated as T3.
        If None, only theoretical/canonical plots will be generated.

    title : str, optional
        Title of the plot. Default is an empty string.

    pname : str, optional
        Name of the output file (*.png). Default is an empty string.

    cmap : str, optional
        Colormap used for visualizing the data. Default is 'jet'.

    plotType : int, optional
        Determines the type of plot to generate:
        
        - 1: Standard polarimetric signature via `pol_sign`.
        - 2: 2D polarimetric signature via `pol_sign2d`.
        - 3: Poincaré sphere mapping via `poincare_plot`.
        - 4: Render empty or canonical Poincaré sphere via `poincare_sphere` (S2 not required).
    """

    if mat is not None:
        if mat.shape == (2, 2):
            cp_sign, xp_sign = prepare_data(mat)  # From S2
        elif mat.shape == (3, 3):
            if not np.allclose(mat, mat.conj().T, atol=1e-8):
                print("Warning: Input matrix is not Hermitian. Results may be inaccurate.")
            cp_sign, xp_sign = prepare_dataT3(mat)  # From T3
        else:
            raise ValueError("Input matrix must be either 2x2 (S2) or 3x3 (T3).")

    # Choose visualization method
    if plotType == 4:
        poincare_sphere(pname=pname)
    elif mat is not None and plotType == 3:
        poincare_plot(cp_sign, xp_sign, title=title, pname=pname, cmap=cmap)
    elif mat is not None and plotType == 2:
        pol_sign2d(cp_sign, xp_sign, title=title, pname=pname, cmap=cmap)
    else:
        figout, axesout = pol_sign(cp_sign, xp_sign, title=title, pname=pname,
                                   cmap=cmap, fig=fig, axes=axes, start_index=start_index)
        return figout, axesout



# S2 = np.array([[1,0],
#                [0,1]])

# T3 = np.array([
#     [2.0, 0.0, 0.0],
#     [0.0, 0.0, 0.0],
#     [0.0, 0.0, 0.0]
# ])

# # fix this oriented dihedral
# S2 = np.array([[1, 0],
#                [0,-1]])

# S2 = np.array([[0,1],
#                [1,0]])


# S2 = np.array([[0,0],
#                [0,1]])

# S2 = np.array([[1,-1],
#                [-1,1]])

# array([[2248.6506   -5218.124j   ,   -4.0451455  -46.991398j],
#     [ -11.6204195  -27.771238j,  800.1831   -5830.6836j  ]],
#     dtype=complex64)


# fp_sign(S2, plotType = 3)
    

