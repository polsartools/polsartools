import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from polsartools.utils.utils import read_bin

def get_dp_halpha_bounds():
    datal = []
    for m in np.arange(0,1.01,0.01):
    
        data = np.array([[1,0],
                        [0,m]])
        evals, evecs = np.linalg.eig(data)
            
        evals[evals <0] = 0
        evals[evals >1] = 1
        eval_norm1 = (evals[1])/(evals[0] + evals[1])   
        eval_norm2 = (evals[0])/(evals[0] + evals[1])

        
        # # %Alpha 1
        eig_vec_r1 = np.real(evecs[0,1])
        eig_vec_c1 = np.imag(evecs[0,1])
        alpha1 = np.arccos(np.sqrt(eig_vec_r1*eig_vec_r1 + eig_vec_c1*eig_vec_c1))*180/np.pi
        
        # # %Alpha 2
        eig_vec_r2 = np.real(evecs[0,0])
        eig_vec_c2 = np.imag(evecs[0,0])
        alpha2 = np.arccos(np.sqrt(eig_vec_r2*eig_vec_r2 + eig_vec_c2*eig_vec_c2))*180/np.pi
        alpha_l = (eval_norm1*alpha1 + eval_norm2*alpha2)
        Hl = - eval_norm1*np.log10(eval_norm1)/np.log10(2) - eval_norm2*np.log10(eval_norm2)/np.log10(2)
        
        
        data = np.array([[m,0],
                        [0,1]])
        evals, evecs = np.linalg.eig(data)
            
            
        evals[evals <0] = 0
        evals[evals >1] = 1
        
        eval_norm1 = (evals[1])/(evals[0] + evals[1])
        eval_norm2 = (evals[0])/(evals[0] + evals[1])
        
        # # %Alpha 1
        eig_vec_r1 = np.real(evecs[0,1])
        eig_vec_c1 = np.imag(evecs[0,1])
        alpha1 = np.arccos(np.sqrt(eig_vec_r1*eig_vec_r1 + eig_vec_c1*eig_vec_c1))*180/np.pi
        
        # # %Alpha 2
        eig_vec_r2 = np.real(evecs[0,0])
        eig_vec_c2 = np.imag(evecs[0,0])
        alpha2 = np.arccos(np.sqrt(eig_vec_r2*eig_vec_r2 + eig_vec_c2*eig_vec_c2))*180/np.pi
        alpha_u = (eval_norm1*alpha1 + eval_norm2*alpha2)
        Hu = - eval_norm1*np.log10(eval_norm1)/np.log10(2) - eval_norm2*np.log10(eval_norm2)/np.log10(2)
        
       
        datal.append([np.float16(Hl),np.float16(alpha_l),np.float16(alpha_u)])
        
    datal = np.array(datal)
    datal = np.nan_to_num(datal)
    return datal


def get_feas_bounds():
    """
    Hard coded feseable boundary curve values for computaional efficiency. 
    One can obtain these values by calling the get_dp_halpha_bounds function.
    Returns:
    --------
    data : numpy.ndarray
        Array of boundary curve values.

    """
    data = np.array([[0.0000e+00, 0.0000e+00, 9.0000e+01],
           [8.0139e-02, 8.9111e-01, 8.9125e+01],
           [1.3928e-01, 1.7646e+00, 8.8250e+01],
           [1.8994e-01, 2.6211e+00, 8.7375e+01],
           [2.3523e-01, 3.4609e+00, 8.6562e+01],
           [2.7612e-01, 4.2852e+00, 8.5688e+01],
           [3.1372e-01, 5.0938e+00, 8.4875e+01],
           [3.4863e-01, 5.8867e+00, 8.4125e+01],
           [3.8086e-01, 6.6680e+00, 8.3312e+01],
           [4.1113e-01, 7.4297e+00, 8.2562e+01],
           [4.3945e-01, 8.1797e+00, 8.1812e+01],
           [4.6606e-01, 8.9219e+00, 8.1062e+01],
           [4.9121e-01, 9.6406e+00, 8.0375e+01],
           [5.1514e-01, 1.0352e+01, 7.9625e+01],
           [5.3760e-01, 1.1055e+01, 7.8938e+01],
           [5.5859e-01, 1.1742e+01, 7.8250e+01],
           [5.7861e-01, 1.2414e+01, 7.7562e+01],
           [5.9814e-01, 1.3078e+01, 7.6938e+01],
           [6.1621e-01, 1.3727e+01, 7.6250e+01],
           [6.3330e-01, 1.4367e+01, 7.5625e+01],
           [6.4990e-01, 1.5000e+01, 7.5000e+01],
           [6.6553e-01, 1.5617e+01, 7.4375e+01],
           [6.8066e-01, 1.6234e+01, 7.3750e+01],
           [6.9531e-01, 1.6828e+01, 7.3188e+01],
           [7.0898e-01, 1.7422e+01, 7.2562e+01],
           [7.2217e-01, 1.8000e+01, 7.2000e+01],
           [7.3438e-01, 1.8578e+01, 7.1438e+01],
           [7.4658e-01, 1.9141e+01, 7.0875e+01],
           [7.5781e-01, 1.9688e+01, 7.0312e+01],
           [7.6904e-01, 2.0234e+01, 6.9750e+01],
           [7.7930e-01, 2.0766e+01, 6.9250e+01],
           [7.8955e-01, 2.1297e+01, 6.8688e+01],
           [7.9883e-01, 2.1812e+01, 6.8188e+01],
           [8.0811e-01, 2.2328e+01, 6.7688e+01],
           [8.1738e-01, 2.2828e+01, 6.7188e+01],
           [8.2568e-01, 2.3328e+01, 6.6688e+01],
           [8.3398e-01, 2.3828e+01, 6.6188e+01],
           [8.4180e-01, 2.4312e+01, 6.5688e+01],
           [8.4912e-01, 2.4781e+01, 6.5188e+01],
           [8.5645e-01, 2.5250e+01, 6.4750e+01],
           [8.6328e-01, 2.5719e+01, 6.4312e+01],
           [8.6963e-01, 2.6172e+01, 6.3844e+01],
           [8.7598e-01, 2.6625e+01, 6.3375e+01],
           [8.8232e-01, 2.7062e+01, 6.2938e+01],
           [8.8818e-01, 2.7500e+01, 6.2500e+01],
           [8.9355e-01, 2.7938e+01, 6.2062e+01],
           [8.9893e-01, 2.8359e+01, 6.1656e+01],
           [9.0430e-01, 2.8781e+01, 6.1219e+01],
           [9.0918e-01, 2.9188e+01, 6.0812e+01],
           [9.1357e-01, 2.9594e+01, 6.0406e+01],
           [9.1846e-01, 3.0000e+01, 6.0000e+01],
           [9.2285e-01, 3.0391e+01, 5.9594e+01],
           [9.2676e-01, 3.0797e+01, 5.9219e+01],
           [9.3066e-01, 3.1172e+01, 5.8812e+01],
           [9.3457e-01, 3.1562e+01, 5.8438e+01],
           [9.3848e-01, 3.1938e+01, 5.8062e+01],
           [9.4189e-01, 3.2312e+01, 5.7688e+01],
           [9.4531e-01, 3.2688e+01, 5.7312e+01],
           [9.4824e-01, 3.3031e+01, 5.6969e+01],
           [9.5166e-01, 3.3406e+01, 5.6594e+01],
           [9.5459e-01, 3.3750e+01, 5.6250e+01],
           [9.5703e-01, 3.4094e+01, 5.5906e+01],
           [9.5996e-01, 3.4438e+01, 5.5562e+01],
           [9.6240e-01, 3.4781e+01, 5.5219e+01],
           [9.6484e-01, 3.5125e+01, 5.4875e+01],
           [9.6729e-01, 3.5469e+01, 5.4531e+01],
           [9.6973e-01, 3.5781e+01, 5.4219e+01],
           [9.7168e-01, 3.6094e+01, 5.3906e+01],
           [9.7363e-01, 3.6438e+01, 5.3562e+01],
           [9.7559e-01, 3.6750e+01, 5.3250e+01],
           [9.7754e-01, 3.7062e+01, 5.2938e+01],
           [9.7900e-01, 3.7375e+01, 5.2625e+01],
           [9.8096e-01, 3.7688e+01, 5.2312e+01],
           [9.8242e-01, 3.7969e+01, 5.2031e+01],
           [9.8389e-01, 3.8281e+01, 5.1719e+01],
           [9.8535e-01, 3.8562e+01, 5.1438e+01],
           [9.8633e-01, 3.8875e+01, 5.1125e+01],
           [9.8779e-01, 3.9156e+01, 5.0844e+01],
           [9.8877e-01, 3.9438e+01, 5.0562e+01],
           [9.9023e-01, 3.9719e+01, 5.0281e+01],
           [9.9121e-01, 4.0000e+01, 5.0000e+01],
           [9.9219e-01, 4.0281e+01, 4.9719e+01],
           [9.9316e-01, 4.0562e+01, 4.9438e+01],
           [9.9365e-01, 4.0812e+01, 4.9188e+01],
           [9.9463e-01, 4.1094e+01, 4.8906e+01],
           [9.9512e-01, 4.1344e+01, 4.8656e+01],
           [9.9609e-01, 4.1625e+01, 4.8375e+01],
           [9.9658e-01, 4.1875e+01, 4.8125e+01],
           [9.9707e-01, 4.2125e+01, 4.7875e+01],
           [9.9756e-01, 4.2375e+01, 4.7625e+01],
           [9.9805e-01, 4.2625e+01, 4.7375e+01],
           [9.9854e-01, 4.2875e+01, 4.7125e+01],
           [9.9854e-01, 4.3125e+01, 4.6875e+01],
           [9.9902e-01, 4.3375e+01, 4.6625e+01],
           [9.9951e-01, 4.3594e+01, 4.6406e+01],
           [9.9951e-01, 4.3844e+01, 4.6156e+01],
           [9.9951e-01, 4.4094e+01, 4.5906e+01],
           [1.0000e+00, 4.4312e+01, 4.5688e+01],
           [1.0000e+00, 4.4531e+01, 4.5469e+01],
           [1.0000e+00, 4.4781e+01, 4.5219e+01],
           [1.0000e+00, 4.5000e+01, 4.5000e+01]], dtype=np.float16)
    
    return data
    
def plot_h_alpha_dp(h, alpha, ppath=None, cmap='viridis', colorbar=True, norm='', grey_region=True, gridsize=300):
    """
    Generates and saves a hexbin density plot of entropy (H) versus alpha (degrees) for dual-pol data.

    Example:
    --------
    >>> plot_h_alpha_dp(h, alpha, ppath="HA_plot.png", cmap='jet', colorbar=True, norm='log')
    This will generates a H/Alpha plot  from the input arrays and save it as HA_plot.png, using the 'jet' colormap and logarithmic normalization
    
    Parameters:
    -----------
    h : array-like
        Array representing entropy values.
    alpha : array-like
        Array representing alpha values in degrees.
    ppath : str, optional
        Path to save the generated plot. If a folder is given, the plot is saved as 'halpha_plot_dp.png' inside that folder.
        If the file already exists, it will be overwritten.
    cmap : str, optional
        Colormap used for the hexbin plot. Defaults to 'viridis'.
    colorbar : bool, optional
        If True, displays a colorbar representing sample count. Defaults to True.
    norm : str, optional
        If set to 'log', applies logarithmic normalization to the hexbin plot.
    grey_region : bool, optional
        If True, fills non-feasible regions of the plot with a grey color to indicate feasible boundaries.
        Defaults to True.
    gridsize : int, optional
        Number of hexagonal bins used in the plot. Higher values result in finer binning. Defaults to 300.
    
    Returns:
    --------
    None
        Displays the plot and optionally saves it to the specified location.
    
    Notes:
    ------
    - Uses `get_feas_bounds()` to obtain feasible boundary curves for plotting.
    - If `norm` is 'log', a `LogNorm()` normalization is applied.
    """

    if isinstance(h, (str, os.PathLike, Path)):
        h = read_bin(h)
    if isinstance(alpha, (str, os.PathLike, Path)):
        alpha = read_bin(alpha)
    


    data = get_feas_bounds()
    fs = 12
    
    fig_size = (3.6, 3) if colorbar else (3.2, 3)
    fig, ax = plt.subplots(figsize=fig_size, dpi=300)
    norm_option = mcolors.LogNorm() if norm == 'log' else None
    
    
    plt.hexbin(h.flatten(), alpha.flatten(), gridsize=gridsize, cmap=cmap,mincnt=1,norm=norm_option)

    plt.plot(data[:,0],data[:,1],'k-',linewidth=0.3)
    plt.plot(data[:,0],data[:,2],'k-',linewidth=0.3)
    plt.yticks(np.arange(0,100,10),np.arange(0,100,10),fontsize=fs)
    plt.xticks(np.round(np.arange(0,1.2,.2),2),np.round(np.arange(0,1.2,.2),2),fontsize=fs)
    plt.xlim([0,1])
    plt.ylim([0,90])
    plt.xlabel('Entropy, H',fontsize=fs)
    plt.ylabel(r'Alpha (degrees)',fontsize=fs)

    ax.tick_params(axis='both', which='both', direction='in',
                    top=True, bottom=True, left=True, right=True)
    
    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', direction='in',
                   top=True, bottom=True, left=True, right=True,
                   length=4, width=0.5)
    ax.tick_params(axis='both', which='minor', length=2, width=0.2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    if colorbar:
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=fs-4)
        cbar.set_label(label='#samples',fontsize=fs-4)

    if grey_region:
        xs,ys = data[:,0],data[:,1]
        ax.fill_between(xs, ys, interpolate=True, color='#dbdbdb',zorder=0)
        xs,ys = data[:,0],data[:,2]
        ax.fill_between(xs, ys, 90, interpolate=True, color='#dbdbdb',zorder=0)
    
    plt.tight_layout()
    if ppath is not None:
        if os.path.isdir(ppath):
            ppath = os.path.join(ppath, 'halpha_plot_dp.png')  
            
        plt.savefig(ppath,dpi=300,bbox_inches='tight')
