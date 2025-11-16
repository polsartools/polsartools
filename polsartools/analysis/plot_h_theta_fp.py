
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
from pathlib import Path
from osgeo import gdal
gdal.UseExceptions()
plt.rcParams.update({'font.size': 8.5})

from polsartools.utils.utils import read_bin

def h_log(p):
    return p * np.log(p) / np.log(3) if p > 0 else 0
def get_bounds():
    #%% Theoretical boundary
    # m=0
    
    c1l=[]
    for m in np.arange(0,1.005,0.005):
        T31 = np.array([[1,0,0],[0,m,0],[0,0,m]])
        # mfp = np.sqrt(1-27*np.linalg.det(T31)/(np.trace(T31))**3)
        
        mfp = np.sqrt(1-((27*np.linalg.det(T31))/(np.trace(T31))**3))
        
        tfp1 = 2*np.arctan(mfp*np.trace(T31)*(1-m-m)/(1*(m+m)+np.trace(T31)**2*mfp**2))*180/np.pi
        
        eigenValues, eigenVectors = np.linalg.eig(T31)
        idx = eigenValues.argsort()[::-1]   
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:,idx]
        
        p1 = eigenValues[0]/eigenValues.sum()
        p2 = eigenValues[1]/eigenValues.sum()
        p3 = eigenValues[2]/eigenValues.sum()
        
        # print(m,p1,p2,p3)        
        # h1=-(p1*np.log(p1) / np.log(3)+ p2*np.log(p2) / np.log(3) +p3*np.log(p3) / np.log(3))
        h1 = -(h_log(p1) + h_log(p2) + h_log(p3))
        if tfp1==90:
            h1=0
            
        c1l.append([tfp1,1-h1,mfp])

    c2l=[[-90,1]]
    c2l=[]

    for m in np.arange(0.5,1.005,0.005):
        T31 = np.array([[2*m-1,0,0],[0,1,0],[0,0,1]])
        # mfp = np.sqrt(1-27*np.linalg.det(T31)/(np.trace(T31))**3)
        
        mfp = np.sqrt(1-((27*np.linalg.det(T31))/(np.trace(T31))**3))
        
        tfp1 = 2*np.arctan(mfp*np.trace(T31)*(2*m-1-1-1)/((2*m-1)*(1+1)+np.trace(T31)**2*mfp**2))*180/np.pi
        
        eigenValues, eigenVectors = np.linalg.eig(T31)
        idx = eigenValues.argsort()[::-1]   
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:,idx]
        
        p1 = eigenValues[0]/eigenValues.sum()
        p2 = eigenValues[1]/eigenValues.sum()
        p3 = eigenValues[2]/eigenValues.sum()
                
        # h1=-(p1*np.log(p1) / np.log(3)+ p2*np.log(p2) / np.log(3) +p3*np.log(p3) / np.log(3))
        
        h1 = -(h_log(p1) + h_log(p2) + h_log(p3))
        # if tfp1==0:
            # h1=0
            
        c2l.append([tfp1,1-h1,mfp])

    # c2l=[[-90,1],[-60,0.5],[-20,0.3],[0,0]]

    # curve-II
    c2l = np.array(c2l)
    c2l = c2l[~np.isnan(c2l).any(axis=1)]


    return c1l,c2l
    
#%%
def plot_h_theta_fp(h,theta, pname=None, cmap='jet', 
                   cbar=True, norm='', vmin=None,vmax=None, 
                    grey_region=True, zone_lines=True,
                    zone_line_color='k',zone_ids=True,gridsize=200):
        
    """
    Generates and saves a density plot of entropy (H) versus theta (degrees) for full-pol data,
    including optional zone lines, zone IDs, and grey regions.
    
    radial axis is 1-H
    angular axis is 2*theta
    
    Example:
    --------
    >>> plot_h_theta_fp(h, theta, path="HT_plot.png", cmap='jet', cbar=True, norm='log')
    This will generates a H/Theta plot  from the input arrays and save it as HT_plot.png, using the 'jet' colormap and logarithmic normalization
    
    Parameters:
    -----------
    h : path or array-like
        path to the Entropy file or Array representing entropy values.
    theta : path or array-like
        path to the Theta file or Array representing theta values in degrees.
    pname : str, optional
        Path to save the generated plot. If a folder is given, the plot is saved as 'htheta_plot_fp.png' inside that folder.
        If the file already exists, it will be overwritten.
    cmap : str, optional
        Colormap used for the hexbin plot. Defaults to 'jet'.
    colorbar : bool, optional
        If True, displays a colorbar representing sample count. Defaults to True.
    norm : str, optional
        If set to 'log', applies logarithmic normalization to the hexbin plot.
    grey_region : bool, optional
        If True, fills non-feasible regions of the plot with a grey color to indicate feasible boundaries. Defaults to True.
    zone_lines : bool, optional
        If True, adds dashed lines to mark different entropy-alpha zones. Defaults to True.
    zone_line_color : str, optional
        Color used for zone boundary lines. Defaults to 'k' (black).
    zone_ids : bool, optional
        If True, labels predefined zones with numerical identifiers. Defaults to True.
    gridsize : int, optional
        Number of hexagonal bins used in the plot. Higher values result in finer binning. Defaults to 200.
    
    Returns:
    --------
    None
        Displays the plot and optionally saves it to the specified location.
    
    """
    
    
    if isinstance(h, (str, os.PathLike, Path)):
        y = read_bin(h)
    if isinstance(theta, (str, os.PathLike, Path)):
        x = read_bin(theta)
            
    x = 2*x*np.pi/180
    y = 1-y
    x = x.flatten()
    y = y.flatten()
    mask = np.isfinite(x) & np.isfinite(y)

    lw=0.3
    if cbar:
        fig = plt.figure(figsize=(3.3,2.5),dpi=300)
        ax = plt.subplot(111, polar=True)
    else:
        fig = plt.figure(figsize=(3,2.3),dpi=300)
        ax = plt.subplot(111, polar=True)


    h = ax.hist2d(x[mask], y[mask], bins=gridsize)

    counts = h[0]
    counts_masked = np.ma.masked_where(counts == 0, counts)
    ax.clear()
    norm_option = mcolors.LogNorm() if norm == 'log' else None
    if vmin is not None and vmax is not None:
        norm_option = mcolors.Normalize(vmin=vmin, vmax=vmax) if norm == '' else norm_option
    if vmin is not None and vmax is None:
        norm_option = mcolors.Normalize(vmin=vmin) if norm == '' else norm_option
    if vmin is None and vmax is not None:
        norm_option = mcolors.Normalize(vmax=vmax) if norm == '' else norm_option


    pcm = ax.pcolormesh(h[1], h[2], counts_masked.T, cmap=cmap, norm=norm_option)


    if cbar:
        fig.colorbar(pcm, ax=ax, label='# samples', pad=0.2, shrink=0.6)

    ax.set_theta_zero_location('N')
    ax.set_theta_direction('clockwise')
    if zone_lines:
        zone_line_color=zone_line_color
        ax.plot((0,20*np.pi/180),(0,1),color=zone_line_color,linestyle='--',linewidth = lw)
        ax.plot((0,-10*np.pi/180),(0,1),color=zone_line_color,linestyle='--',linewidth = lw)
        ax.plot((0,0),(0,1),color=zone_line_color,linestyle='--',linewidth = lw)
        theta_line = np.linspace(-np.pi,np.pi)
        r = np.repeat(0.3, np.size(theta_line), axis=None)
        ax.plot(theta_line, r, color=zone_line_color,linestyle='--',linewidth = lw)
        r = np.repeat(0.5, np.size(theta_line), axis=None)
        ax.plot(theta_line, r, color=zone_line_color,linestyle='--',linewidth = lw)

    if grey_region:
        c1l,c2l = get_bounds()
        # curve-I
        ax.plot((np.array(c1l)[:,0]*np.pi/180),np.array(c1l)[:,1],'k-',linewidth = lw)
        ax.fill_between((np.array(c1l)[:,0]*np.pi/180),np.array(c1l)[:,1],color='#bababa', 
                        linewidth=0,
                        zorder=10)
        # curve-II
        ax.plot((np.array(c2l)[:,0]*np.pi/180),np.array(c2l)[:,1],'k-',linewidth = lw)
        ax.fill_between((np.array(c2l)[:,0]*np.pi/180),np.array(c2l)[:,1],color='#bababa', 
                        linewidth=0,
                        zorder=10)

    #Curve-III
    # ax.plot(np.repeat(-90*np.pi/180,np.size(np.arange(0.34,1.1,.1))),np.arange(0.34,1.1,.1),'k-',linewidth = lw,zorder=0)


    if cbar:

        ax.text(0.44, 0.16,' 'r'{:.1f}' ' {:.1f}'' {:.1f}''      {:.1f}'.format(0.0,0.3,0.5,1.0),
                transform=ax.transAxes,
                )
        # #left H labels 
        ax.text(-0.07, 0.16, ' 'r'{:.1f}' '     {:.1f}'' {:.1f}'.format(1.0,0.5,0.3),
            transform=ax.transAxes)
        ax.text(0.7, 0.05,'  'r'$\overline{H}$', transform=ax.transAxes  )
        
    else:
        ax.text(0.44, 0.18,' 'r'{:.1f}' '    {:.1f}'' {:.1f}''      {:.1f}'.format(0.0,0.3,0.5,1.0),
                transform=ax.transAxes,            )
        # #left H labels 
        ax.text(-0.07, 0.18, '  'r'{:.1f}' '       {:.1f}'' {:.1f}'.format(1.0,0.5,0.3),
            transform=ax.transAxes)
        ax.text(0.7, 0.1,'  'r'$\overline{H}$',transform=ax.transAxes)


    ax.text(0.45, 0.9,'  'r'$\overline{\theta}_{\mathrm{FP}}$',
            transform=ax.transAxes,
            )

    ax.set_ylim(0,1)
    ax.xaxis.set_tick_params(pad=0) 
    ax.set_yticks(np.array([]))
    ax.set_xticks(np.array([-90, -10, 0, 20, 90])/180*np.pi)
    ax.set_thetalim(-1/2*np.pi, 1/2*np.pi)

    for side in ax.spines.keys(): 
        ax.spines[side].set_linewidth(.7)

    ax.annotate('',
                xy=(0.39,0.88), xycoords='axes fraction',
                xytext=(0.48,.91), textcoords='axes fraction',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="angle3,angleA=0,angleB=-150",
                                # connectionstyle="arc3,rad=0.2",
                                color='k',
                                linewidth=0.3))

    ax.annotate('',
                xy=(0.66,0.88), xycoords='axes fraction',
                xytext=(0.58,0.91), textcoords='axes fraction',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="angle3,angleA=0,angleB=-30",
                                color='k',
                                linewidth=0.3)) 

    # radial_ticks = [0.2, 0.4, 0.6, 0.8, 1.0]
    # ax.set_yticks(radial_ticks)
    # ax.set_yticklabels([''] * len(radial_ticks)) 
    # for r in radial_ticks:
    #     ax.plot([np.radians(-90), np.radians(90)], [r, r], linestyle='--', color='gray', linewidth=0.5)



    plt.grid(False)
    plt.tight_layout()
    fig.tight_layout()

    if pname is not None:
        if os.path.isdir(pname):
            pname = os.path.join(pname, 'htheta_plot_fp.png')  
            
        plt.savefig(pname,dpi=300,bbox_inches='tight')

