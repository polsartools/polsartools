
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm
import os
from pathlib import Path
from osgeo import gdal
from polsartools.utils.utils import read_bin, time_it
gdal.UseExceptions()
plt.rcParams.update({'font.size': 8.5})

from polsartools.utils.utils import read_bin

def h_log(p):
    return p * np.log(p) / np.log(3) if p > 0 else 0
def get_bounds():
    #%% Theoretical boundary
    # m=0
    
    c1l=[]
    for m in np.arange(0,0.505,0.005):
        chi = -45
        C21 = np.array([[(2*m+1)/4, 1j*(2*m-1)/4 ],
                        [-1j*(2*m-1)/4,(2*m+1)/4]])
        # mfp = np.sqrt(1-27*np.linalg.det(T31)/(np.trace(T31))**3)
        
        c11s = C21[0,0]
        c22s = C21[1,1]
        c12s = C21[0,1]
        c21s = C21[1,0]
    
        c2_detr = (c11s*c22s-c12s*c21s)
        c2_trace = c11s+c22s
        # t2_span = t11s*t22s
        mcp = np.real(np.sqrt(1.0-(4.0*c2_detr/np.power(c2_trace,2))))
    
        # Stokes Parameter
        s0  = c11s + c22s;
        s1 = c11s - c22s;
        s2 = (c12s + c21s);
    
        if (chi >= 0):
            s3 = (1j*(c12s - c21s)); # The sign is according to RC or LC sign !!
        if (chi < 0):
            s3 = -(1j*(c12s - c21s)); # The sign is according to RC or LC sign !!
        
        SC = ((s0)-(s3))/2;
        OC = ((s0)+(s3))/2;
    
        h = (OC-SC)
        span = c11s + c22s
    
        val = ((mcp*s0*h))/((SC*OC + (mcp**2)*(s0**2)))
        tcp1 = np.real(np.arctan(val))
        # tcp1 = np.rad2deg(thet) 
        
        eigenValues, eigenVectors = np.linalg.eig(C21)
        idx = eigenValues.argsort()[::-1]   
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:,idx]
        
        p1 = eigenValues[0]/eigenValues.sum()
        p2 = eigenValues[1]/eigenValues.sum()

                
        h1=-(p1*np.log(p1) / np.log(2)+ p2*np.log(p2) / np.log(2) )
        h1 = np.real(h1)
        
        if np.isnan(h1):
            h1=0       
        c1l.append([tcp1,1-h1,mcp])
    
    # curve-I

    c2l=[]
    for m in np.arange(0,0.505,0.005):
        C21 = np.array([[(2*m+1)/4, -1j*(2*m-1)/4 ],
                        [1j*(2*m-1)/4,(2*m+1)/4]])
        # mfp = np.sqrt(1-27*np.linalg.det(T31)/(np.trace(T31))**3)
        
        c11s = C21[0,0]
        c22s = C21[1,1]
        c12s = C21[0,1]
        c21s = C21[1,0]
    
        c2_detr = (c11s*c22s-c12s*c21s)
        c2_trace = c11s+c22s
        # t2_span = t11s*t22s
        mcp = np.real(np.sqrt(1.0-(4.0*c2_detr/np.power(c2_trace,2))))
    
        # Stokes Parameter
        s0  = c11s + c22s;
        s1 = c11s - c22s;
        s2 = (c12s + c21s);
    
        if (chi >= 0):
            s3 = (1j*(c12s - c21s)); # The sign is according to RC or LC sign !!
        if (chi < 0):
            s3 = -(1j*(c12s - c21s)); # The sign is according to RC or LC sign !!
        
        SC = ((s0)-(s3))/2;
        OC = ((s0)+(s3))/2;
    
        h = (OC-SC)
        span = c11s + c22s
    
        val = ((mcp*s0*h))/((SC*OC + (mcp**2)*(s0**2)))
        tcp1 = np.real(np.arctan(val))
        # tcp1 = np.rad2deg(thet) 
        
        eigenValues, eigenVectors = np.linalg.eig(C21)
        idx = eigenValues.argsort()[::-1]   
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:,idx]
        
        p1 = eigenValues[0]/eigenValues.sum()
        p2 = eigenValues[1]/eigenValues.sum()

                
        h1=-(p1*np.log(p1) / np.log(2)+ p2*np.log(p2) / np.log(2) )
        h1 = np.real(h1)
        
        if np.isnan(h1):
            h1=0       
        c2l.append([tcp1,1-h1,mcp])  
    
    # curve-II
    c2l = np.array(c2l)
    c2l = c2l[~np.isnan(c2l).any(axis=1)]


    return c1l,c2l
    
#%%
@time_it
def plot_h_theta_cp(h,theta, ppath=None, cmap='jet', 
                   cbar=True, norm='', vmin=None,vmax=None, 
                    grey_region=True, zone_lines=True,
                    zone_line_color='k',zone_ids=True,density=False,gridsize=200, ):
        
    """
    Generates and saves a density plot of entropy (H) versus theta (degrees) for compact-pol data,
    including optional zone lines, zone IDs, and grey regions.
    
    radial axis is 1-H
    angular axis is 2*theta
    
    Example:
    --------
    >>> plot_h_theta_cp(h, theta, ppath="HT_plot.png", cmap='jet', cbar=True, norm='log')
    This will generates a H/Theta plot  from the input arrays and save it as HT_plot.png, using the 'jet' colormap and logarithmic normalization
    
    Parameters:
    -----------
    h : path or array-like
        path to the Entropy file or Array representing entropy values.
    theta : path or array-like
        path to the Theta file or Array representing theta values in degrees.
    ppath : str, optional
        Path to save the generated plot. If a folder is given, the plot is saved as 'htheta_plot_cp.png' inside that folder.
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
    y = read_bin(h) if isinstance(h, (str, os.PathLike, Path)) else np.asarray(h)
    x = read_bin(theta) if isinstance(theta, (str, os.PathLike, Path)) else np.asarray(theta)
            
    x = 2 * x * np.pi / 180
    y = 1 - y
    x = x.flatten()
    y = y.flatten()
    mask = np.isfinite(x) & np.isfinite(y)
    
    x = x[mask]
    y = y[mask]

    lw = 0.3
    if cbar:
        fig = plt.figure(figsize=(3.3, 2.5), dpi=300)
        ax = plt.subplot(111, polar=True)
    else:
        fig = plt.figure(figsize=(3, 2.3), dpi=300)
        ax = plt.subplot(111, polar=True)

    if density:
        # Setup normalization
        norm_option = mcolors.LogNorm() if norm == 'log' else None
        if vmin is not None and vmax is not None:
            norm_option = mcolors.Normalize(vmin=vmin, vmax=vmax) if norm == '' else norm_option
        elif vmin is not None and vmax is None:
            norm_option = mcolors.Normalize(vmin=vmin) if norm == '' else norm_option
        elif vmin is None and vmax is not None:
            norm_option = mcolors.Normalize(vmax=vmax) if norm == '' else norm_option
        # Density plot using hist2d and pcolormesh
        hist_res = ax.hist2d(x, y, bins=gridsize)
        counts = hist_res[0]
        counts_masked = np.ma.masked_where(counts == 0, counts)
        ax.clear()
        
        pcm = ax.pcolormesh(hist_res[1], hist_res[2], counts_masked.T, cmap=cmap, norm=norm_option)
        cbar_mappable = pcm
        cbar_label = '# samples'
    else:
        # 12 Zone classification based on angular and radial bounds
        deg = np.degrees(x)
        r = y

        conditions = [
            # Radial bin 1: [0, 0.3)
            (r >= 0) & (r < 0.3) & (deg >= -90) & (deg < -10),
            (r >= 0) & (r < 0.3) & (deg >= -10) & (deg < 0),
            (r >= 0) & (r < 0.3) & (deg >= 0) & (deg < 20),
            (r >= 0) & (r < 0.3) & (deg >= 20) & (deg <= 90),
            # Radial bin 2: [0.3, 0.5)
            (r >= 0.3) & (r < 0.5) & (deg >= -90) & (deg < -10),
            (r >= 0.3) & (r < 0.5) & (deg >= -10) & (deg < 0),
            (r >= 0.3) & (r < 0.5) & (deg >= 0) & (deg < 20),
            (r >= 0.3) & (r < 0.5) & (deg >= 20) & (deg <= 90),
            # Radial bin 3: [0.5, 1.0]
            (r >= 0.5) & (r <= 1.0) & (deg >= -90) & (deg < -10),
            (r >= 0.5) & (r <= 1.0) & (deg >= -10) & (deg < 0),
            (r >= 0.5) & (r <= 1.0) & (deg >= 0) & (deg < 20),
            (r >= 0.5) & (r <= 1.0) & (deg >= 20) & (deg <= 90),
        ]

        choices_idx = list(range(12))
        zone_indices = np.select(conditions, choices_idx, default=12)

        # 12 Distinct zone colors
        choices_color = [
            '#ffb297', '#ff4f4d', '#c10400', 
            '#efffff',             '#16fcc5', '#b1fb98', 
            '#a3ff85', '#7edda9',             '#319449',
              '#b5befe', '#4e71ff', '#0037fd'
        ]
        zone_cmap = ListedColormap(choices_color)
        zone_norm = BoundaryNorm(np.arange(13) + 0.5, zone_cmap.N)
        for idx, col in enumerate(choices_color):
            mask_zone = (zone_indices == idx)
            if not np.any(mask_zone):
                continue
            ax.scatter(
                x[mask_zone], 
                y[mask_zone], 
                color=col, 
                s=20, 
                zorder=3, 
                edgecolors='none', 
                alpha=0.9
            )

        cbar_mappable = plt.cm.ScalarMappable(cmap=zone_cmap, norm=zone_norm)
        cbar_mappable.set_array([])
        cbar_label = 'Zones'

    if cbar:
        if density:
            fig.colorbar(cbar_mappable, ax=ax, label=cbar_label, pad=0.2, shrink=0.6)
        else:
            cbar_obj = fig.colorbar(cbar_mappable, ax=ax, orientation='vertical', pad=0.2, shrink=0.6, ticks=range(1, 13))
            cbar_obj.ax.set_yticklabels([f'Z{i}' for i in range(1, 13)], fontsize=8)
            cbar_obj.ax.tick_params(size=0)

    ax.set_theta_zero_location('N')
    ax.set_theta_direction('clockwise')
    
    if zone_lines:
        ax.plot((0, 20 * np.pi / 180), (0, 1), color=zone_line_color, linestyle='--', linewidth=lw)
        ax.plot((0, -10 * np.pi / 180), (0, 1), color=zone_line_color, linestyle='--', linewidth=lw)
        ax.plot((0, 0), (0, 1), color=zone_line_color, linestyle='--', linewidth=lw)
        theta_line = np.linspace(-np.pi, np.pi)
        r = np.repeat(0.3, np.size(theta_line), axis=None)
        ax.plot(theta_line, r, color=zone_line_color, linestyle='--', linewidth=lw)
        r = np.repeat(0.5, np.size(theta_line), axis=None)
        ax.plot(theta_line, r, color=zone_line_color, linestyle='--', linewidth=lw)

    if grey_region:
        c1l, c2l = get_bounds()
        # curve-I
        ax.plot((np.array(c1l)[:, 0] - 45 * np.pi / 180), np.array(c1l)[:, 1], 'k-', linewidth=lw)
        ax.fill_between((np.array(c1l)[:, 0] - 45 * np.pi / 180), np.array(c1l)[:, 1], color='#bababa', 
                        linewidth=0, zorder=10)
        # curve-II
        ax.plot((np.array(c2l)[:, 0] + 45 * np.pi / 180), np.array(c2l)[:, 1], 'k-', linewidth=lw)
        ax.fill_between((np.array(c2l)[:, 0] + 45 * np.pi / 180), np.array(c2l)[:, 1], color='#bababa', 
                        linewidth=0, zorder=10)

    if cbar:
        ax.text(0.44, 0.16, ' 'r'{:.1f}' ' {:.1f}'' {:.1f}''      {:.1f}'.format(0.0, 0.3, 0.5, 1.0),
                transform=ax.transAxes)
        # left H labels 
        ax.text(-0.07, 0.16, ' 'r'{:.1f}' '     {:.1f}'' {:.1f}'.format(1.0, 0.5, 0.3),
                transform=ax.transAxes)
        ax.text(0.7, 0.05, '  'r'$\overline{H}$', transform=ax.transAxes)
    else:
        ax.text(0.44, 0.18, ' 'r'{:.1f}' '    {:.1f}'' {:.1f}''      {:.1f}'.format(0.0, 0.3, 0.5, 1.0),
                transform=ax.transAxes)
        # left H labels 
        ax.text(-0.07, 0.18, '  'r'{:.1f}' '       {:.1f}'' {:.1f}'.format(1.0, 0.5, 0.3),
                transform=ax.transAxes)
        ax.text(0.7, 0.1, '  'r'$\overline{H}$', transform=ax.transAxes)

    ax.text(0.45, 0.9, '  'r'$\overline{\theta}_{\mathrm{CP}}$',
            transform=ax.transAxes)

    ax.set_ylim(0, 1)
    ax.xaxis.set_tick_params(pad=0) 
    ax.set_yticks(np.array([]))
    ax.set_xticks(np.array([-90, -10, 0, 20, 90]) / 180 * np.pi)
    ax.set_thetalim(-1 / 2 * np.pi, 1 / 2 * np.pi)

    for side in ax.spines.keys(): 
        ax.spines[side].set_linewidth(.7)

    ax.annotate('',
                xy=(0.39, 0.88), xycoords='axes fraction',
                xytext=(0.48, .91), textcoords='axes fraction',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="angle3,angleA=0,angleB=-150",
                                color='k',
                                linewidth=0.3))

    ax.annotate('',
                xy=(0.66, 0.88), xycoords='axes fraction',
                xytext=(0.58, .91), textcoords='axes fraction',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="angle3,angleA=0,angleB=-30",
                                color='k',
                                linewidth=0.3)) 

    plt.grid(False)
    plt.tight_layout()
    fig.tight_layout()

    if ppath is not None:
        if os.path.isdir(ppath):
            ppath = os.path.join(ppath, 'htheta_plot_cp.png')  
            
        plt.savefig(ppath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Plot saved as '{ppath}'")
    
    # y = read_bin(h) if isinstance(h, (str, os.PathLike, Path)) else np.asarray(h)
    # x = read_bin(theta) if isinstance(theta, (str, os.PathLike, Path)) else np.asarray(theta)
            
    # x = 2*x*np.pi/180
    # y = 1-y
    # x = x.flatten()
    # y = y.flatten()
    # mask = np.isfinite(x) & np.isfinite(y)

    # lw=0.3
    # if cbar:
    #     fig = plt.figure(figsize=(3.3,2.5),dpi=300)
    #     ax = plt.subplot(111, polar=True)
    # else:
    #     fig = plt.figure(figsize=(3,2.3),dpi=300)
    #     ax = plt.subplot(111, polar=True)


    # h = ax.hist2d(x[mask], y[mask], bins=gridsize)

    # counts = h[0]
    # counts_masked = np.ma.masked_where(counts == 0, counts)
    # ax.clear()
    # norm_option = mcolors.LogNorm() if norm == 'log' else None
    # if vmin is not None and vmax is not None:
    #     norm_option = mcolors.Normalize(vmin=vmin, vmax=vmax) if norm == '' else norm_option
    # if vmin is not None and vmax is None:
    #     norm_option = mcolors.Normalize(vmin=vmin) if norm == '' else norm_option
    # if vmin is None and vmax is not None:
    #     norm_option = mcolors.Normalize(vmax=vmax) if norm == '' else norm_option


    # pcm = ax.pcolormesh(h[1], h[2], counts_masked.T, cmap=cmap, norm=norm_option)


    # if cbar:
    #     fig.colorbar(pcm, ax=ax, label='# samples', pad=0.2, shrink=0.6)

    # ax.set_theta_zero_location('N')
    # ax.set_theta_direction('clockwise')
    # if zone_lines:
    #     zone_line_color=zone_line_color
    #     ax.plot((0,20*np.pi/180),(0,1),color=zone_line_color,linestyle='--',linewidth = lw)
    #     ax.plot((0,-10*np.pi/180),(0,1),color=zone_line_color,linestyle='--',linewidth = lw)
    #     ax.plot((0,0),(0,1),color=zone_line_color,linestyle='--',linewidth = lw)
    #     theta_line = np.linspace(-np.pi,np.pi)
    #     r = np.repeat(0.3, np.size(theta_line), axis=None)
    #     ax.plot(theta_line, r, color=zone_line_color,linestyle='--',linewidth = lw)
    #     r = np.repeat(0.5, np.size(theta_line), axis=None)
    #     ax.plot(theta_line, r, color=zone_line_color,linestyle='--',linewidth = lw)

    # if grey_region:
    #     c1l,c2l = get_bounds()
    #     # curve-I
    #     ax.plot((np.array(c1l)[:,0]-45*np.pi/180),np.array(c1l)[:,1],'k-',linewidth = lw)
    #     ax.fill_between((np.array(c1l)[:,0]-45*np.pi/180),np.array(c1l)[:,1],color='#bababa', 
    #                     linewidth=0,
    #                     zorder=10)
    #     # curve-II
    #     ax.plot((np.array(c2l)[:,0]+45*np.pi/180),np.array(c2l)[:,1],'k-',linewidth = lw)
    #     ax.fill_between((np.array(c2l)[:,0]+45*np.pi/180),np.array(c2l)[:,1],color='#bababa', 
    #                     linewidth=0,
    #                     zorder=10)

    # #Curve-III
    # # ax.plot(np.repeat(-90*np.pi/180,np.size(np.arange(0.34,1.1,.1))),np.arange(0.34,1.1,.1),'k-',linewidth = lw,zorder=0)


    # if cbar:

    #     ax.text(0.44, 0.16,' 'r'{:.1f}' ' {:.1f}'' {:.1f}''      {:.1f}'.format(0.0,0.3,0.5,1.0),
    #             transform=ax.transAxes,
    #             )
    #     # #left H labels 
    #     ax.text(-0.07, 0.16, ' 'r'{:.1f}' '     {:.1f}'' {:.1f}'.format(1.0,0.5,0.3),
    #         transform=ax.transAxes)
    #     ax.text(0.7, 0.05,'  'r'$\overline{H}$', transform=ax.transAxes  )
        
    # else:
    #     ax.text(0.44, 0.18,' 'r'{:.1f}' '    {:.1f}'' {:.1f}''      {:.1f}'.format(0.0,0.3,0.5,1.0),
    #             transform=ax.transAxes,            )
    #     # #left H labels 
    #     ax.text(-0.07, 0.18, '  'r'{:.1f}' '       {:.1f}'' {:.1f}'.format(1.0,0.5,0.3),
    #         transform=ax.transAxes)
    #     ax.text(0.7, 0.1,'  'r'$\overline{H}$',transform=ax.transAxes)


    # ax.text(0.45, 0.9,'  'r'$\overline{\theta}_{\mathrm{CP}}$',
    #         transform=ax.transAxes,
    #         )

    # ax.set_ylim(0,1)
    # ax.xaxis.set_tick_params(pad=0) 
    # ax.set_yticks(np.array([]))
    # ax.set_xticks(np.array([-90, -10, 0, 20, 90])/180*np.pi)
    # ax.set_thetalim(-1/2*np.pi, 1/2*np.pi)

    # for side in ax.spines.keys(): 
    #     ax.spines[side].set_linewidth(.7)

    # ax.annotate('',
    #             xy=(0.39,0.88), xycoords='axes fraction',
    #             xytext=(0.48,.91), textcoords='axes fraction',
    #             arrowprops=dict(arrowstyle="->",
    #                             connectionstyle="angle3,angleA=0,angleB=-150",
    #                             # connectionstyle="arc3,rad=0.2",
    #                             color='k',
    #                             linewidth=0.3))

    # ax.annotate('',
    #             xy=(0.66,0.88), xycoords='axes fraction',
    #             xytext=(0.58,0.91), textcoords='axes fraction',
    #             arrowprops=dict(arrowstyle="->",
    #                             connectionstyle="angle3,angleA=0,angleB=-30",
    #                             color='k',
    #                             linewidth=0.3)) 

    # plt.grid(False)
    # plt.tight_layout()
    # fig.tight_layout()

    # if ppath is not None:
    #     if os.path.isdir(ppath):
    #         ppath = os.path.join(ppath, 'htheta_plot_cp.png')  
            
    #     plt.savefig(ppath,dpi=300,bbox_inches='tight')

