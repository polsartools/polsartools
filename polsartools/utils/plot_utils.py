import numpy as np
import os
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8})
# plt.rcParams["font.family"] = "arial"
# plt.rcParams["font.family"] = "sans-serif"
# plt.rcParams["mathtext.fontset"] = "dejavuserif"

def pol_sign(data1, data2, title='', ppath='',cmap='jet',fig=None, axes=None, start_index=0):
    x, y = np.mgrid[-90: 91 : 1, -45 : 46 : 1]
    if axes is None or fig is None:
        fig, axes = plt.subplots(1, 2, figsize=(6, 4), dpi=300, subplot_kw={'projection': '3d'})
    else:
        axes = axes[start_index:start_index+2]

    # fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=300, subplot_kw={'projection': '3d'})

    for ax, data, subtitle in zip(axes, [data1, data2], ["Co-pol","Cross-pol"]):
        ax.plot_surface(x, y, data, rstride=2, cstride=2, lw=0.2, edgecolor="black", cmap=cmap, alpha=0.3)
        ax.contourf(x, y, data, 100, zdir='z', offset=-0.5, cmap=cmap)
        ax.set_zlim([-0.5, 1])
        
        ax.set_xticks([-90, -45, 0, 45, 90])
        ax.set_xticklabels([f"{tick}°" for tick in [-90, -45, 0, 45, 90]])
        ax.tick_params(axis='y', pad=-3) 
        
        ax.set_yticks([-45, -30, -15, 0, 15, 30, 45])
        ax.set_yticklabels([f"{tick}°" for tick in [-45, -30, -15, 0, 15, 30, 45]])
        ax.tick_params(axis='x', pad=-4) 
        
        ax.set_zticks([0,0.2,0.4,0.6,0.8,1.0])
        ax.set_zticklabels([f"{tick}" for tick in [0,0.2,0.4,0.6,0.8,1.0]])
        ax.tick_params(axis='z', pad=-1) 
        
        ax.set_xlabel(r'Orientation, $\psi$ ', labelpad=-6)
        ax.set_ylabel(r'Ellipticity, $\tau$', labelpad=-6)
        ax.set_zlabel('    Normalized power', labelpad=-4)
        
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis._axinfo['grid']['linewidth'] = 0.5
        
        ax.set_title(subtitle)

    plt.suptitle(title)
    plt.tight_layout()
    
    if ppath is not None:
        if os.path.isdir(ppath):
            ppath = os.path.join(ppath, 'pol_signature.png')  
            
        plt.savefig(ppath,dpi=300,bbox_inches='tight')
    
    plt.show()
    return fig, axes

def poincare_plot(cp_sign, xp_sign, title='', ppath='',cmap='jet'):
    
    # Azimuth and elevation grids
    azimuth, elevation = np.mgrid[-180:182:2, -90:92:2]
    azimuth_rad = np.radians(azimuth)
    elevation_rad = np.radians(elevation)
    
    def spherical_to_cartesian(r, azimuth_rad, elevation_rad):
        X = r * np.cos(elevation_rad) * np.cos(azimuth_rad)
        Y = r * np.cos(elevation_rad) * np.sin(azimuth_rad)
        Z = r * np.sin(elevation_rad)
        return X, Y, Z
    
    # Cartesian coordinates for cp_sign and xp_sign
    X1, Y1, Z1 = spherical_to_cartesian(cp_sign, azimuth_rad, elevation_rad)
    X2, Y2, Z2 = spherical_to_cartesian(xp_sign, azimuth_rad, elevation_rad)

    
    fig = plt.figure(figsize=(8, 4),dpi=300)
    # Subplot for cp_sign
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    clrmap = plt.get_cmap(cmap)  
    ax1.plot_surface(X1, Y1, Z1, facecolors=clrmap(cp_sign / np.nanmax(cp_sign)),
                     rstride=1, cstride=1, linewidth=0, antialiased=False,
                     alpha=0.5,
                    )
    ax1.set_title("Co-pol")
    ax1.set_xlim([-1,1])
    ax1.set_ylim([-1,1])
    ax1.set_zlim([-1,1])
    ax1.view_init(elev=30, azim=-135,roll=0)
    ax1.grid(False)
    plt.axis('off')
    # Subplot for xp_sign
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_surface(X2, Y2, Z2, facecolors=clrmap(xp_sign / np.nanmax(xp_sign)),
                     rstride=1, cstride=1, linewidth=0, antialiased=False,
                     alpha=0.5,
                     )
    ax2.set_title("Cross-pol")
    ax2.set_xlim([-1,1])
    ax2.set_ylim([-1,1])
    ax2.set_zlim([-1,1])
    
    ax2.view_init(elev=30, azim=45,roll=0)
    ax1.grid(False) 
    plt.axis('off') 
    
   
    # Add axis arrows and labels for ax1
    arrow_length = 1.5
    ax1.quiver(0, 0, 0, -1*arrow_length, 0, 0, color='k', arrow_length_ratio=0.1,zorder = 0)
    ax1.quiver(0, 0, 0, 0, -1*arrow_length, 0, color='k', arrow_length_ratio=0.1,zorder = 0)
    ax1.quiver(0, 0, 0, 0, 0, arrow_length, color='k', arrow_length_ratio=0.1,zorder = 0)
    ax1.text(-1*arrow_length-0.3, 0, 0, 'S$_1$', color='k', fontsize=12)
    ax1.text(0, -1*arrow_length-0.15, -0.1, 'S$_2$', color='k', fontsize=12)
    ax1.text(-0.1, 0, arrow_length+0.2, 'S$_3$', color='k', fontsize=12)
    
    # Add axis arrows and labels for ax2
    ax2.quiver(0, 0, 0, arrow_length, 0, 0, color='k', arrow_length_ratio=0.1,zorder = 0)
    ax2.quiver(0, 0, 0, 0, arrow_length, 0, color='k', arrow_length_ratio=0.1,zorder = 0)
    ax2.quiver(0, 0, 0, 0, 0, arrow_length, color='k', arrow_length_ratio=0.1,zorder = 0)
    ax2.text(arrow_length+0.3, 0, 0, 'S$_1$', color='k', fontsize=12)
    ax2.text(0, arrow_length+0.15, -0.1, 'S$_2$', color='k', fontsize=12)
    ax2.text(0.1, 0, arrow_length+0.2, 'S$_3$', color='k', fontsize=12)
    
    plt.tight_layout()
    plt.suptitle(title)
    # if pname:
    #     plt.savefig(pname, dpi=300)


    if ppath is not None:
        if os.path.isdir(ppath):
            ppath = os.path.join(ppath, 'poincare_plot.png')  
            
        plt.savefig(ppath,dpi=300,bbox_inches='tight')

def pol_sign2d(cp_sign, xp_sign, title='', ppath='',cmap='jet'):
    x, y = np.mgrid[-90:91:1, -45:46:1]
    extent = [y.min(), y.max(), x.min(), x.max()]
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6.5, 4), dpi=300)  
    
    im1 = axes[0].imshow(cp_sign, extent=extent, cmap=cmap, origin='lower', aspect=1)
    axes[0].set_title('Co-pol')
    axes[0].set_ylabel(r'Orientation, $\psi$')
    axes[0].set_xlabel(r'Ellipticity, $\tau$')
    axes[0].set_xticks([-45, -30, -15, 0, 15, 30, 45])
    axes[0].set_xticklabels([f"{tick}°" for tick in [-45, -30, -15, 0, 15, 30, 45]])
    axes[0].set_yticks([-90, -45, 0, 45, 90])
    axes[0].set_yticklabels([f"{tick}°" for tick in [-90, -45, 0, 45, 90]])
    fig.colorbar(im1, ax=axes[0], label='Normalized power')
    
    im2 = axes[1].imshow(xp_sign, extent=extent, cmap=cmap,origin='lower', aspect=1)
    axes[1].set_title('Cross-pol')
    axes[1].set_ylabel(r'Orientation, $\psi$')
    axes[1].set_xlabel(r'Ellipticity, $\tau$')
    axes[1].set_xticks([-45, -30, -15, 0, 15, 30, 45])
    axes[1].set_xticklabels([f"{tick}°" for tick in [-45, -30, -15, 0, 15, 30, 45]])
    axes[1].set_yticks([-90, -45, 0, 45, 90])
    axes[1].set_yticklabels([f"{tick}°" for tick in [-90, -45, 0, 45, 90]])
    fig.colorbar(im2, ax=axes[1], label='Normalized power')
    plt.suptitle(title)
    plt.tight_layout()

    if ppath is not None:
        if os.path.isdir(ppath):
            ppath = os.path.join(ppath, 'pol_sign2d.png')  
            
        plt.savefig(ppath,dpi=300,bbox_inches='tight')


def poincare_sphere(ppath=''):
     # Axii
    stokes_vectors = np.array([
        [1.8, 0, 0],   # Horizontal linear
        [-1.8, 0, 0],  # Vertical linear
        [0, 1.8, 0],   # 45° linear
        [0, -1.8, 0],  # -45° linear
        [0, 0, 1.8],   # Right circular
        [0, 0, -1.8]   # Left circular
        
    ])
    labels = ['S$_1$','',  '','S$_2$', '','S$_3$']
    # dots
    th_svect = np.array([
    
        [1, 0, 0],    # Horizontal linear
        [-1, 0, 0],   # Vertical linear
        [0, 0, 1],    # Right circular
        [0, 0, -1],   # Left circular
        [0, 1, 0],    # +45 deg linear
        [0, -1, 0]    # -45 deg linear
            
    ])
    # labels for dots
    th_labels = ['H', 'V', 'RCP', 'LCP', '-45°', '+45°']
    
    # Plotting
    fig = plt.figure(figsize=(4, 6),dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a wireframe Poincaré sphere
    u, v = np.linspace(0, 2 * np.pi, 100), np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    
    # Mask the octant: remove points where x > 0, y > 0, z > 0
    mask = ~((x > 0) & (y < 0) & (z < 0))
    
    # Apply the mask: set those values to NaN
    x_masked = np.where(mask, x, np.nan)
    y_masked = np.where(mask, y, np.nan)
    z_masked = np.where(mask, z, np.nan)
    
    # ax.plot_surface(x, y, z, color='lightblue', alpha=0.4, linewidth=0.2, edgecolor='gray')
    ax.plot_surface(x_masked, y_masked, z_masked, color='skyblue', 
                    alpha=0.3, 
                    # edgecolor='gray', 
                    edgecolor='none',
                    linewidth=0.3)
    
    
    theta = -90*np.pi/180  # azimuthal angle (0 to 2π)
    phi = 150*np.pi/180    # polar angle (0 to π)
    
    # Convert to Cartesian coordinates (unit vector)
    vx = np.sin(phi) * np.cos(theta)
    vy = np.sin(phi) * np.sin(theta)
    vz = np.cos(phi)
    
    ax.quiver(0, 0, 0, vx, vy, vz, color='r', linewidth=0.6, arrow_length_ratio=0.1)
    ax.text(0, -0.02, -0.5, r'S$_0$', fontsize=8, color='r')  # S0
    ax.text(0.45, -0.9, 0, r'$2\psi$', fontsize=8, color='g')  # azimuth
    ax.text(0, -0.4, -0.4, r'$2\tau$', fontsize=8, color='b')    # polar
    
    # Draw azimuth arc in xy-plane
    az_arc = np.linspace(0, theta, 50)
    r = 0.5
    x_arc = r * np.cos(az_arc)
    y_arc = r * np.sin(az_arc)
    z_arc = np.zeros_like(x_arc)
    ax.plot(x_arc, y_arc, z_arc, 'g-', linewidth=0.5)
    ax.plot([x_arc[-2], x_arc[-1]], [y_arc[-2], y_arc[-1]], [z_arc[-2], z_arc[-1]], color='k')
    
    # Draw polar arc in xz-plane aligned with theta
    polar_arc = np.linspace(np.pi/2, phi, 50)
    rp = 0.5
    px = rp * np.sin(polar_arc) * np.cos(theta)
    py = rp * np.sin(polar_arc) * np.sin(theta)
    pz = rp * np.cos(polar_arc)
    ax.plot(px, py, pz, 'b-', linewidth=0.5)
    ax.plot([px[-2], px[-1]], [py[-2], py[-1]], [pz[-2], pz[-1]], color='k') 
    
    for vec, label in zip(stokes_vectors, labels):
        if label:
            ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], color='k', linewidth = 0.7, arrow_length_ratio=0.1)
        else:
            ax.plot([0, vec[0]], [0, vec[1]], [0, vec[2]],  linewidth = 0.7, linestyle=':', color='gray')
        ax.text(vec[0]*1.25, vec[1]*1.1, vec[2]*1.1, label, color='black', fontsize=8)
    
    for vec, label in zip(th_svect, th_labels):
        ax.scatter(*vec* 1.05, color='k', s=5)
        ax.text(*vec * 1.3, label, fontsize=8)
    
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.set_box_aspect([1, 1, 1])
    ax.grid(False)
    plt.axis('off')  
    ax.view_init(elev=-18, azim=-45,roll=-180)
    
    plt.title('Poincaré sphere')
    
    plt.tight_layout()
    if ppath is not None:
        if os.path.isdir(ppath):
            ppath = os.path.join(ppath, 'poincare_sphere.png')  
            
        plt.savefig(ppath,dpi=300,bbox_inches='tight')