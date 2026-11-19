import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import ListedColormap, BoundaryNorm
from polsartools.utils.utils import read_bin, time_it

@time_it
def plot_hc_thetac_dp(Hc, thetac, ppath=None, cmap='viridis', colorbar=True, norm='', grey_region=True, gridsize=300, polar=False):


    if isinstance(Hc, (str, os.PathLike, Path)):
        Hc = read_bin(Hc)
    if isinstance(thetac, (str, os.PathLike, Path)):
        thetac = read_bin(thetac)

    Hc = np.asarray(Hc).flatten()
    thetac = np.asarray(thetac).flatten()

    valid_mask = (
        ~np.isnan(Hc) & ~np.isnan(thetac) & 
        np.isfinite(Hc) & np.isfinite(thetac) & 
        ~((Hc == 0) & (thetac == 0))
    )

    Hc = Hc[valid_mask]
    thetac = thetac[valid_mask]
    del valid_mask  


    phi_val = 67.5 + thetac
    theta_rad = np.radians(phi_val)

    # Define conditions
    conditions = [
        (thetac > 30) & (Hc < 0.3),                               # Z1 (index 0)
        (thetac > 30) & (Hc >= 0.3) & (Hc < 0.5),                 # Z2 (index 1)
        (thetac > 30) & (Hc >= 0.5) & (Hc < 0.7),                 # Z3 (index 2)
        (thetac > 30) & (Hc >= 0.7),                              # Z4 (index 3)
        (thetac <= 30) & (Hc >= 0.7) & (thetac > 15),             # Z5 (index 4)
        (thetac <= 30) & (Hc >= 0.7) & (thetac <= 15),            # Z6 (index 5)
        (thetac <= 30) & (Hc < 0.7)                               # Gray Region (index 6)
    ]

    choices_color = ['deeppink', '#7b1f50', 'limegreen', '#f0c27b', 'cyan', 'forestgreen', 'silver']
    choices_zone = ['Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6', 'Gray Region']
    choices_idx = [0, 1, 2, 3, 4, 5, 6]

    # Use integer array instead of strings for blazing fast vectorized assignment
    zone_indices = np.select(conditions, choices_idx, default=6)

    # Setup figure
    dpi = 300
    if polar:
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 5), dpi=dpi)
        ax.grid(False) 
        ax.set_thetamin(67.5)
        ax.set_thetamax(112.5)
        ax.set_rlim(0, 1.0)

        if grey_region:
            theta_gray = np.radians(np.linspace(67.5, 97.5, 100))
            ax.fill_between(theta_gray, 0, 0.7, color='silver', zorder=1)

        # Plot points grouped by integer zone indices (much faster than string comparisons)
        for idx, col in enumerate(choices_color):
            mask = (zone_indices == idx)
            if not np.any(mask):
                continue
            edge_c = 'none' if col not in ['#f0c27b', 'cyan'] else 'None'
            
            ax.scatter(
                theta_rad[mask], 
                Hc[mask], 
                color=col, 
                s=120, 
                zorder=3, 
                edgecolor=edge_c, 
                alpha=0.9
            )

        ax.set_rgrids([0, 0.3, 0.5, 0.7, 1.0], labels=['', '', '', '', ''])

        r_fs = 16
        ax.text(np.radians(182), 0.03, '0', fontsize=r_fs, ha='center', va='center')
        ax.text(np.radians(127), 0.33, '0.3', fontsize=r_fs, ha='left', va='center')
        ax.text(np.radians(123), 0.53, '0.5', fontsize=r_fs, ha='left', va='center')
        ax.text(np.radians(121), 0.72, '0.7', fontsize=r_fs, ha='left', va='center')
        ax.text(np.radians(118), 1.02, '1.0', fontsize=r_fs, ha='left', va='center')

        ax.text(np.radians(128.0), 0.53, '$H_c$', fontsize=18, ha='center', va='center')
        ax.text(np.radians(90.0), 1.05, r'$\theta_c$', fontsize=18, ha='center', va='bottom')

        tick_angles = [67.5, 82.5, 97.5, 112.5]
        ax.set_thetagrids(tick_angles, labels=['0°', '15°', '30°', '45°'], fontsize=15)

        grid_color = 'black'
        grid_ls = '--'
        grid_lw = 1
        grid_alpha = 0.8

        for r_val in [0.3, 0.5]:
            th = np.radians(np.linspace(97.5, 112.5, 50))
            ax.plot(th, [r_val] * len(th), linestyle=grid_ls, color=grid_color, linewidth=grid_lw, alpha=grid_alpha, zorder=2)

        for r_val in [0.7, 1.0]:
            th = np.radians(np.linspace(67.5, 112.5, 100))
            ax.plot(th, [r_val] * len(th), linestyle=grid_ls, color=grid_color, linewidth=grid_lw, alpha=grid_alpha, zorder=2)

        for angle in [67.5, 82.5]:
            r_vals = np.linspace(0.7, 1.0, 50)
            ax.plot([np.radians(angle)] * len(r_vals), r_vals, linestyle=grid_ls, color=grid_color, linewidth=grid_lw, alpha=grid_alpha, zorder=2)

        for angle in [97.5, 112.5]:
            r_vals = np.linspace(0, 1.0, 100)
            ax.plot([np.radians(angle)] * len(r_vals), r_vals, linestyle=grid_ls, color=grid_color, linewidth=grid_lw, alpha=grid_alpha, zorder=2)

        ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, top=True, bottom=True, left=True, right=True)
    else:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=dpi)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 45)
        
        if grey_region:
            ax.fill_between([0, 0.7], [0, 0], [30, 30], color='silver', zorder=0)

        for idx, col in enumerate(choices_color):
            mask = (zone_indices == idx)
            if not np.any(mask):
                continue
            edge_c = 'none' if col not in ['#f0c27b', 'cyan'] else 'None'
            
            ax.scatter(
                Hc[mask], 
                thetac[mask], 
                color=col, 
                s=120, 
                zorder=2, 
                edgecolor=edge_c, 
                alpha=0.9
            )

        ax.set_xticks([0, 0.3, 0.5, 0.7, 1.0])
        ax.set_yticks([0, 15, 30, 45])
        ax.set_yticklabels(['0°', '15°', '30°', '45°'], fontsize=18)
        ax.set_xticklabels(['0', '0.3', '0.5', '0.7', '1.0'], fontsize=18)

        grid_kw = {'linestyle': '--', 'color': 'black', 'linewidth': 1, 'alpha': 0.8, 'zorder': 1}
        ax.vlines(x=[0, 0.3, 0.5], ymin=30, ymax=45, **grid_kw) 
        ax.vlines(x=[0.7], ymin=0, ymax=45, **grid_kw)           
        ax.vlines(x=[1.0], ymin=0, ymax=45, **grid_kw)          
        ax.hlines(y=[0, 15], xmin=0.7, xmax=1.0, **grid_kw)    
        ax.hlines(y=[30, 45], xmin=0, xmax=1.0, **grid_kw)    

        ax.set_xlabel(r'$H_c$', fontsize=22)
        ax.set_ylabel(r'$\theta_c$', fontsize=22)

        ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, 
                    top=True, bottom=True, left=True, right=True)

        for spine in ax.spines.values():
            spine.set_linewidth(1)
            spine.set_color('black')

    # Colorbar configuration
    zone_colors = ['deeppink', '#7b1f50', 'limegreen', '#f0c27b', 'cyan', 'forestgreen']
    cmap = ListedColormap(zone_colors)
    norm = BoundaryNorm(np.arange(7) + 0.5, cmap.N)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    if colorbar:
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical', pad=0.08 if polar else 0.03, aspect=16, ticks=range(1, 7))
        cbar.ax.set_yticklabels(['Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6'], fontsize=14)
        cbar.ax.invert_yaxis()  
        cbar.ax.tick_params(size=0) 
        cbar.outline.set_linewidth(1.5)

    plt.tight_layout()
    if ppath is not None:
        if os.path.isdir(ppath):
            filename = os.path.join(ppath, 'hc_thetac_plot_dp.png')
        else:
            filename = ppath
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved successfully as '{filename}'")
        # plt.close(fig)