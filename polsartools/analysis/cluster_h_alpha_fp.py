import os
import numpy as np
from polsartools.utils.proc_utils import process_chunks_parallel
from polsartools.utils.utils import conv2d,time_it,read_rst

@time_it
def cluster_h_alpha_fp(hFile,alphaFile , window_size=1, outType="tif", cog_flag=False, 
          cog_overviews = [2, 4, 8, 16], write_flag=True, 
          max_workers=None,block_size=(512, 512),
          progress_callback=None,  # for QGIS plugin
            ):
    """
    Perform H-alpha clustering on given H-alpha files for full-pol SAR data.

    Examples
    --------
    >>> # Basic usage with default parameters
    >>> cluster_h_alpha_fp("h_fp.tif", "alpha_fp.tif")
    
    >>> # Advanced usage with custom parameters
    >>> halpha_cluster_fp(
    ...     hFile="h_fp.tif",
    ...     alphaFile="alpha_fp.tif",
    ...     window_size=5,
    ...     outType="tif",
    ...     cog_flag=True,
    ...     block_size=(1024, 1024)
    ... )


    Parameters
    ----------
    hFile : str
        Path to the input Entropy file, H  (polarimetric entropy).
    alphaFile : str
        Path to the input alpha file (average scattering angle).
    window_size : int, default=1
        Size of the spatial averaging window. Larger windows reduce speckle noise
        but decrease spatial resolution.
    outType : {'tif', 'bin'}, default='tif'
        Output file format:
        - 'tif': GeoTIFF format with georeferencing information
        - 'bin': Raw binary format
    cog_flag : bool, default=False
        If True, creates a Cloud Optimized GeoTIFF (COG) with internal tiling
        and overviews for efficient web access.
    cog_overviews : list[int], default=[2, 4, 8, 16]
        Overview levels for COG creation. Each number represents the
        decimation factor for that overview level.
    write_flag : bool, default=True
        If True, writes results to disk. If False, only processes data in memory.
    max_workers : int | None, default=None
        Maximum number of parallel processing workers. If None, uses
        CPU count - 1 workers.
    block_size : tuple[int, int], default=(512, 512)
        Size of processing blocks (rows, cols) for parallel computation.
        Larger blocks use more memory but may be more efficient.

    Returns
    -------
    None
        Writes one output file to disk:
        - 'ha_cluster.tif' or 'ha_cluster.bin'
        - 'ha_cluster.png'

    """
    input_filepaths = [hFile,alphaFile]
    infolder = os.path.dirname(hFile)
    output_filepaths = []
    if outType == "bin":
        output_filepaths.append(os.path.join(infolder, "ha_cluster.bin"))
    else:
        output_filepaths.append(os.path.join(infolder, "ha_cluster.tif"))
    
    
    
    def post_processing_task(*args, **kwargs):
        zones = read_rst(output_filepaths[0]) 
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        # Define colors for Z1–Z9 (index 0 is unused)
        colors = [
            '#000000',     # Index 0 (unused)
            '#00FF00',     # Z1 - bright green (strong volume)
            '#ADFF2F',     # Z2 - yellow-green (moderate volume)
            '#FF00FF',     # Z3 - megenta (low volume/helicity)
            '#228B22',     # Z4 - forest green (dense volume)
            '#FFD700',     # Z5 - gold (mixed vegetation-ground)
            '#4682B4',     # Z6 - steel blue (rough surface)
            '#FF0000',     # Z7 - red (strong double-bounce)
            '#D2691E',     # Z8 - tan/brown (mixed corner-ground)
            '#0000ff'      # Z9 - light gray (flat surface / Bragg)
        ]

        # Create custom colormap
        zone_cmap = ListedColormap(colors[1:])  # Skip index 0

        # Display zone map
        plt.figure()
        im = plt.imshow(zones, cmap=zone_cmap, vmin=1, vmax=10) 
        cbar = plt.colorbar(im, ticks=np.arange(1.5, 10, 1))  
        cbar.ax.set_yticklabels([f'Z{i}' for i in range(1, 10)])
        plt.title(r"H-$\overline{\alpha}$ clusters")
        plt.axis('off') 
        plt.tight_layout()
        plt.savefig(os.path.join(infolder, "ha_cluster.png"), bbox_inches='tight',dpi=300)
        plt.show()
        
        
        
        
    process_chunks_parallel(input_filepaths, list(output_filepaths), 
                            window_size=window_size, write_flag=write_flag,
                        processing_func=process_chunk_hacfp,block_size=block_size, 
                        max_workers=max_workers,  num_outputs=len(output_filepaths),
                        cog_flag=cog_flag,
                        cog_overviews=cog_overviews,post_proc=post_processing_task,
                        progress_callback=progress_callback
                        
                        )

def process_chunk_hacfp(chunks, window_size,input_filepaths,*args):

    h = np.array(chunks[0])
    alpha = np.array(chunks[1])


    if window_size>1:
        kernel = np.ones((window_size,window_size),np.float32)/(window_size*window_size)

        h = conv2d(h,kernel)
        alpha = conv2d(alpha,kernel)
        
    zones = np.zeros_like(h, dtype=int)

    # Z1–Z3: h ∈ [0.9, 1.0]
    zones[(h >= 0.9) & (alpha >= 60)] = 1  # Z1
    zones[(h >= 0.9) & (alpha >= 40) & (alpha < 60)] = 2  # Z2
    zones[(h >= 0.9) & (alpha < 40)] = 3  # Z3

    # Z4–Z6: h ∈ [0.5, 0.9)
    zones[(h >= 0.5) & (h < 0.9) & (alpha >= 50)] = 4  # Z4
    zones[(h >= 0.5) & (h < 0.9) & (alpha >= 40) & (alpha < 50)] = 5  # Z5
    zones[(h >= 0.5) & (h < 0.9) & (alpha < 40)] = 6  # Z6

    # Z7–Z9: h < 0.5
    zones[(h < 0.5) & (alpha >= 47.5)] = 7  # Z7
    zones[(h < 0.5) & (alpha >= 42.5) & (alpha < 47.5)] = 8  # Z8
    zones[(h < 0.5) & (alpha < 42.5)] = 9  # Z9

    return zones.astype(np.uint8)

