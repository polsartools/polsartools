
import os
import numpy as np
from polsartools.utils.utils import conv2d,time_it
from osgeo import gdal
gdal.UseExceptions()

from .pauli_rgb import generate_rgb_png, create_pgw, generate_rgb_tif, norm_data, read_bin #create_prj

def ensure_array(x):
    # If it's a path-like object (string or Path), read from file
    if isinstance(x, (str, os.PathLike)):
        return np.array(read_bin(x), dtype=np.float32)
    # If it's already a NumPy array, just cast dtype
    elif isinstance(x, np.ndarray):
        return x.astype(np.float32)
    # If it's a list or other sequence, convert to array
    else:
        return np.array(x, dtype=np.float32)



@time_it
def rgb(Rpath,Gpath,Bpath,
        norm_each=True,lp=5,gp=95,
        norm_span = False,
        save_tif=False,
        window_size=None):
    
    """Generate false-color RGB visualization from three input raster files.

    Examples
    --------
    >>> # Generate Pauli RGB from a polarimetric folder
    >>> rgb("/path/to/red.tif", "/path/to/green.tif", "/path/to/blue.tif")

    >>> # Output both PNG and GeoTIFF versions
    >>> rgb("/path/to/red.tif", "/path/to/green.tif", "/path/to/blue.tif", tif_flag=True)

    Parameters
    ----------
    Rpath : str
        Path to the red channel raster file.
    Gpath : str
        Path to the green channel raster file.
    Bpath : str
        Path to the blue channel raster file.
    norm_each : bool, default=True
        If True, normalize each channel separately.
    lp : int, default=5
        Lower percentile for normalization (0-100). Only used if `norm_each=True`.
    gp : int, default=95
        Upper percentile for normalization(0-100). Only used if `norm_each=True`.
    norm_span : bool, default=False
        If True, normalize each channel with total power.
    save_tif : bool, default=False
        If True, generates a GeoTIFF (.tif) file alongside the PNG image.
    window_size : int, optional
        Size of the moving average window for smoothing. If None, no smoothing is applied.
        
    Returns 
    -------
    None
    Writes output to:
    
    - RGB.png: RGB composite with world file for georeferencing
    - RGB.tif: Optional GeoTIFF output if `tif_flag=True`
    
    """ 
    
    # R = read_bin(Rpath)
    # G = read_bin(Gpath)
    # B = read_bin(Bpath)

    R = ensure_array(Rpath)
    G = ensure_array(Gpath)
    B = ensure_array(Bpath)

    # R = np.array(read_bin(Rpath), dtype=np.float32)
    # G = np.array(read_bin(Gpath), dtype=np.float32)
    # B = np.array(read_bin(Bpath), dtype=np.float32)

    if window_size is not None:
        kernel = np.ones((window_size, window_size), np.float32) / (window_size * window_size)
        R = conv2d(R, kernel).astype(np.float32)
        G = conv2d(G, kernel).astype(np.float32)
        B = conv2d(B, kernel).astype(np.float32)
        
    if norm_each:
        R = norm_data(R, lp, gp, dB_scale=True)
        G = norm_data(G, lp, gp, dB_scale=True)
        B = norm_data(B, lp, gp, dB_scale=True)
        
    if norm_span:
        total = R + G + B
        red = ((R / total)*255).astype(np.uint8)
        green = ((G / total)*255).astype(np.uint8)
        blue = ((B / total)*255).astype(np.uint8)
        
        del R,G,B,total
    else:
        red = R
        green = G
        blue = B
    
    if isinstance(Rpath, (str, os.PathLike)):
        georef_file = Rpath
        infolder = os.path.dirname(georef_file)
        output_path = os.path.join(infolder, 'RGB.png')
        generate_rgb_png(red, green, blue, georef_file, output_path)
        create_pgw(georef_file, output_path)
        
        print(f"RGB image saved as {output_path}")
        
        if save_tif:
            tif_path = os.path.join(infolder, f'RGB.tif')
            generate_rgb_tif(red, green, blue, georef_file, tif_path)
            print(f"RGB GeoTIFF saved as {tif_path}")
    elif isinstance(Rpath, np.ndarray):
        georef_file = None
        infolder = './'
        output_path = os.path.join(infolder, 'RGB.png')
        generate_rgb_png(red, green, blue, georef_file, output_path)
        # create_pgw(georef_file, output_path)
        
        print(f"RGB image saved as {output_path}")
    else:
        raise('Invalid input!!')

    
    


    # plt.imshow(RGB_image)
    # plt.axis('off')  # Hide axes
    # if pname:
    #     plt.savefig(pname, bbox_inches='tight', pad_inches=0,dpi=300)
    # plt.show()