import os
import numpy as np
from osgeo import gdal,osr
gdal.UseExceptions()
import matplotlib.pyplot as plt
from polsartools.utils.utils import conv2d,time_it
# from pyproj import CRS
import os

def read_bin(file):
    ds = gdal.Open(file)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    arr[arr==0] = np.nan
    return arr

def norm_data(data,lp=5,gp=95,dB_scale = True):
    
    if dB_scale:
        data = 10*np.log10(data)
    data[data==-np.inf] = np.nan
    data[data==np.inf] = np.nan
    data = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))

    p5 = np.nanpercentile(data, lp)
    p95 = np.nanpercentile(data, gp)
    data = np.clip(data, p5, p95)
    data = (data - p5) / (p95 - p5)

    return data 

def read_and_normalize(file_path):
    """Reads binary data and normalizes it."""
    return norm_data(read_bin(file_path)) if os.path.isfile(file_path) else None


# def create_prj(ref_raster_path, output_png_path):
#     """
#     Creates a .prj file with WKT from the given EPSG code.
#     """
#     ds = gdal.Open(ref_raster_path)
#     wkt = ds.GetProjection()
    
#     # if not wkt:
#     #     raise ValueError("Reference raster does not contain spatial projection information.")
    
#     crs = CRS.from_wkt(wkt)
#     wkt_text = crs.to_wkt()

#     prj_path = os.path.splitext(output_png_path)[0] + ".prj"
#     with open(prj_path, "w") as f:
#         f.write(wkt_text)


def get_alpha_channel(red, green, blue):
    """
    Creates an alpha channel based on the RGB values.
    Transparent where all RGB values are zero or NaN.
    """
    zeros_mask = (red == 0.0) & (green == 0.0) & (blue == 0.0)
    nan_mask = np.isnan(red) & np.isnan(green) & np.isnan(blue)
    transparent_mask = nan_mask
    # transparent_mask = zeros_mask | nan_mask
    return np.where(transparent_mask, 0, 255).astype(np.uint8)

def generate_rgb_png(red, green, blue,georef_file, output_path):
    

    alpha_channel = get_alpha_channel(red, green, blue)
    rgb_uint8 = (np.dstack((red, green, blue)) * 255).astype(np.uint8)
    # alpha_channel = np.where(np.all(rgb_uint8 == 0, axis=2), 0, 255).astype(np.uint8)
    rgba_uint8 = np.dstack((rgb_uint8, alpha_channel))
    
    plt.imsave(output_path, rgba_uint8)
    # print(f"Pauli RGB image saved as {output_path}")
    
    fig, ax = plt.subplots()
    plt.imshow(rgba_uint8, vmin=0, vmax=255)
    ax.axis('off')
    plt.savefig(output_path.replace(".png", "_thumb.png"), format='png', 
                bbox_inches='tight', pad_inches=0, transparent=True)

def generate_rgb_tif(red, green, blue, georef_file, output_path):
    driver = gdal.GetDriverByName('GTiff')
    height, width = red.shape
    dataset = driver.Create(output_path, width, height, 4, gdal.GDT_Byte,options=['COMPRESS=DEFLATE','PREDICTOR=2','ZLEVEL=9', 'BIGTIFF=YES','TILED=YES'])
    # Prepare the image bands
    red_uint8 = (red * 255).astype(np.uint8)
    green_uint8 = (green * 255).astype(np.uint8)
    blue_uint8 = (blue * 255).astype(np.uint8)
    # alpha = np.where((red_uint8 == 0) & (green_uint8 == 0) & (blue_uint8 == 0), 0, 255).astype(np.uint8)
    
    # Get geotransform and projection from reference file
    ref = gdal.Open(georef_file)
    dataset.SetGeoTransform(ref.GetGeoTransform())
    dataset.SetProjection(ref.GetProjection())

    
    alpha = get_alpha_channel(red, green, blue)
    
    

    dataset.GetRasterBand(1).WriteArray(red_uint8)
    dataset.GetRasterBand(2).WriteArray(green_uint8)
    dataset.GetRasterBand(3).WriteArray(blue_uint8)
    dataset.GetRasterBand(4).WriteArray(alpha)

    dataset.FlushCache()
    ref = None
    # print(f"RGB GeoTIFF saved to {output_path}")


def create_pgw(reference_file, output_png_path):
    ds = gdal.Open(reference_file)
    gt = ds.GetGeoTransform()
    ds=None
    pgw_values = [
        gt[1],     # pixel width
        gt[2],     # rotation (typically 0)
        gt[4],     # rotation (typically 0)
        gt[5],     # pixel height (typically negative)
        gt[0] + gt[1] / 2,  # x of center of top-left pixel
        gt[3] + gt[5] / 2   # y of center of top-left pixel
    ]
    
    pgw_path = os.path.splitext(output_png_path)[0] + ".pgw"
    with open(pgw_path, "w") as f:
        for val in pgw_values:
            f.write(f"{val:.10f}\n")
    # print(f"PGW file created at {pgw_path}")\



# def create_pgw(reference_file, output_png_path):
#     ds = gdal.Open(reference_file)
#     gt = ds.GetGeoTransform()
#     proj = ds.GetProjection()
#     ds = None

#     # Prepare source spatial reference
#     source_ref = osr.SpatialReference()
#     source_ref.ImportFromWkt(proj)

#     # Prepare target spatial reference (WGS84)
#     target_ref = osr.SpatialReference()
#     target_ref.ImportFromEPSG(4326)

#     # Create coordinate transformation if needed
#     if not source_ref.IsGeographic() or not source_ref.IsSame(target_ref):
#         transform = osr.CoordinateTransformation(source_ref, target_ref)

#         # Transform center of top-left pixel to WGS84
#         x_geo = gt[0] + gt[1] / 2
#         y_geo = gt[3] + gt[5] / 2
#         x_wgs84, y_wgs84, _ = transform.TransformPoint(x_geo, y_geo)
#     else:
#         x_wgs84 = gt[0] + gt[1] / 2
#         y_wgs84 = gt[3] + gt[5] / 2

#     # Create PGW values (preserve pixel scale and rotation)
#     pgw_values = [
#         gt[1],     # pixel width
#         gt[2],     # rotation
#         gt[4],     # rotation
#         gt[5],     # pixel height
#         x_wgs84,   # transformed X
#         y_wgs84    # transformed Y
#     ]

#     # Write PGW
#     pgw_path = os.path.splitext(output_png_path)[0] + ".pgw"
#     with open(pgw_path, "w") as f:
#         for val in pgw_values:
#             f.write(f"{val:.10f}\n")

    
@time_it
def pauli_rgb(infolder,save_tif=False, window_size=None):
    """
    Generate Pauli RGB image from polarimetric SAR data (C4, C3, T4, T3, or S2 matrices).

    This function creates a Pauli decomposition RGB composite image from full polarimetric SAR data.
    
    Examples
    --------
    >>> # Generate Pauli RGB from a polarimetric folder
    >>> pauli_rgb("/path/to/polSAR_data")

    >>> # Output both PNG and GeoTIFF versions
    >>> pauli_rgb("/path/to/polSAR_data", tif_flag=True)

    Parameters
    ----------
    infolder : str
        Path to the folder containing polarimetric matrix files (.bin or .tif). 
        Supports full- and dual-pol data in formats: C4, C3, T4, T3, or S2.
    save_tif : bool, default=False
        If True, generates both PNG and GeoTIFF (.tif) versions of the Pauli RGB image.
    window_size : int, optional
        Size of the moving average window for smoothing. If None, no smoothing is applied.
    Returns
    -------
    None
    
    Writes output to:
        
        - PauliRGB.png: RGB composite with world file for georeferencing
        - PauliRGB.tif: Optional GeoTIFF output if `tif_flag=True`

    """
    def find_file(infolder, base_name):
        for ext in [".bin", ".tif"]:
            path = os.path.join(infolder, f"{base_name}{ext}")
            if os.path.isfile(path):
                return path
        return None
    c4_keys = ["C11", "C22", "C33", "C44", "C14_real"]
    c4_files = [find_file(infolder, key) for key in c4_keys]

    if all(c4_files):
        C11, C22, C33, C44 = [read_bin(f) for f in c4_files[:4]]
        C14_real = read_bin(c4_files[4])
        red = norm_data(C11 + C44 - 2 * C14_real)
        green = norm_data(C22 + C33)
        blue = norm_data(C11 + C44 + 2 * C14_real)
        georef_file = c4_files[0]

    # C3 fallback
    elif all(find_file(infolder, f"C{i}") for i in [11, 22, 33]):
        C11 = read_bin(find_file(infolder, "C11"))
        C22 = read_bin(find_file(infolder, "C22"))
        C33 = read_bin(find_file(infolder, "C33"))
        C13_real = read_bin(find_file(infolder, "C13_real"))
        blue = norm_data(0.5 * (C11 + C33 + 2 * C13_real))
        red = norm_data(0.5 * (C11 + C33 - 2 * C13_real))
        green = norm_data(C22)
        georef_file = find_file(infolder, "C11")

    # T3 fallback
    elif all(find_file(infolder, f"T{i}") for i in [11, 22, 33]):
        red = read_and_normalize(find_file(infolder, "T22"))
        green = read_and_normalize(find_file(infolder, "T33"))
        blue = read_and_normalize(find_file(infolder, "T11"))
        georef_file = find_file(infolder, "T11")

    # S2 fallback
    elif all(find_file(infolder, f"s{i}") for i in [11, 12, 22]):
        S11 = read_bin(find_file(infolder, "s11"))
        S12 = read_bin(find_file(infolder, "s12"))
        S22 = read_bin(find_file(infolder, "s22"))
        blue = norm_data(np.abs((1 / np.sqrt(2)) * (S11 + S22))**2)
        red = norm_data(np.abs((1 / np.sqrt(2)) * (S11 - S22))**2)
        green = norm_data(np.abs((2 / np.sqrt(2)) * S12)**2)
        georef_file = find_file(infolder, "s11")

    else:
        raise ValueError("No matching dataset found for C4, C3, T3, or S2!")
    output_path = os.path.join(infolder, "PauliRGB.png")
    
    if window_size is not None:
        kernel = np.ones((window_size, window_size), np.float32) / (window_size * window_size)
        red = conv2d(red, kernel).astype(np.float32)
        green = conv2d(green, kernel).astype(np.float32)
        blue = conv2d(blue, kernel).astype(np.float32)
    
    generate_rgb_png(red, green, blue,georef_file, output_path)
    create_pgw(georef_file, output_path)
    # create_prj(georef_file, output_path)
    print(f"Pauli RGB image saved as {output_path}")
    if save_tif:
        output_path = os.path.join(infolder, "PauliRGB.tif")
        generate_rgb_tif(red, green, blue, georef_file, output_path)
        print(f"Pauli RGB image saved as {output_path}")
