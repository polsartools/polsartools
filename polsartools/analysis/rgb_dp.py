import os
import numpy as np
from polsartools.utils.utils import conv2d,time_it
from osgeo import gdal
gdal.UseExceptions()
import matplotlib.pyplot as plt

from .pauli_rgb import generate_rgb_png, create_pgw, generate_rgb_tif, norm_data, read_bin #create_prj

@time_it
def rgb_dp(infolder, type=1, save_tif=False, window_size=None):
    
    """
    Generate false-color RGB visualization from dual-polarimetric Sxy, C2 SAR data.

    This function creates an RGB image from dual-pol (Sxy, C2)  matrix inputs.

    Examples
    --------
    >>> # Generate type-1 false-color RGB from C2 folder
    >>> rgb_dp("/path/to/data")

    >>> # Output both PNG and GeoTIFF versions
    >>> rgb_dp("/path/to/data", type=3, tif_flag=True)

    Parameters
    ----------
    infolder : str
        Path to the input folder containing C2 matrix components (C11, C22, C12_real).
    type : int, {1, 2, 3, 4}, default=1
        RGB combination type:
        - 1: Blue=C11, Red=C22, Green=|C11 + C22 - 2·C12_real|
        - 2: Blue=C22, Red=C11, Green=|C11 + C22 - 2·C12_real|
        - 3: Blue=C11, Red=C22, Green=|C11 - C22|
        - 4: Blue=C22, Red=C11, Green=|C22 - C11|
        - 5: Blue=C11, Red=C11, Green=C22
    save_tif : bool, default=False
        If True, generates a GeoTIFF (.tif) file alongside the PNG image.

    Returns
    -------
    None
        Writes RGB visualization to disk:
        - 'RGB#.png': False-color PNG image (with world file for georeferencing)
        - 'RGB#.tif': Optional GeoTIFF output if `tif_flag=True`

    """
    def find_file(base):
        for ext in [".bin", ".tif"]:
            path = os.path.join(infolder, f"{base}{ext}")
            if os.path.isfile(path):
                return path
        return None

    c11_path = find_file("C11")
    c22_path = find_file("C22")
    c12r_path = find_file("C12_real")

    if c11_path and c22_path:
        pass
    else:
        # Try all valid input combos
        combinations = [
            ("s11", "s12"),
            ("s11", "s21"),
            ("s22", "s12"),
            ("s22", "s21"),
            
        ]
        for sA, sB in combinations:
            pathA = find_file(sA)
            pathB = find_file(sB)





    def load_cov(name):
        path = find_file(name)
        return read_bin(path) if path else None

    def construct_from_sxy():
        for sA, sB in [("s11","s12"),("s11","s21"),("s22","s12"),("s22","s21")]:
            pathA, pathB = find_file(sA), find_file(sB)
            if pathA and pathB:
                S1 = read_bin(pathA).astype(np.complex64)
                S2 = read_bin(pathB).astype(np.complex64)
                return S1, S2
        raise FileNotFoundError("No valid C2 or Sxy data found.")

    if type == 1:
        C22 = load_cov("C22")
        if C22 is None:
            _, S2 = construct_from_sxy()
            C22 = (np.abs(S2)**2).astype(np.float32)
        red = norm_data(C22)
        del C22

        C11 = load_cov("C11")
        if C11 is None:
            S1, _ = construct_from_sxy()
            C11 = (np.abs(S1)**2).astype(np.float32)
        blue = norm_data(C11)

        C12r = load_cov("C12_real")
        if C12r is None:
            S1, S2 = construct_from_sxy()
            C12r = (S1*np.conj(S2)).real.astype(np.float32)
        green = norm_data(np.abs(C11 + blue*0 + red*0 + C22 - 2*C12r))
        del C11, C12r

    elif type == 2:
        C11 = load_cov("C11")
        if C11 is None:
            S1, _ = construct_from_sxy()
            C11 = (np.abs(S1)**2).astype(np.float32)
        red = norm_data(C11)

        C22 = load_cov("C22")
        if C22 is None:
            _, S2 = construct_from_sxy()
            C22 = (np.abs(S2)**2).astype(np.float32)
        blue = norm_data(C22)

        C12r = load_cov("C12_real")
        if C12r is None:
            S1, S2 = construct_from_sxy()
            C12r = (S1*np.conj(S2)).real.astype(np.float32)
        green = norm_data(np.abs(C11 + C22 - 2*C12r))
        del C11, C22, C12r

    elif type == 3:
        C22 = load_cov("C22")
        if C22 is None:
            _, S2 = construct_from_sxy()
            C22 = (np.abs(S2)**2).astype(np.float32)
        red = norm_data(C22)

        C11 = load_cov("C11")
        if C11 is None:
            S1, _ = construct_from_sxy()
            C11 = (np.abs(S1)**2).astype(np.float32)
        blue = norm_data(C11)

        green = norm_data(np.abs(C11 - C22))
        del C11, C22

    elif type == 4:
        C11 = load_cov("C11")
        if C11 is None:
            S1, _ = construct_from_sxy()
            C11 = (np.abs(S1)**2).astype(np.float32)
        red = norm_data(C11)

        C22 = load_cov("C22")
        if C22 is None:
            _, S2 = construct_from_sxy()
            C22 = (np.abs(S2)**2).astype(np.float32)
        blue = norm_data(C22)

        green = norm_data(np.abs(C22 - C11))
        del C11, C22

    elif type == 5:
        C11 = load_cov("C11")
        if C11 is None:
            S1, _ = construct_from_sxy()
            C11 = (np.abs(S1)**2).astype(np.float32)
        red = norm_data(C11)
        blue = red
        del C11

        C22 = load_cov("C22")
        if C22 is None:
            _, S2 = construct_from_sxy()
            C22 = (np.abs(S2)**2).astype(np.float32)
        green = norm_data(C22)
        del C22

    else:
        raise ValueError("Invalid type!! Valid types are 1,2,3,4,5")

        
    if c11_path:
        georef_file = c11_path
    else:
        georef_file = pathA  # Use the first file for georeferencing
    
    
    if window_size is not None:
        kernel = np.ones((window_size, window_size), np.uint8) / (window_size * window_size)
        red = conv2d(red, kernel).astype(np.uint8)
        green = conv2d(green, kernel).astype(np.uint8)
        blue = conv2d(blue, kernel).astype(np.uint8)

    if save_tif:
        tif_path = os.path.join(infolder, f'RGB{type}.tif')
        generate_rgb_tif(red, green, blue, georef_file, tif_path)
        print(f"RGB GeoTIFF saved as {tif_path}")
        
    output_path = os.path.join(infolder, f'RGB{type}.png')
    generate_rgb_png(red, green, blue, georef_file, output_path)
    create_pgw(georef_file, output_path)
    
    print(f"RGB image saved as {output_path}")
    


@time_it
def rgb_dp_old(infolder, type=1, save_tif=False, window_size=None):
    
    """
    Generate false-color RGB visualization from dual-polarimetric Sxy, C2 SAR data.

    This function creates an RGB image from dual-pol (Sxy, C2)  matrix inputs.

    Examples
    --------
    >>> # Generate type-1 false-color RGB from C2 folder
    >>> rgb_dp("/path/to/data")

    >>> # Output both PNG and GeoTIFF versions
    >>> rgb_dp("/path/to/data", type=3, tif_flag=True)

    Parameters
    ----------
    infolder : str
        Path to the input folder containing C2 matrix components (C11, C22, C12_real).
    type : int, {1, 2, 3, 4}, default=1
        RGB combination type:
        - 1: Blue=C11, Red=C22, Green=|C11 + C22 - 2·C12_real|
        - 2: Blue=C22, Red=C11, Green=|C11 + C22 - 2·C12_real|
        - 3: Blue=C11, Red=C22, Green=|C11 - C22|
        - 4: Blue=C22, Red=C11, Green=|C22 - C11|
        - 5: Blue=C11, Red=C11, Green=C22
    save_tif : bool, default=False
        If True, generates a GeoTIFF (.tif) file alongside the PNG image.

    Returns
    -------
    None
        Writes RGB visualization to disk:
        - 'RGB#.png': False-color PNG image (with world file for georeferencing)
        - 'RGB#.tif': Optional GeoTIFF output if `tif_flag=True`

    """
    def find_file(base):
        for ext in [".bin", ".tif"]:
            path = os.path.join(infolder, f"{base}{ext}")
            if os.path.isfile(path):
                return path
        return None

    c11_path = find_file("C11")
    c22_path = find_file("C22")
    c12r_path = find_file("C12_real")

    if c11_path and c22_path:
        C11 = read_bin(c11_path)
        C22 = read_bin(c22_path)
        C12 = read_bin(c12r_path) if c12r_path else None
    else:
        # Try all valid input combos
        combinations = [
            ("s11", "s12"),
            ("s11", "s21"),
            ("s22", "s12"),
            ("s22", "s21"),
            
        ]

        for sA, sB in combinations:
            pathA = find_file(sA)
            pathB = find_file(sB)
            if pathA and pathB:
                S1 = read_bin(pathA).astype(np.complex64)
                S2 = read_bin(pathB).astype(np.complex64)

                # Construct C11, C22, C12
                C11 = (np.abs(S1) ** 2).astype(np.float32)
                C22 = (np.abs(S2) ** 2).astype(np.float32)
                C12 = (S1 * np.conj(S2)).astype(np.complex64)
                break
        else:
            raise FileNotFoundError("No valid C2 or Sxy data found. Expected one of:  (s11,s12), (s11,s21), (s22,s12), (s22,s21)  (C11,C22,C12_real)")

    # Compute RGB channels
    if type == 1:
        red = norm_data(C22)
        blue = norm_data(C11)
        green = norm_data(np.abs(C11 + C22 - 2 * C12.real))
    elif type == 2:
        red = norm_data(C11)
        blue = norm_data(C22)
        green = norm_data(np.abs(C11 + C22 - 2 * C12.real))
    elif type == 3:
        red = norm_data(C22)
        blue = norm_data(C11)
        green = norm_data(np.abs(C11 - C22))
    elif type == 4:
        red = norm_data(C11)
        blue = norm_data(C22)
        green = norm_data(np.abs(C22 - C11))
    elif type == 5:
        red = norm_data(C11)
        blue = red
        green = norm_data(C22)

    else:
        raise ValueError("Invalid type!! Valid types are 1,2,3,4,5")


    if c11_path:
        georef_file = c11_path
    else:
        georef_file = pathA  # Use the first file for georeferencing
    output_path = os.path.join(infolder, f'RGB{type}.png')
    
    if window_size is not None:
        kernel = np.ones((window_size, window_size), np.uint8) / (window_size * window_size)
        red = conv2d(red, kernel).astype(np.uint8)
        green = conv2d(green, kernel).astype(np.uint8)
        blue = conv2d(blue, kernel).astype(np.uint8)

    if save_tif:
        tif_path = os.path.join(infolder, f'RGB{type}.tif')
        generate_rgb_tif(red, green, blue, georef_file, tif_path)
        print(f"RGB GeoTIFF saved as {tif_path}")

    generate_rgb_png(red, green, blue, georef_file, output_path)
    create_pgw(georef_file, output_path)
    
    print(f"RGB image saved as {output_path}")
    
