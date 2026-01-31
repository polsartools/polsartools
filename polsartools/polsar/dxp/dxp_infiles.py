import os
import numpy as np
def find_file(infolder, base_name):
    for ext in [".bin", ".tif"]:
        path = os.path.join(infolder, f"{base_name}{ext}")
        if os.path.isfile(path):
            return path
    return None

def dxpc2files(infolder):
    """
    Returns file paths for C2 matrix elements, supporting .bin and .tif formats.
    """
    required_keys = ["C11", "C12_real", "C12_imag", "C22"]
    input_filepaths = [find_file(infolder, key) for key in required_keys]

    if all(input_filepaths):
        return input_filepaths
    else:
        print("Invalid C2 folder: missing required files or unsupported formats.")
        return None

def S_norm(S_array):
    S_5 = np.nanpercentile(S_array, 2)
    S_95 = np.nanpercentile(S_array, 98)
    S_cln = np.where(S_array > S_95, S_95, S_array)
    S_cln = np.where(S_cln < S_5, S_5, S_cln)
    S_cln_max = np.nanmax(S_cln)
    S_norm_array = np.divide(S_cln,S_cln_max) 
    
    return S_norm_array