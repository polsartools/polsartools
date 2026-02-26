import os

def find_file(infolder, base_name):
    for ext in [".bin", ".tif"]:
        path = os.path.join(infolder, f"{base_name}{ext}")
        if os.path.isfile(path):
            return path
    return None

def fp_c3t3files(infolder):
    """
    Returns a list of paths for T3 or C3 matrix components, supporting .bin and .tif.
    T3 is prioritized if both are present.
    """
    keys_t3 = ["T11", "T12_real", "T12_imag", "T13_real", "T13_imag",
               "T22", "T23_real", "T23_imag", "T33"]
    
    keys_c3 = ["C11", "C12_real", "C12_imag", "C13_real", "C13_imag",
               "C22", "C23_real", "C23_imag", "C33"]

    filepaths = []
    if find_file(infolder, "T11"):
        filepaths = [find_file(infolder, key) for key in keys_t3]
    elif find_file(infolder, "C11"):
        filepaths = [find_file(infolder, key) for key in keys_c3]
    else:
        print("Invalid C3 or T3 folder!!")
        return None

    if all(filepaths):
        return filepaths
    else:
        print("Some required files are missing or in unsupported formats.")
        return None