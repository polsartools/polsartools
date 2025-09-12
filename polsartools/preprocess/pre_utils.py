import os


def find_file(infolder, basename):
    for ext in ['.bin', '.tif']:
        path = os.path.join(infolder, basename + ext)
        if os.path.isfile(path):
            return path
    return None

def collect_input_files(infolder, basenames):
    files = []
    for name in basenames:
        path = find_file(infolder, name)
        if not path:
            raise FileNotFoundError(f"Missing file for {name} (.bin or .tif)")
        files.append(path)
    return files



def get_filter_io_paths(infolder, window_size, fmt="tif", filter_type=None,sub_dir=True):
    input_filepaths = []
    matrix_type = None

    def has_all(basenames):
        return all(find_file(infolder, name) for name in basenames)

    # Detection order matters — more specific first
    if has_all(["T11", "T12_real", "T12_imag", "T13_real", "T13_imag", "T22", "T23_real", "T23_imag", "T33", "T14_real", "T14_imag", "T24_real", "T24_imag", "T34_real", "T34_imag", "T44"]):
        matrix_type = "T4"
        base_names = [
            "T11", "T12_real", "T12_imag", "T13_real", "T13_imag", "T14_real", "T14_imag",
            "T22", "T23_real", "T23_imag", "T24_real", "T24_imag",
            "T33", "T34_real", "T34_imag", "T44"
        ]
    elif has_all(["C11", "C12_real", "C12_imag", "C13_real", "C13_imag", "C14_real", "C14_imag", "C22", "C23_real", "C23_imag", "C24_real", "C24_imag", "C33", "C34_real", "C34_imag", "C44"]):
        matrix_type = "C4"
        base_names = [
            "C11", "C12_real", "C12_imag", "C13_real", "C13_imag", "C14_real", "C14_imag",
            "C22", "C23_real", "C23_imag", "C24_real", "C24_imag",
            "C33", "C34_real", "C34_imag", "C44"
        ]
    elif has_all(["s11", "s12", "s21", "s22"]):
        matrix_type = "s2"
        base_names = ["s11", "s12", "s21", "s22"]
    elif has_all(["T11", "T33"]):
        matrix_type = "T3"
        base_names = ["T11", "T12_real", "T12_imag", "T13_real", "T13_imag", "T22", "T23_real", "T23_imag", "T33"]
    elif has_all(["C11", "C33"]):
        matrix_type = "C3"
        base_names = ["C11", "C12_real", "C12_imag", "C13_real", "C13_imag", "C22", "C23_real", "C23_imag", "C33"]
    elif has_all(["C11", "C22"]) and not find_file(infolder, "C33"):
        matrix_type = "C2"
        base_names = ["C11", "C12_real", "C12_imag", "C22"]
    elif has_all(["T11", "T22"]) and not find_file(infolder, "T33"):
        matrix_type = "T2"
        base_names = ["T11", "T12_real", "T12_imag", "T22"]
    else:
        raise ValueError("Unsupported or incomplete matrix data.")

    input_filepaths = collect_input_files(infolder, base_names)

    matrix_prefix = matrix_type[0]  # e.g., 'C', 'T', 'S'
    ext = '.bin' if fmt == 'bin' else '.tif'
    if filter_type and sub_dir:
        outFolder = os.path.join(
            os.path.dirname(infolder), f"{filter_type}_{window_size[0]}x{window_size[1]}",
            os.path.basename(infolder)
        )
    elif filter_type and not sub_dir:
        outFolder = os.path.join(
            os.path.dirname(infolder) + f"_{filter_type}_{window_size[0]}x{window_size[1]}",
            os.path.basename(infolder)
        )
    else:
        outFolder = os.path.join(
            os.path.dirname(infolder) + f"_{window_size[0]}x{window_size[1]}",
            os.path.basename(infolder)
        )

    os.makedirs(outFolder, exist_ok=True)

    output_filepaths = [os.path.join(outFolder, f"{name.replace(name[:1], matrix_prefix)}{ext}") for name in base_names]

    return input_filepaths, output_filepaths