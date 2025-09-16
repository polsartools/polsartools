import numpy as np
from osgeo import gdal 
gdal.UseExceptions()
from skimage.util.shape import view_as_blocks
import os
def read_bin(file):
    ds = gdal.Open(file,gdal.GA_ReadOnly)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    return arr

def write_bin(file,wdata):
    
    # ds = gdal.Open(refData)
    [cols, rows] = wdata.shape

    driver = gdal.GetDriverByName("ENVI")
    outdata = driver.Create(file, rows, cols, 1, gdal.GDT_Float32)
    # outdata.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
    # outdata.SetProjection(ds.GetProjection())##sets same projection as input
    
    outdata.SetDescription(file)
    outdata.GetRasterBand(1).WriteArray(wdata)
    # outdata.GetRasterBand(1).SetNoDataValue(np.NaN)##if you want these values transparent
    outdata.FlushCache() ##saves to disk!! 

def write_tif(filepath, array, reference_path=None):
    driver = gdal.GetDriverByName("GTiff")
    height, width = array.shape
    ds = driver.Create(filepath, width, height, 1, gdal.GDT_Float32)

    if reference_path:
        ref = gdal.Open(reference_path)
        ds.SetGeoTransform(ref.GetGeoTransform())
        ds.SetProjection(ref.GetProjection())

    ds.GetRasterBand(1).WriteArray(array)
    ds.FlushCache()
    
    
def write_rst(filepath, array, driver_name="ENVI", reference_path=None, nodata_value=None):
    if array.ndim != 2:
        raise ValueError("Only 2D arrays are supported.")
    rows, cols = array.shape
    driver = gdal.GetDriverByName(driver_name)
    options = None
    if driver_name == "GTiff":
        options = [
            "COMPRESS=DEFLATE",
            "PREDICTOR=2",
            "TILED=YES",
            "ZLEVEL=9",
            "BIGTIFF=YES",
            # "BLOCKXSIZE=256",
            # "BLOCKYSIZE=256"
        ]

    ds = driver.Create(filepath, cols, rows, 1, gdal.GDT_Float32, options=options)

    if reference_path:
        ref = gdal.Open(reference_path)
        ds.SetGeoTransform(ref.GetGeoTransform())
        ds.SetProjection(ref.GetProjection())
        ref = None
    band = ds.GetRasterBand(1)
    band.WriteArray(array)
    if nodata_value is not None:
        band.SetNoDataValue(nodata_value)
    ds.SetDescription(filepath)
    ds.FlushCache()


def mlook_arr(data,az,rg):
    temp = data[0:data.shape[0]-data.shape[0]%az,0:data.shape[1]-data.shape[1]%rg]
    blocks = view_as_blocks(temp, block_shape=(az, rg))
    flatten = blocks.reshape(blocks.shape[0], blocks.shape[1], -1)
    mean = np.nanmean(flatten, axis=2)
    return mean

def write_s2_bin(file,wdata):
    [cols, rows] = wdata.shape
    driver = gdal.GetDriverByName("ENVI")
    outdata = driver.Create(file, rows, cols, 1, gdal.GDT_CFloat64)
    outdata.SetDescription(file)
    outdata.GetRasterBand(1).WriteArray(wdata)
    outdata.FlushCache()

def write_s2_bin_ref(file, wdata_ref, chunk_size=1000):
    """Writes large datasets lazily to an ENVI binary file using GDAL."""
    
    rows, cols = wdata_ref.shape  # h5py.Dataset directly provides shape
    driver = gdal.GetDriverByName("ENVI")
    outdata = driver.Create(file, cols, rows, 1, gdal.GDT_CFloat64)  # Maintain correct order
    outdata.SetDescription(file)
    band = outdata.GetRasterBand(1)

    # Process dataset in chunks instead of loading everything into RAM
    for i in range(0, rows, chunk_size):
        row_end = min(i + chunk_size, rows)
        chunk = wdata_ref[i:row_end, :]
        band.WriteArray(chunk, 0, i)  # Write at correct position

    outdata.FlushCache()
    print(f"Saved file {file}")

 

def write_C3(C3_list,folder,fmt='bin'):
    
    out_file = folder +'/C11.bin'
    write_bin(out_file,C3_list[0])
    print("Saved file "+out_file)
    out_file = folder +'/C12_real.bin'
    write_bin(out_file,C3_list[1])
    print("Saved file "+out_file)
    out_file = folder +'/C12_imag.bin'
    write_bin(out_file,C3_list[2])
    print("Saved file "+out_file)
    
    out_file = folder +'/C13_real.bin'
    write_bin(out_file,C3_list[3])
    print("Saved file "+out_file)
    out_file = folder +'/C13_imag.bin'
    write_bin(out_file,C3_list[4])
    print("Saved file "+out_file)
    
    
    out_file = folder +'/C22.bin'
    write_bin(out_file,C3_list[5])
    print("Saved file "+out_file)
    out_file = folder +'/C23_real.bin'
    write_bin(out_file,C3_list[6])
    print("Saved file "+out_file)
    out_file = folder +'/C23_imag.bin'
    write_bin(out_file,C3_list[7])
    print("Saved file "+out_file)
    out_file = folder +'/C33.bin'
    write_bin(out_file,C3_list[8])
    print("Saved file "+out_file)
    
    rows, cols = np.shape(C3_list[0])
    file = folder +'/config.txt'
    file = open(file,"w+")
    file.write('Nrow\n%d\n---------\nNcol\n%d\n---------\nPolarCase\nmonostatic\n---------\nPolarType\nfull'%(rows,cols))
    file.close()

    
def write_C4(C4_list,folder):
    
    out_file = folder +'/C11.bin'
    write_bin(out_file,C4_list[0])
    print("Saved file "+out_file)
    out_file = folder +'/C12_real.bin'
    write_bin(out_file,C4_list[1])
    print("Saved file "+out_file)
    out_file = folder +'/C12_imag.bin'
    write_bin(out_file,C4_list[2])
    print("Saved file "+out_file)
    

    
    out_file = folder +'/C13_real.bin'
    write_bin(out_file,C4_list[3])
    print("Saved file "+out_file)
    out_file = folder +'/C13_imag.bin'
    write_bin(out_file,C4_list[4])
    print("Saved file "+out_file)
    
    out_file = folder +'/C14_real.bin'
    write_bin(out_file,C4_list[5])
    print("Saved file "+out_file)
    out_file = folder +'/C14_imag.bin'
    write_bin(out_file,C4_list[6])
    print("Saved file "+out_file)

    out_file = folder +'/C22.bin'
    write_bin(out_file,C4_list[7])
    print("Saved file "+out_file)
    out_file = folder +'/C23_real.bin'
    write_bin(out_file,C4_list[8])
    print("Saved file "+out_file)
    out_file = folder +'/C23_imag.bin'
    write_bin(out_file,C4_list[9])
    print("Saved file "+out_file)
    
    out_file = folder +'/C24_real.bin'
    write_bin(out_file,C4_list[10])
    print("Saved file "+out_file)
    out_file = folder +'/C24_imag.bin'
    write_bin(out_file,C4_list[11])
    print("Saved file "+out_file)
    
    
    
    out_file = folder +'/C33.bin'
    write_bin(out_file,C4_list[12])
    print("Saved file "+out_file)
    
    
    out_file = folder +'/C34_real.bin'
    write_bin(out_file,C4_list[13])
    print("Saved file "+out_file)
    out_file = folder +'/C34_imag.bin'
    write_bin(out_file,C4_list[14])
    print("Saved file "+out_file)    
    
    
    out_file = folder +'/C44.bin'
    write_bin(out_file,C4_list[15])
    print("Saved file "+out_file)  
    
    
    
    rows, cols = np.shape(C4_list[0])
    file = folder +'/config.txt'
    file = open(file,"w+")
    file.write('Nrow\n%d\n---------\nNcol\n%d\n---------\nPolarCase\nmonostatic\n---------\nPolarType\nfull'%(rows,cols))
    file.close()


def write_T3(t3_list, folder, fmt="tif", reference_path=None):

    os.makedirs(folder, exist_ok=True)

    keys = [
        "T11", "T12_real", "T12_imag", "T13_real", "T13_imag",
        "T22", "T23_real", "T23_imag",
        "T33"
    ]
    
    ext = ".bin" if fmt == "bin" else ".tif"
    driver_name = "ENVI" if fmt == "bin" else "GTiff"
    
    for name, array in zip(keys, t3_list):
        out_path = os.path.join(folder, f"{name}{ext}")
        write_rst(out_path, array, driver_name, reference_path)
        print(f"Saved file {out_path}")
    
    rows, cols = t3_list[0].shape
    config_path = os.path.join(folder, "config.txt")
    with open(config_path, "w") as f:
        f.write(
            f"Nrow\n{rows}\n---------\n"
            f"Ncol\n{cols}\n---------\n"
            f"PolarCase\nmonostatic\n---------\n"
            f"PolarType\nfull"
        )

def write_T4(t4_list, folder, fmt="tif", reference_path=None):
    """
    Saves T4 matrix to the specified folder in .bin or .tif format.
    Also writes a config.txt file with metadata.
    """
    os.makedirs(folder, exist_ok=True)

    keys = [
        "T11", "T12_real", "T12_imag", "T13_real", "T13_imag", "T14_real", "T14_imag",
        "T22", "T23_real", "T23_imag", "T24_real", "T24_imag",
        "T33", "T34_real", "T34_imag", "T44"
    ]
    
    ext = ".bin" if fmt == "bin" else ".tif"
    driver_name = "ENVI" if fmt == "bin" else "GTiff"
    
    for name, array in zip(keys, t4_list):
        out_path = os.path.join(folder, f"{name}{ext}")
        write_rst(out_path, array, driver_name, reference_path)
        print(f"Saved file {out_path}")
    
    rows, cols = t4_list[0].shape
    config_path = os.path.join(folder, "config.txt")
    with open(config_path, "w") as f:
        f.write(
            f"Nrow\n{rows}\n---------\n"
            f"Ncol\n{cols}\n---------\n"
            f"PolarCase\nmonostatic\n---------\n"
            f"PolarType\nfull"
        )
   
def write_bin_s2(file,wdata,refData):
    
    # ds = gdal.Open(refData)
    [cols, rows] = wdata.shape

    driver = gdal.GetDriverByName("ENVI")
    outdata = driver.Create(file, rows, cols, 1, gdal.GDT_Float32)
    # outdata.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
    # outdata.SetProjection(ds.GetProjection())##sets same projection as input
    
    outdata.SetDescription(file)
    outdata.GetRasterBand(1).WriteArray(wdata)
    # outdata.GetRasterBand(1).SetNoDataValue(np.NaN)##if you want these values transparent
    outdata.FlushCache() ##saves to disk!!   
    

def write_s2_ct(matrix_type, matrixFolder, K, azlks, rglks):
    """
    Processes and saves C3, T3, C4, or T4 matrices sequentially.
    
    Args:
        matrix_type (str): One of "C3", "T3", "C4", "T4".
        K (np.ndarray): Covariance matrix elements.
        azlks (int): Azimuth looks.
        rglks (int): Range looks.

    Saves:
        Binary files for matrix elements in the appropriate directory.
        Configuration file specifying matrix dimensions.
    """

    if matrix_type not in ["C3", "T3", "C4", "T4"]:
        raise ValueError("Invalid matrix type. Choose 'C3', 'T3', 'C4', or 'T4'.")

    prefix = matrix_type[0]  # Extract first character ('C' or 'T')

    def process_and_write(key, computation):
        """ Helper function to compute and write each element sequentially. """
        matrix = computation.astype(np.float32)
        out_file = os.path.join(matrixFolder, f"{key}.bin")
        write_bin(out_file, matrix)
        print(f"Saved file {out_file}")
        return matrix.shape  # Capture shape for config file

    # Process each matrix element one by one
    rows, cols = process_and_write(f'{prefix}11', mlook_arr(np.abs(K[0])**2, azlks, rglks))
    process_and_write(f'{prefix}22', mlook_arr(np.abs(K[1])**2, azlks, rglks))
    process_and_write(f'{prefix}33', mlook_arr(np.abs(K[2])**2, azlks, rglks))
    process_and_write(f'{prefix}12_real', mlook_arr(np.real(K[0] * np.conj(K[1])), azlks, rglks))
    process_and_write(f'{prefix}12_imag', mlook_arr(np.imag(K[0] * np.conj(K[1])), azlks, rglks))
    process_and_write(f'{prefix}13_real', mlook_arr(np.real(K[0] * np.conj(K[2])), azlks, rglks))
    process_and_write(f'{prefix}13_imag', mlook_arr(np.imag(K[0] * np.conj(K[2])), azlks, rglks))
    process_and_write(f'{prefix}23_real', mlook_arr(np.real(K[1] * np.conj(K[2])), azlks, rglks))
    process_and_write(f'{prefix}23_imag', mlook_arr(np.imag(K[1] * np.conj(K[2])), azlks, rglks))

    # Extend for C4/T4 processing if K has 4 elements
    if len(K) == 4:
        process_and_write(f'{prefix}44', mlook_arr(np.abs(K[3])**2, azlks, rglks))
        process_and_write(f'{prefix}14_real', mlook_arr(np.real(K[0] * np.conj(K[3])), azlks, rglks))
        process_and_write(f'{prefix}14_imag', mlook_arr(np.imag(K[0] * np.conj(K[3])), azlks, rglks))
        process_and_write(f'{prefix}24_real', mlook_arr(np.real(K[1] * np.conj(K[3])), azlks, rglks))
        process_and_write(f'{prefix}24_imag', mlook_arr(np.imag(K[1] * np.conj(K[3])), azlks, rglks))
        process_and_write(f'{prefix}34_real', mlook_arr(np.real(K[2] * np.conj(K[3])), azlks, rglks))
        process_and_write(f'{prefix}34_imag', mlook_arr(np.imag(K[2] * np.conj(K[3])), azlks, rglks))

    # Write configuration file
    config_file_path = os.path.join(matrixFolder, 'config.txt')
    with open(config_file_path, "w") as file:
        file.write(f'Nrow\n{rows}\n---------\nNcol\n{cols}\n---------\nPolarCase\nmonostatic\n---------\nPolarType\nfull')
    print(f"Saved config file {config_file_path}")


def write_s2_ct_ref(matrix_type, matrixFolder, K_refs, azlks, rglks):
    """
    Processes and saves C3, T3, C4, or T4 matrices sequentially using lazy loading.

    Args:
        matrix_type (str): One of "C3", "T3", "C4", "T4".
        K_refs (list): References to covariance matrix elements (`h5py.Dataset`).
        azlks (int): Azimuth looks.
        rglks (int): Range looks.

    Saves:
        Binary files for matrix elements in the appropriate directory.
        Configuration file specifying matrix dimensions.
    """

    if matrix_type not in ["C3", "T3", "C4", "T4"]:
        raise ValueError("Invalid matrix type. Choose 'C3', 'T3', 'C4', or 'T4'.")

    prefix = matrix_type[0]  # Extract first character ('C' or 'T')

    def process_and_write(key, computation):
        """ Helper function to compute and write each element sequentially. """
        matrix = computation.astype(np.float32)  # Ensure correct type
        out_file = os.path.join(matrixFolder, f"{key}.bin")
        write_bin(out_file, matrix)
        print(f"Saved file {out_file}")
        return matrix.shape  # Capture shape for config file

    # Process each matrix element **lazily** by only loading required portions
    rows, cols = process_and_write(f'{prefix}11', mlook_arr(np.abs(K_refs[0][()])**2, azlks, rglks))
    process_and_write(f'{prefix}22', mlook_arr(np.abs(K_refs[1][()])**2, azlks, rglks))
    process_and_write(f'{prefix}33', mlook_arr(np.abs(K_refs[2][()])**2, azlks, rglks))
    process_and_write(f'{prefix}12_real', mlook_arr(np.real(K_refs[0][()] * np.conj(K_refs[1][()])), azlks, rglks))
    process_and_write(f'{prefix}12_imag', mlook_arr(np.imag(K_refs[0][()] * np.conj(K_refs[1][()])), azlks, rglks))
    process_and_write(f'{prefix}13_real', mlook_arr(np.real(K_refs[0][()] * np.conj(K_refs[2][()])), azlks, rglks))
    process_and_write(f'{prefix}13_imag', mlook_arr(np.imag(K_refs[0][()] * np.conj(K_refs[2][()])), azlks, rglks))
    process_and_write(f'{prefix}23_real', mlook_arr(np.real(K_refs[1][()] * np.conj(K_refs[2][()])), azlks, rglks))
    process_and_write(f'{prefix}23_imag', mlook_arr(np.imag(K_refs[1][()] * np.conj(K_refs[2][()])), azlks, rglks))

    # Extend for C4/T4 processing if K has 4 elements
    if len(K_refs) == 4:
        process_and_write(f'{prefix}44', mlook_arr(np.abs(K_refs[3][()])**2, azlks, rglks))
        process_and_write(f'{prefix}14_real', mlook_arr(np.real(K_refs[0][()] * np.conj(K_refs[3][()])), azlks, rglks))
        process_and_write(f'{prefix}14_imag', mlook_arr(np.imag(K_refs[0][()] * np.conj(K_refs[3][()])), azlks, rglks))
        process_and_write(f'{prefix}24_real', mlook_arr(np.real(K_refs[1][()] * np.conj(K_refs[3][()])), azlks, rglks))
        process_and_write(f'{prefix}24_imag', mlook_arr(np.imag(K_refs[1][()] * np.conj(K_refs[3][()])), azlks, rglks))
        process_and_write(f'{prefix}34_real', mlook_arr(np.real(K_refs[2][()] * np.conj(K_refs[3][()])), azlks, rglks))
        process_and_write(f'{prefix}34_imag', mlook_arr(np.imag(K_refs[2][()] * np.conj(K_refs[3][()])), azlks, rglks))

    # Write configuration file
    config_file_path = os.path.join(matrixFolder, 'config.txt')
    with open(config_file_path, "w") as file:
        file.write(f'Nrow\n{rows}\n---------\nNcol\n{cols}\n---------\nPolarCase\nmonostatic\n---------\nPolarType\nfull')
    print(f"Saved config file {config_file_path}")