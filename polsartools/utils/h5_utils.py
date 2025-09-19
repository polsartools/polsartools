

import os
import glob
import numpy as np
import tables
from skimage.util.shape import view_as_blocks
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import sys
from netCDF4 import Dataset
from osgeo import gdal, osr
gdal.UseExceptions()
import multiprocessing



import warnings
warnings.filterwarnings("ignore", category=tables.exceptions.DataTypeWarning)
warnings.filterwarnings(action='ignore', message='Mean of empty slice')


def get_ml_chunk(multiplier, default_size):
    # Rounds up to the next multiple of `multiplier`
    return ((default_size + multiplier - 1) // multiplier) * multiplier

def get_dtype(numpy_dtype):
    """Map numpy dtype to GDAL data type."""
    dtype = np.dtype(numpy_dtype)  # Ensure it's a dtype object
    return {
        np.uint8: gdal.GDT_Byte,
        np.int16: gdal.GDT_Int16,
        np.uint16: gdal.GDT_UInt16,
        np.int32: gdal.GDT_Int32,
        np.uint32: gdal.GDT_UInt32,
        np.float32: gdal.GDT_Float32,
        np.float64: gdal.GDT_Float64,
        np.complex64: gdal.GDT_CFloat32,
        np.complex128: gdal.GDT_CFloat64
    }.get(dtype.type, gdal.GDT_Unknown)
def mlook_arr(data, az, rg):
    temp = data[0:data.shape[0] - data.shape[0] % az,
                0:data.shape[1] - data.shape[1] % rg]
    blocks = view_as_blocks(temp, block_shape=(az, rg))
    flatten = blocks.reshape(blocks.shape[0], blocks.shape[1], -1)
    return np.nanmean(flatten, axis=2)



def compute_elements(chunks, matrix_type, azlks, rglks, apply_multilook,recip,calibration_constant=1):
    if matrix_type == "S2":
        return compute_s2(chunks,calibration_constant,recip)     
    elif matrix_type == "C4":
        return compute_c4(chunks, azlks, rglks, apply_multilook,calibration_constant,recip)    
    elif matrix_type == "C3":
        return compute_c3(chunks, azlks, rglks, apply_multilook,calibration_constant)
    elif matrix_type == "T3":
        return compute_t3(chunks, azlks, rglks, apply_multilook,calibration_constant)
    elif matrix_type == "T4":
        return compute_t4(chunks, azlks, rglks, apply_multilook,calibration_constant,recip) 
    elif matrix_type == "T2HV":
        return compute_t2hv(chunks, azlks, rglks, apply_multilook,calibration_constant)   
    elif matrix_type == "C2HV":
        return compute_c2hv(chunks, azlks, rglks, apply_multilook,calibration_constant)
    elif matrix_type == "C2HX":
        return compute_c2hx(chunks, azlks, rglks, apply_multilook,calibration_constant)
    elif matrix_type == "C2VX":
        return compute_c2vx(chunks, azlks, rglks, apply_multilook,calibration_constant)
    elif matrix_type == "Sxy":
        return compute_sxy(chunks, calibration_constant)

    else:
        raise ValueError(f"Unsupported matrix type: {matrix_type}")
        
def compute_c3(chunks, azlks, rglks, apply_multilook,calibration_constant):
    def opt_mlook(data):
        return mlook_arr(data, azlks, rglks) if apply_multilook else data

    Kl = np.array([chunks["HH"]/calibration_constant, 
                   np.sqrt(2)*0.5*(chunks["HV"]/calibration_constant+chunks["VH"]/calibration_constant), 
                   chunks["VV"]/calibration_constant])

    return {
        "C11": opt_mlook(np.real(np.abs(Kl[0])**2)),
        "C12_real": opt_mlook(np.real(Kl[0]*np.conj(Kl[1]))),
        "C12_imag": opt_mlook(np.imag(Kl[0]*np.conj(Kl[1]))),
        "C13_real": opt_mlook(np.real(Kl[0]*np.conj(Kl[2]))),
        "C13_imag": opt_mlook(np.imag(Kl[0]*np.conj(Kl[2]))),
        "C22": opt_mlook(np.real(np.abs(Kl[1])**2)),
        "C23_real": opt_mlook(np.real(Kl[1]*np.conj(Kl[2]))),
        "C23_imag": opt_mlook(np.imag(Kl[1]*np.conj(Kl[2]))),
        "C33": opt_mlook(np.real(np.abs(Kl[2])**2))
    }

def compute_s2(chunks,calibration_constant,recip=False):
    if recip:
        return {
            "s11": chunks["VV"]/calibration_constant,
            "s12": (chunks["HV"]/calibration_constant + chunks["VH"]/calibration_constant)*0.5,
            "s21": (chunks["HV"]/calibration_constant + chunks["VH"]/calibration_constant)*0.5,
            "s22": chunks["HH"]/calibration_constant,
        }
    else:
        return {
            "s11": chunks["HH"]/calibration_constant,
            "s12": chunks["HV"]/calibration_constant,
            "s21": chunks["VH"]/calibration_constant,
            "s22": chunks["VV"]/calibration_constant,
        }


def compute_sxy(chunks, calibration_constant):
    # Identify available channels
    available_channels = set(chunks.keys())

    # Define co-pol and cross-pol combinations
    copol_priority = ['HH', 'VV']
    crosspol_priority = ['HV', 'VH']

    # Select co-pol channel
    copol = next((ch for ch in copol_priority if ch in available_channels), None)
    crosspol = next((ch for ch in crosspol_priority if ch in available_channels), None)

    if copol and crosspol:
        return {
            "s11": chunks[copol] / calibration_constant,
            "s12": chunks[crosspol] / calibration_constant
        }
    else:
        raise ValueError(f"Insufficient dual-pol channels. Found: {available_channels}")



def compute_c4(chunks, azlks, rglks, apply_multilook,calibration_constant,recip=False):
    def opt_mlook(data):
        return mlook_arr(data, azlks, rglks) if apply_multilook else data
    
    if recip:
        chunks["HV"] = (chunks["HV"]+chunks["VH"])*0.5
        chunks["VH"] = chunks["HV"]

    Kl = np.array([chunks["HH"]/calibration_constant, 
                chunks["HV"]/calibration_constant, 
                chunks["VH"]/calibration_constant, 
                chunks["VV"]/calibration_constant])

    return {
        "C11": opt_mlook(np.real(np.abs(Kl[0])**2)),
        "C12_real": opt_mlook(np.real(Kl[0]*np.conj(Kl[1]))),
        "C12_imag": opt_mlook(np.imag(Kl[0]*np.conj(Kl[1]))),
        "C13_real": opt_mlook(np.real(Kl[0]*np.conj(Kl[2]))),
        "C13_imag": opt_mlook(np.imag(Kl[0]*np.conj(Kl[2]))),
        "C14_real": opt_mlook(np.real(Kl[0]*np.conj(Kl[3]))),
        "C14_imag": opt_mlook(np.imag(Kl[0]*np.conj(Kl[3]))),
        "C22": opt_mlook(np.real(np.abs(Kl[1])**2)),
        "C23_real": opt_mlook(np.real(Kl[1]*np.conj(Kl[2]))),
        "C23_imag": opt_mlook(np.imag(Kl[1]*np.conj(Kl[2]))),
        "C24_real": opt_mlook(np.real(Kl[1]*np.conj(Kl[3]))),
        "C24_imag": opt_mlook(np.imag(Kl[1]*np.conj(Kl[3]))),
        "C33": opt_mlook(np.real(np.abs(Kl[2])**2)),
        "C34_real": opt_mlook(np.real(Kl[2]*np.conj(Kl[3]))),
        "C34_imag": opt_mlook(np.imag(Kl[2]*np.conj(Kl[3]))),
        "C44": opt_mlook(np.real(np.abs(Kl[3])**2))
    }

def compute_c2hv(chunks, azlks, rglks, apply_multilook,calibration_constant):
    def opt_mlook(data):
        return mlook_arr(data, azlks, rglks) if apply_multilook else data

    C11 = opt_mlook(np.abs(chunks["HH"]/calibration_constant)**2).astype(np.float32)
    C22 = opt_mlook(np.abs(chunks["VV"]/calibration_constant)**2).astype(np.float32)
    C12 = opt_mlook(chunks["HH"]/calibration_constant * np.conj(chunks["VV"]/calibration_constant)).astype(np.complex64)

    return {
        "C11": C11,
        "C12_real": np.real(C12),
        "C12_imag": np.imag(C12),
        "C22": C22
    }

def compute_c2hx(chunks, azlks, rglks, apply_multilook,calibration_constant):
    def opt_mlook(data):
        return mlook_arr(data, azlks, rglks) if apply_multilook else data

    C11 = opt_mlook(np.abs(chunks["HH"]/calibration_constant)**2).astype(np.float32)
    C22 = opt_mlook(np.abs(chunks["HV"]/calibration_constant)**2).astype(np.float32)
    C12 = opt_mlook(chunks["HH"]/calibration_constant * np.conj(chunks["HV"]/calibration_constant)).astype(np.complex64)

    return {
        "C11": C11,
        "C12_real": np.real(C12),
        "C12_imag": np.imag(C12),
        "C22": C22
    }

def compute_c2vx(chunks, azlks, rglks, apply_multilook,calibration_constant):
    def opt_mlook(data):
        return mlook_arr(data, azlks, rglks) if apply_multilook else data

    C11 = opt_mlook(np.abs(chunks["VV"]/calibration_constant)**2).astype(np.float32)
    C22 = opt_mlook(np.abs(chunks["VH"]/calibration_constant)**2).astype(np.float32)
    C12 = opt_mlook(chunks["VV"]/calibration_constant * np.conj(chunks["VH"]/calibration_constant)).astype(np.complex64)

    return {
        "C11": C11,
        "C12_real": np.real(C12),
        "C12_imag": np.imag(C12),
        "C22": C22
    }

def compute_t2hv(chunks, azlks, rglks, apply_multilook,calibration_constant):
    def opt_mlook(data):
        return mlook_arr(data, azlks, rglks) if apply_multilook else data

    S1 = chunks["HH"]/calibration_constant + chunks["VV"]/calibration_constant
    S2 = chunks["HH"]/calibration_constant - chunks["VV"]/calibration_constant

    T11 = opt_mlook(np.abs(S1)**2).astype(np.float32)
    T22 = opt_mlook(np.abs(S2)**2).astype(np.float32)
    T12 = opt_mlook(S1 * np.conj(S2)).astype(np.complex64)

    return {
        "T11": T11,
        "T12_real": np.real(T12),
        "T12_imag": np.imag(T12),
        "T22": T22
    }

def compute_t3(chunks, azlks, rglks, apply_multilook,calibration_constant):
    def opt_mlook(data):
        return mlook_arr(data, azlks, rglks) if apply_multilook else data

    # Pauli basis vector for T3: [S1+S2, S1−S2, S12+S21]
    Kp = (1 / np.sqrt(2)) * np.array([
        chunks["HH"]/calibration_constant + chunks["VV"]/calibration_constant,
        chunks["HH"]/calibration_constant - chunks["VV"]/calibration_constant,
        chunks["HV"]/calibration_constant + chunks["VH"]/calibration_constant
    ])

    return {
        "T11": opt_mlook(np.real(np.abs(Kp[0])**2)),
        "T12_real": opt_mlook(np.real(Kp[0]*np.conj(Kp[1]))),
        "T12_imag": opt_mlook(np.imag(Kp[0]*np.conj(Kp[1]))),
        "T13_real": opt_mlook(np.real(Kp[0]*np.conj(Kp[2]))),
        "T13_imag": opt_mlook(np.imag(Kp[0]*np.conj(Kp[2]))),
        "T22": opt_mlook(np.real(np.abs(Kp[1])**2)),
        "T23_real": opt_mlook(np.real(Kp[1]*np.conj(Kp[2]))),
        "T23_imag": opt_mlook(np.imag(Kp[1]*np.conj(Kp[2]))),
        "T33": opt_mlook(np.real(np.abs(Kp[2])**2))
    }

def compute_t4(chunks, azlks, rglks, apply_multilook,calibration_constant,recip=False):
    def opt_mlook(data):
        return mlook_arr(data, azlks, rglks) if apply_multilook else data

    if recip:
        chunks["HV"] = (chunks["HV"]+chunks["VH"])*0.5
        chunks["VH"] = chunks["HV"]

    # Extended Pauli basis vector for T4
    Kp = (1 / np.sqrt(2)) * np.array([
        chunks["HH"]/calibration_constant + chunks["VV"]/calibration_constant,
        chunks["HH"]/calibration_constant - chunks["VV"]/calibration_constant,
        chunks["HV"]/calibration_constant + chunks["VH"]/calibration_constant,
        1j * (chunks["HV"]/calibration_constant - chunks["VH"]/calibration_constant)
    ])

    return {
        "T11": opt_mlook(np.real(np.abs(Kp[0])**2)),
        "T12_real": opt_mlook(np.real(Kp[0]*np.conj(Kp[1]))),
        "T12_imag": opt_mlook(np.imag(Kp[0]*np.conj(Kp[1]))),
        "T13_real": opt_mlook(np.real(Kp[0]*np.conj(Kp[2]))),
        "T13_imag": opt_mlook(np.imag(Kp[0]*np.conj(Kp[2]))),
        "T14_real": opt_mlook(np.real(Kp[0]*np.conj(Kp[3]))),
        "T14_imag": opt_mlook(np.imag(Kp[0]*np.conj(Kp[3]))),
        "T22": opt_mlook(np.real(np.abs(Kp[1])**2)),
        "T23_real": opt_mlook(np.real(Kp[1]*np.conj(Kp[2]))),
        "T23_imag": opt_mlook(np.imag(Kp[1]*np.conj(Kp[2]))),
        "T24_real": opt_mlook(np.real(Kp[1]*np.conj(Kp[3]))),
        "T24_imag": opt_mlook(np.imag(Kp[1]*np.conj(Kp[3]))),
        "T33": opt_mlook(np.real(np.abs(Kp[2])**2)),
        "T34_real": opt_mlook(np.real(Kp[2]*np.conj(Kp[3]))),
        "T34_imag": opt_mlook(np.imag(Kp[2]*np.conj(Kp[3]))),
        "T44": opt_mlook(np.real(np.abs(Kp[3])**2))
    }


def get_chunk_jobs(h5_file, dataset_path, chunk_size_x, chunk_size_y):
    # import tables
    jobs = []
    with tables.open_file(h5_file, 'r') as h5:
        arr = h5.get_node(dataset_path)
        height, width = arr.shape
        for y in range(0, height, chunk_size_y):
            for x in range(0, width, chunk_size_x):
                jobs.append({
                    "x_start": x,
                    "x_end": min(x + chunk_size_x, width),
                    "y_start": y,
                    "y_end": min(y + chunk_size_y, height)
                })
    return jobs



def process_and_write_tile(job, h5_file, dataset_paths,  azlks, rglks, matrix_type,apply_multilook,recip, temp_dir,
                           start_x, start_y, xres, yres, epsg,
                           calibration_constant=1):
    os.makedirs(temp_dir, exist_ok=True)
    x0, x1, y0, y1 = job["x_start"], job["x_end"], job["y_start"], job["y_end"]

    chunks = {}
    with tables.open_file(h5_file, 'r') as h5:
        for name, path in dataset_paths.items():
            chunks[name] = h5.get_node(path)[y0:y1, x0:x1]

    # results = compute_c3_elements(chunks, azlks, rglks, apply_multilook)
    results = compute_elements(chunks, matrix_type, azlks, rglks, apply_multilook, recip,calibration_constant)

    for name, data in results.items():
        save_tiff(name, data, x0, y0, azlks, rglks, apply_multilook, temp_dir,
                  start_x=start_x, start_y=start_y, xres=xres, yres=yres, epsg=epsg)

def save_tiff(name, data, x0, y0, azlks, rglks, apply_multilook, temp_dir,
              start_x=0, start_y=0, xres=1.0, yres=1.0, epsg=4326):


    h_out, w_out = data.shape
    out_path = os.path.join(temp_dir, f"{name}_{y0}_{x0}.tif")
    driver = gdal.GetDriverByName('GTiff')
    dst = driver.Create(out_path, w_out, h_out, 1, gdal.GDT_Float32)

    if not apply_multilook:
        azlks = rglks = 1
        
    if x0==0 and y0==0:
        res_x = xres * 1
        res_y = yres * 1
    else:        
        res_x = xres * rglks
        res_y = yres * azlks

    gt_x = start_x + (x0 // rglks) * res_x
    gt_y = start_y - (y0 // azlks) * res_y

    dst.SetGeoTransform([gt_x, res_x, 0, gt_y, 0, -res_y])
    srs = osr.SpatialReference(); srs.ImportFromEPSG(epsg)
    dst.SetProjection(srs.ExportToWkt())
    dst.GetRasterBand(1).WriteArray(data)
    dst.GetRasterBand(1).SetNoDataValue(0)
    # dst.SetMetadata({
    #     'AzimuthLooks': str(azlks),
    #     'RangeLooks': str(rglks),
    #     'Multilooked': str(apply_multilook)
    # })
    dst = None


def mosaic_chunks(element_name, temp_dir, output_dir, chunk_size_x, chunk_size_y, 
                  azlks, rglks, apply_multilook,
                  start_x=0, start_y=0, xres=1.0, yres=1.0, epsg=4326,
                  fmt='tif',cog=False,ovr = [2, 4, 8, 16],comp=False,
                  dtype=np.float32,
                  inshape=None,outshape=None
                  
                  ):

    dtype = get_dtype(dtype)
    os.makedirs(output_dir, exist_ok=True)
    chunk_files = sorted(glob.glob(os.path.join(temp_dir, f"{element_name}_*.tif")))
    if not chunk_files:
        print(f"No chunks found for {element_name}")
        return

    if not apply_multilook:
        azlks = rglks = 1
    
    dataset = gdal.Open(chunk_files[0], gdal.GA_ReadOnly)
    in_geotransform = dataset.GetGeoTransform()
    
    out_geotransform = list(in_geotransform)
    normalized = tuple(round(float(x), 6) for x in in_geotransform)
    if normalized in [
        (0.0, 1.0, 0.0, 0.0, 0.0, -1.0),
        (0.0, 1.0, 0.0, 0.0, 0.0,  1.0)
                    ]:
        # out_geotransform[1] *= 1 
        # out_geotransform[5] *= 1 
        
        res_x = xres * 1
        res_y = yres * 1
    else:
        # out_geotransform[1] = (in_geotransform[1] * in_cols) / out_x_size
        # out_geotransform[5] = (in_geotransform[5] * in_rows) / out_y_size

        res_x = xres * rglks
        res_y = yres * azlks
    
    # print(res_x,res_y)
    
    if apply_multilook and (inshape is not None) and (outshape is not None):
        old_width = inshape[0]
        old_height = inshape[1]
        ml_width = outshape[0]
        ml_height = outshape[1]
        res_x = (xres * old_width) / ml_width
        res_y = (yres * old_height) / ml_height
        
        # print(res_x,res_y)
    

    max_x = 0
    max_y = 0
    offsets = []

    for f in chunk_files:
        y0 = int(f.split("_")[-2])
        x0 = int(f.split("_")[-1].split(".")[0])
        x_ml = x0 // rglks
        y_ml = y0 // azlks
        tile = gdal.Open(f)
        h, w = tile.RasterYSize, tile.RasterXSize
        max_x = max(max_x, x_ml + w)
        max_y = max(max_y, y_ml + h)
        offsets.append((f, x_ml, y_ml))
        tile = None

    driver = gdal.GetDriverByName('ENVI') if fmt == 'bin' else gdal.GetDriverByName('GTiff')

    options = ['BIGTIFF=IF_SAFER']
    if comp:
        # options += ['COMPRESS=DEFLATE', 'PREDICTOR=2', 'ZLEVEL=9']
        options += ['COMPRESS=LZW']
    if cog:
        options += ['TILED=YES', 'BLOCKXSIZE=512', 'BLOCKYSIZE=512']
    
    options = options if fmt == 'tif' else []
    
    
    out_path = os.path.join(output_dir, f"{element_name}.{fmt}")
    dst = driver.Create(out_path, max_x, max_y, 1, dtype, options=options)

    dst.SetGeoTransform([start_x, res_x, 0, start_y, 0, -abs(res_y)])
   
    # print(start_x, start_y, res_x, -abs(res_y))
    srs = osr.SpatialReference(); srs.ImportFromEPSG(epsg)
    dst.SetProjection(srs.ExportToWkt())

    for f, x_ml, y_ml in offsets:
        tile = gdal.Open(f)
        data = tile.GetRasterBand(1).ReadAsArray()
        dst.GetRasterBand(1).WriteArray(data, x_ml, y_ml)
        tile = None

    dst.GetRasterBand(1).SetNoDataValue(0)
    if apply_multilook:
        dst.SetMetadata({
            'AzimuthLooks': str(azlks),
            'RangeLooks': str(rglks),
        })

    dst.SetMetadataItem('Generated by','polsartools')
    dst.FlushCache()
    if cog:
        dst.BuildOverviews("NEAREST", ovr)
    # dst.SetMetadataItem('TIFFTAG_DOCUMENTNAME', element_name)
    # dst.SetMetadataItem('DESCRIPTION', element_name)
    # dst.SetDescription(element_name)
    dst = None
    print(f"Saved file {out_path}")


def cleanup_temp_files(temp_dir):
    for f in glob.glob(os.path.join(temp_dir, "*.tif")):
        os.remove(f)
    
    os.rmdir(temp_dir)


def h5_polsar(h5_file, dataset_paths, output_dir, temp_dir,
                            azlks, rglks,matrix_type, apply_multilook,recip=False,
                            chunk_size_x=512, chunk_size_y=512,
                            max_workers=None,
                            start_x=None, start_y=None, xres=1.0, yres=1.0, epsg=4326,
                            fmt='tif',cog=False,ovr = [2, 4, 8, 16],comp=False,
                            dtype=np.float32,
                            inshape=None,outshape=None,
                            calibration_constant=1):
    if max_workers is None:
        max_workers = max(multiprocessing.cpu_count() - 1, 1)
    
    polarization_key = 'HH' if 'HH' in dataset_paths else 'VV' if 'VV' in dataset_paths else None
    
    # jobs = get_chunk_jobs(h5_file, dataset_paths["HH"], chunk_size_x, chunk_size_y)
    jobs = get_chunk_jobs(h5_file, dataset_paths[polarization_key], chunk_size_x, chunk_size_y)

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(process_and_write_tile, job, h5_file, dataset_paths,
                               azlks, rglks, matrix_type, apply_multilook, recip,temp_dir, 
                               start_x=start_x, start_y=start_y, xres=xres, yres=yres, epsg=epsg,
                               calibration_constant=calibration_constant) for job in jobs]
        with tqdm(total=len(futures), desc="Processing chunks") as pbar:
            for _ in as_completed(futures):
                pbar.update(1)

    # Discover output band names
    dummy_shape = (azlks * 2, rglks * 2)
    dummy_data = {k: np.zeros(dummy_shape, dtype=np.complex64) for k in dataset_paths.keys()}
    keys = compute_elements(dummy_data, matrix_type, azlks, rglks, 
                            apply_multilook,recip,calibration_constant).keys()

    for name in keys:
        mosaic_chunks(name, temp_dir, output_dir, chunk_size_x, chunk_size_y,
                      azlks, rglks, apply_multilook, 
                      start_x, start_y, xres, yres, epsg, 
                      fmt,cog,ovr,comp, dtype,inshape,outshape)

    cleanup_temp_files(temp_dir)