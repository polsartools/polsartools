

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
import math
from polsartools.utils.clip_utils import apply_vector_mask

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
    elif matrix_type == "M4":
        return compute_m4(chunks, azlks, rglks, apply_multilook,calibration_constant,recip)
    elif matrix_type == "T2HV":
        return compute_t2hv(chunks, azlks, rglks, apply_multilook,calibration_constant)   
    elif matrix_type == "C2HV":
        return compute_c2hv(chunks, azlks, rglks, apply_multilook,calibration_constant)
    elif matrix_type == "C2HX":
        return compute_c2hx(chunks, azlks, rglks, apply_multilook,calibration_constant)
    elif matrix_type == "C2VX":
        return compute_c2vx(chunks, azlks, rglks, apply_multilook,calibration_constant)
    elif matrix_type == "C2L":
        return compute_c2c(chunks, azlks, rglks, apply_multilook,calibration_constant)
    elif matrix_type == "C2R":
        return compute_c2c(chunks, azlks, rglks, apply_multilook,calibration_constant)
    elif matrix_type == "Sxy":
        return compute_sxy(chunks, calibration_constant)
    elif matrix_type == "I2" or matrix_type == "I4":
        return compute_II(chunks, azlks, rglks, apply_multilook, calibration_constant)
    else:
        raise ValueError(f"Unsupported matrix type: {matrix_type}")

def compute_m4(
    chunks, azlks, rglks, apply_multilook, calibration_constant, recip=False
):
  def opt_mlook(data):
    return mlook_arr(data, azlks, rglks) if apply_multilook else data

  if recip:
    chunks["HV"] = (chunks["HV"] + chunks["VH"]) * 0.5
    chunks["VH"] = chunks["HV"]

  a = chunks["HH"] / calibration_constant
  b = chunks["HV"] / calibration_constant
  c = chunks["VH"] / calibration_constant
  d = chunks["VV"] / calibration_constant

  # Precompute common absolute squares and inner products
  aa = np.abs(a) ** 2
  bb = np.abs(b) ** 2
  cc = np.abs(c) ** 2
  dd = np.abs(d) ** 2

  ab_conj = a * np.conj(b)
  cd_conj = c * np.conj(d)
  ac_conj = a * np.conj(c)
  bd_conj = b * np.conj(d)
  ad_conj = a * np.conj(d)
  bc_conj = b * np.conj(c)

  return {
      # Row 0
      "M00": opt_mlook(0.5 * (aa + bb + cc + dd)),
      "M01": opt_mlook(0.5 * (aa - bb + cc - dd)),
      "M02": opt_mlook(np.real(ab_conj + cd_conj)),
      "M03": opt_mlook(np.imag(ab_conj + cd_conj)),
      # Row 1
      "M10": opt_mlook(0.5 * (aa + bb - cc - dd)),
      "M11": opt_mlook(0.5 * (aa - bb - cc + dd)),
      "M12": opt_mlook(np.real(ab_conj - cd_conj)),
      "M13": opt_mlook(np.imag(ab_conj - cd_conj)),
      # Row 2
      "M20": opt_mlook(np.real(ac_conj + bd_conj)),
      "M21": opt_mlook(np.real(ac_conj - bd_conj)),
      "M22": opt_mlook(np.real(ad_conj + bc_conj)),
      "M23": opt_mlook(np.imag(ad_conj - bc_conj)),
      # Row 3
      "M30": opt_mlook(-np.imag(ac_conj + bd_conj)),
      "M31": opt_mlook(-np.imag(ac_conj - bd_conj)),
      "M32": opt_mlook(-np.imag(ad_conj + bc_conj)),
      "M33": opt_mlook(np.real(ad_conj - bc_conj)),
  }

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


def compute_II(chunks, azlks, rglks, apply_multilook, calibration_constant):
    def opt_mlook(data):
        return mlook_arr(data, azlks, rglks) if apply_multilook else data

    # Identify available channels
    available_channels = set(chunks.keys())

    # Process all channels without priority
    processed = {}
    for ch in available_channels:
        processed[ch] = opt_mlook(chunks[ch] / calibration_constant)

    return processed



def compute_c4(chunks, azlks, rglks, apply_multilook,calibration_constant,recip=False):
    def opt_mlook(data):
        return mlook_arr(data, azlks, rglks) if apply_multilook else data
    if "HH" in chunks and "VV" in chunks and "HV" in chunks and "VH" in chunks:

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
    elif "HHHH" in chunks and "HVHV" in chunks and "VVVV" in chunks and "VHVH" in chunks:

        if recip:
            chunks["HVHV"] = (chunks["HVHV"]+chunks["VHVH"])*0.5
            chunks["VHVH"] = chunks["HVHV"]

        return {
            "C11": opt_mlook(chunks["HHHH"]/calibration_constant),
            "C12_real": opt_mlook(np.real(chunks["HHHV"]/calibration_constant)),
            "C12_imag": opt_mlook(np.imag(chunks["HHHV"]/calibration_constant)),
            "C13_real": opt_mlook(np.real(chunks["HHVH"]/calibration_constant)),
            "C13_imag": opt_mlook(np.imag(chunks["HHVH"]/calibration_constant)),
            "C14_real": opt_mlook(np.real(chunks["HHVV"]/calibration_constant)),
            "C14_imag": opt_mlook(np.imag(chunks["HHVV"]/calibration_constant)),
            "C22": opt_mlook(chunks["HVHV"]/calibration_constant),
            "C23_real": opt_mlook(np.real(chunks["HVVH"]/calibration_constant)),
            "C23_imag": opt_mlook(np.imag(chunks["HVVH"]/calibration_constant)),
            "C24_real": opt_mlook(np.real(chunks["HVVV"]/calibration_constant)),
            "C24_imag": opt_mlook(np.imag(chunks["HVVV"]/calibration_constant)),
            "C33": opt_mlook(chunks["VHVH"]/calibration_constant),
            "C34_real": opt_mlook(np.real(chunks["VHVV"]/calibration_constant)),
            "C34_imag": opt_mlook(np.imag(chunks["VHVV"]/calibration_constant)),
            "C44": opt_mlook(chunks["VVVV"]/calibration_constant)
        }

def compute_c2hv(chunks, azlks, rglks, apply_multilook,calibration_constant):
    def opt_mlook(data):
        return mlook_arr(data, azlks, rglks) if apply_multilook else data

    if "HHHH" in chunks and "VVVV" in chunks:
        C11 = opt_mlook(np.abs(chunks["HHHH"]/calibration_constant)).astype(np.float32)
        C22 = opt_mlook(np.abs(chunks["VVVV"]/calibration_constant)).astype(np.float32)
        C12 = opt_mlook(np.real(chunks["HHVV"])/calibration_constant +1j * np.imag(chunks["HHVV"])/calibration_constant).astype(np.complex64)
    else:

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

    if 'HHHH' in chunks and 'HVHV' in chunks:
        C11 = opt_mlook(np.abs(chunks["HHHH"]/calibration_constant)).astype(np.float32)
        C22 = opt_mlook(np.abs(chunks["HVHV"]/calibration_constant)).astype(np.float32)
        C12 = opt_mlook(np.real(chunks["HHHV"])/calibration_constant +1j * np.imag(chunks["HHHV"])/calibration_constant).astype(np.complex64)
    else:
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
    if 'VVVV' in chunks and 'VHVH' in chunks:
        C11 = opt_mlook(np.abs(chunks["VVVV"]/calibration_constant)).astype(np.float32)
        C22 = opt_mlook(np.abs(chunks["VHVH"]/calibration_constant)).astype(np.float32)
        C12 = opt_mlook(np.real(chunks["VVVH"])/calibration_constant +1j * np.imag(chunks["VVVH"])/calibration_constant).astype(np.complex64)
    else:

        C11 = opt_mlook(np.abs(chunks["VV"]/calibration_constant)**2).astype(np.float32)
        C22 = opt_mlook(np.abs(chunks["VH"]/calibration_constant)**2).astype(np.float32)
        C12 = opt_mlook(chunks["VV"]/calibration_constant * np.conj(chunks["VH"]/calibration_constant)).astype(np.complex64)

    return {
        "C11": C11,
        "C12_real": np.real(C12),
        "C12_imag": np.imag(C12),
        "C22": C22
    }

def compute_c2c(chunks, azlks, rglks, apply_multilook,calibration_constant):
    def opt_mlook(data):
        return mlook_arr(data, azlks, rglks) if apply_multilook else data
    if "RH" in chunks and "RV" in chunks:
        
        C11 = opt_mlook(np.abs(chunks["RH"]/calibration_constant)**2).astype(np.float32)
        C22 = opt_mlook(np.abs(chunks["RV"]/calibration_constant)**2).astype(np.float32)
        C12 = opt_mlook(chunks["RH"]/calibration_constant * np.conj(chunks["RV"]/calibration_constant)).astype(np.complex64)
    elif "LH" in chunks and "LV" in chunks:
        
        C11 = opt_mlook(np.abs(chunks["LH"]/calibration_constant)**2).astype(np.float32)
        C22 = opt_mlook(np.abs(chunks["LV"]/calibration_constant)**2).astype(np.float32)
        C12 = opt_mlook(chunks["LH"]/calibration_constant * np.conj(chunks["LV"]/calibration_constant)).astype(np.complex64)
    
    elif "RHRH" in chunks and "RVRV" in chunks:
        
        C11 = opt_mlook(np.abs(chunks["RHRH"]/calibration_constant)).astype(np.float32)
        C22 = opt_mlook(np.abs(chunks["RVRV"]/calibration_constant)).astype(np.float32)
        C12 = opt_mlook(np.real(chunks["RHRV"]/calibration_constant) + 1j * np.imag(chunks["RHRV"]/calibration_constant)).astype(np.complex64)
    elif "LHLH" in chunks and "LVLV" in chunks:
        
        C11 = opt_mlook(np.abs(chunks["LHLH"]/calibration_constant)).astype(np.float32)
        C22 = opt_mlook(np.abs(chunks["LVLV"]/calibration_constant)).astype(np.float32)
        C12 = opt_mlook(np.real(chunks["LHLV"]/calibration_constant) + 1j * np.imag(chunks["LHLV"]/calibration_constant)).astype(np.complex64)
    else:
        raise ValueError("Neither RH/RV nor LH/LV channels found for C2L/C2R computation.")
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

def get_chunk_jobs(
    h5_file,
    dataset_path,
    chunk_size_x,
    chunk_size_y,
    clip_window=None
):

    jobs = []

    with tables.open_file(h5_file, 'r') as h5:

        arr = h5.get_node(dataset_path)

        height, width = arr.shape

        # Default: process entire raster
        if clip_window is None:

            x_min, x_max = 0, width
            y_min, y_max = 0, height

        else:

            x_min, x_max, y_min, y_max = clip_window

        # Generate only intersecting chunks
        for y in range(
            (y_min // chunk_size_y) * chunk_size_y,
            y_max,
            chunk_size_y
        ):

            for x in range(
                (x_min // chunk_size_x) * chunk_size_x,
                x_max,
                chunk_size_x
            ):

                jobs.append({

                    "x_start": x,

                    "x_end": min(
                        x + chunk_size_x,
                        width
                    ),

                    "y_start": y,

                    "y_end": min(
                        y + chunk_size_y,
                        height
                    )

                })

    return jobs




def process_and_write_tile(job, h5_file, dataset_paths,  azlks, rglks, matrix_type,apply_multilook,recip, temp_dir,
                           start_x, start_y, xres, yres, epsg,
                           calibration_constant=1,rtc_factor_path=None):
    os.makedirs(temp_dir, exist_ok=True)
    x0, x1, y0, y1 = job["x_start"], job["x_end"], job["y_start"], job["y_end"]

    # chunks = {}
    # with tables.open_file(h5_file, 'r') as h5:
    #     for name, path in dataset_paths.items():
    #         chunks[name] = h5.get_node(path)[y0:y1, x0:x1]
    chunks = {}

    with tables.open_file(h5_file, 'r') as h5:

        # Read original GCOV layers
        for name, path in dataset_paths.items():
            chunks[name] = h5.get_node(path)[y0:y1, x0:x1]

        # Apply Gamma0-to-Sigma0 conversion
        if rtc_factor_path is not None:

            factor = h5.get_node(rtc_factor_path)[y0:y1, x0:x1]
            factor = np.asarray(factor, dtype=np.float32)

            for name in chunks:
                chunks[name] = chunks[name] * factor

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


def mosaic_chunks( element_name, temp_dir, output_dir, chunk_size_x, chunk_size_y, azlks, rglks, apply_multilook, start_x=0, start_y=0, xres=1.0, yres=1.0, epsg=4326, fmt='tif', cog=False, ovr=[2, 4, 8, 16], comp=False, dtype=np.float32, inshape=None, outshape=None, clip_window=None ):
    dtype = get_dtype(dtype)

    os.makedirs(output_dir, exist_ok=True)

    chunk_files = sorted(
        glob.glob(os.path.join(temp_dir, f"{element_name}_*.tif"))
    )

    if not chunk_files:
        print(f"No chunks found for {element_name}")
        return

    # --------------------------------------------------
    # 1. Handle multilooking
    # --------------------------------------------------

    if not apply_multilook:
        azlks = 1
        rglks = 1

    # --------------------------------------------------
    # 2. Calculate output resolution
    # --------------------------------------------------

    res_x = xres * rglks
    res_y = abs(yres) * azlks

    # Preserve original full-raster dimensions
    if inshape is None:
        raise ValueError("inshape must be provided.")

    original_width = inshape[0]
    original_height = inshape[1]

    # --------------------------------------------------
    # 3. Calculate output dimensions and origin
    # --------------------------------------------------

    if clip_window is not None:

        x_min, x_max, y_min, y_max = map(int, clip_window)

        # Align clipping boundaries to multilook blocks
        clip_x_start = x_min // rglks
        clip_y_start = y_min // azlks

        clip_x_end = min(
            math.ceil(x_max / rglks),
            math.ceil(original_width / rglks)
        )

        clip_y_end = min(
            math.ceil(y_max / azlks),
            math.ceil(original_height / azlks)
        )

        out_width = clip_x_end - clip_x_start
        out_height = clip_y_end - clip_y_start

    else:

        clip_x_start = 0
        clip_y_start = 0

        out_width = (
            outshape[0]
            if apply_multilook and outshape is not None
            else math.ceil(original_width / rglks)
        )

        out_height = (
            outshape[1]
            if apply_multilook and outshape is not None
            else math.ceil(original_height / azlks)
        )

    if out_width <= 0 or out_height <= 0:
        raise ValueError("Invalid output dimensions.")

    # print("\n========== CLIP DEBUG ==========")

    # print("clip_window:", clip_window)

    # print("inshape:", inshape)

    # print("outshape:", outshape)

    # print("azlks:", azlks)

    # print("rglks:", rglks)

    # print("start_x:", start_x)

    # print("start_y:", start_y)

    # print("xres:", xres)

    # print("yres:", yres)

    # print("================================\n")










    # --------------------------------------------------
    # 4. Calculate output geotransform
    # --------------------------------------------------

    clipped_start_x = start_x + clip_x_start * res_x

    clipped_start_y = start_y - clip_y_start * res_y

    geotransform = [
        clipped_start_x,
        res_x,
        0,
        clipped_start_y,
        0,
        -res_y
    ]

    # --------------------------------------------------
    # 5. Calculate intersecting chunk offsets
    # --------------------------------------------------

    offsets = []

    for f in chunk_files:

        filename = os.path.basename(f)

        y0 = int(filename.split("_")[-2])
        x0 = int(filename.split("_")[-1].split(".")[0])

        tile = gdal.Open(f, gdal.GA_ReadOnly)

        data = tile.GetRasterBand(1).ReadAsArray()

        h, w = data.shape

        tile = None

        # Original tile coordinates in multilooked pixels
        tile_x_start = x0 // rglks
        tile_y_start = y0 // azlks

        tile_x_end = tile_x_start + w
        tile_y_end = tile_y_start + h

        # AOI intersection
        ix0 = max(tile_x_start, clip_x_start)
        iy0 = max(tile_y_start, clip_y_start)

        ix1 = min(
            tile_x_end,
            clip_x_start + out_width
        )

        iy1 = min(
            tile_y_end,
            clip_y_start + out_height
        )

        if ix0 >= ix1 or iy0 >= iy1:
            continue

        # Source offsets inside original tile
        src_x0 = ix0 - tile_x_start
        src_y0 = iy0 - tile_y_start

        src_x1 = ix1 - tile_x_start
        src_y1 = iy1 - tile_y_start

        # Destination offsets inside output raster
        dst_x = ix0 - clip_x_start
        dst_y = iy0 - clip_y_start

        offsets.append((
            f,
            src_x0,
            src_y0,
            src_x1,
            src_y1,
            dst_x,
            dst_y
        ))

    # --------------------------------------------------
    # 6. Create output raster
    # --------------------------------------------------

    driver = (
        gdal.GetDriverByName('ENVI')
        if fmt == 'bin'
        else gdal.GetDriverByName('GTiff')
    )

    options = ['BIGTIFF=IF_SAFER']

    if comp:
        options += ['COMPRESS=LZW']

    if cog:
        options += [
            'TILED=YES',
            'BLOCKXSIZE=512',
            'BLOCKYSIZE=512'
        ]

    if fmt != 'tif':
        options = []

    out_path = os.path.join(
        output_dir,
        f"{element_name}.{fmt}"
    )

    dst = driver.Create(
        out_path,
        out_width,
        out_height,
        1,
        dtype,
        options=options
    )

    if dst is None:
        raise RuntimeError(
            f"Could not create output raster: {out_path}"
        )

    dst.SetGeoTransform(geotransform)

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(int(epsg))

    dst.SetProjection(srs.ExportToWkt())

    band = dst.GetRasterBand(1)

    band.SetNoDataValue(0)

    # --------------------------------------------------
    # 7. Write intersecting chunks
    # --------------------------------------------------

    for (
        f,
        src_x0,
        src_y0,
        src_x1,
        src_y1,
        dst_x,
        dst_y
    ) in offsets:

        tile = gdal.Open(f, gdal.GA_ReadOnly)

        data = tile.GetRasterBand(1).ReadAsArray()

        # Crop tile to AOI intersection
        clipped_data = data[
            src_y0:src_y1,
            src_x0:src_x1
        ]

        band.WriteArray(
            clipped_data,
            dst_x,
            dst_y
        )

        tile = None

    # --------------------------------------------------
    # 8. Metadata
    # --------------------------------------------------

    if apply_multilook:

        dst.SetMetadata({
            'AzimuthLooks': str(azlks),
            'RangeLooks': str(rglks)
        })

    dst.SetMetadataItem(
        'Generated by',
        'polsartools'
    )

    if clip_window is not None:

        dst.SetMetadataItem(
            'ClipWindow',
            str(clip_window)
        )

    dst.FlushCache()

    # --------------------------------------------------
    # 9. Build overviews
    # --------------------------------------------------

    if cog:

        dst.BuildOverviews(
            "NEAREST",
            ovr
        )

    dst = None

    print(
        f"Saved file: {out_path} "
        # f"| Size: {out_width} x {out_height}"
    )

def cleanup_temp_files(temp_dir):
    for f in glob.glob(os.path.join(temp_dir, "*.tif")):
        os.remove(f)
    
    os.rmdir(temp_dir)

def h5_polsar(h5_file, dataset_paths, output_dir, temp_dir,
              azlks, rglks, matrix_type, apply_multilook, recip=False,
              chunk_size_x=512, chunk_size_y=512,
              max_workers=None,
              start_x=None, start_y=None, xres=1.0, yres=1.0, epsg=4326,
              fmt='tif', cog=False, ovr=[2, 4, 8, 16], comp=False,
              dtype=np.float32,
              inshape=None, outshape=None,
              calibration_constant=1,
              rtc_factor_path=None,
              clip_window=None,
              clip_mode='crop',
              clip=None,
              progress_callback=None):   # NEW PARAM
    if max_workers is None:
        max_workers = max(multiprocessing.cpu_count() - 1, 1)

    polarization_key = (
        'HH'   if 'HH'   in dataset_paths else
        'VV'   if 'VV'   in dataset_paths else
        'RH'   if 'RH'   in dataset_paths else
        'LH'   if 'LH'   in dataset_paths else
        'HHHH' if 'HHHH' in dataset_paths else
        'VVVV' if 'VVVV' in dataset_paths else
        'HVHV' if 'HVHV' in dataset_paths else
        'RHRH' if 'RHRH' in dataset_paths else
        None
    )
    # Validate RTC factor dimensions
    if rtc_factor_path is not None:

        with tables.open_file(h5_file, 'r') as h5:

            factor_node = h5.get_node(rtc_factor_path)
            source_node = h5.get_node(dataset_paths[polarization_key])

            if factor_node.shape != source_node.shape:
                raise ValueError(
                    f"RTC factor shape {factor_node.shape} "
                    f"does not match GCOV shape {source_node.shape}"
                )


    jobs = get_chunk_jobs(h5_file, dataset_paths[polarization_key], chunk_size_x, chunk_size_y,clip_window=clip_window)

    total_tasks = len(jobs)
    completed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(process_and_write_tile, job, h5_file, dataset_paths,
                               azlks, rglks, matrix_type, apply_multilook, recip, temp_dir,
                               start_x=start_x, start_y=start_y, xres=xres, yres=yres, epsg=epsg,
                               calibration_constant=calibration_constant, rtc_factor_path=rtc_factor_path) for job in jobs]

        use_tqdm = progress_callback is None
        pbar = tqdm(total=total_tasks, desc="Processing chunks") if use_tqdm else None

        for _ in as_completed(futures):
            completed += 1
            if progress_callback:
                progress_callback(completed / total_tasks)
            elif use_tqdm:
                pbar.update(1)

        if pbar:
            pbar.close()

    # Discover output band names
    dummy_shape = (azlks * 2, rglks * 2)
    dummy_data = {k: np.zeros(dummy_shape, dtype=np.complex64) for k in dataset_paths.keys()}
    keys = compute_elements(dummy_data, matrix_type, azlks, rglks,
                            apply_multilook, recip, calibration_constant).keys()

    # for name in keys:
    #     mosaic_chunks(name, temp_dir, output_dir, chunk_size_x, chunk_size_y,
    #                   azlks, rglks, apply_multilook,
    #                   start_x, start_y, xres, yres, epsg,
    #                   fmt, cog, ovr, comp, dtype, inshape, outshape,
    #                   clip_window=clip_window)

    # cleanup_temp_files(temp_dir)
    for name in keys:

        mosaic_chunks(
            name, temp_dir, output_dir,
            chunk_size_x, chunk_size_y,
            azlks, rglks, apply_multilook,
            start_x, start_y, xres, yres, epsg,
            fmt, cog, ovr, comp, dtype,
            inshape, outshape,
            clip_window=clip_window
        )

        # Apply polygon mask after mosaicking
        if isinstance(clip, (str, os.PathLike)):

            output_path = os.path.join(
                output_dir,
                f"{name}.{fmt}"
            )

            apply_vector_mask(
                output_path,
                clip,
                nodata=0
            )

    cleanup_temp_files(temp_dir)
