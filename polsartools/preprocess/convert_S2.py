

from polsartools.utils.utils import time_it
from polsartools.utils.io_utils import write_T3, write_C3,mlook_arr,read_bin
from polsartools.utils.proc_utils import process_chunks_parallel
import numpy as np
import os
from osgeo import gdal
gdal.UseExceptions()

# def get_s2_input_filepaths(in_dir):
#     """
#     Searches for s11/s12/s21/s22 files in .bin or .tif format.
#     Returns input file paths or raises an exception if files are missing.
#     """
#     def find_file(base):
#         for ext in [".bin", ".tif"]:
#             path = os.path.join(in_dir, f"{base}{ext}")
#             if os.path.isfile(path):
#                 return path
#         return None

#     keys = ["s11", "s12", "s21", "s22"]
#     input_filepaths = [find_file(k) for k in keys]

#     if not all(input_filepaths):
#         raise FileNotFoundError("Invalid S2 folder: missing s11/s12/s21/s22 files")
    
#     return input_filepaths

def find_file(base, in_dir):
    for ext in [".bin", ".tif"]:
        path = os.path.join(in_dir, f"{base}{ext}")
        if os.path.isfile(path):
            return path
    return None

# def get_s_input_filepaths(in_dir):
#     keys = ["s11", "s12", "s21", "s22"]
#     found_files = {k: find_file(k, in_dir) for k in keys}
#     available = {k: v for k, v in found_files.items() if v is not None}

#     if len(available) in [2, 4]:
#         print(f"Found valid S-matrix set with {len(available)} files:")
#         for k, path in available.items():
#             print(f"  {k}: {path}")
#         return available
#     else:
#         raise FileNotFoundError

def get_s_input_filepaths(in_dir):
    keys = ["s11", "s12", "s21", "s22"]
    found_files = {k: find_file(k, in_dir) for k in keys}
    available = [v for v in found_files.values() if v is not None]
    if len(available) in [2, 4]:
        # print(f"Found valid S-matrix set with {len(available)} files:")
        # for path in available:
        #     print(f"  {path}")
        return available
    else:
        raise FileNotFoundError(f"Only found {len(available)} S-matrix files; need exactly 2 or 4.")


def get_output_filepaths(in_dir, out_dir, matrix, fmt):
    """
    Returns output filepaths for the specified matrix and output type (bin or tif).
    Also ensures the target directory exists.
    """
    matrix_keys = {
        "T3": ["T11", "T12_real", "T12_imag", "T13_real", "T13_imag",
               "T22", "T23_real", "T23_imag", "T33"],
        "C3": ["C11", "C12_real", "C12_imag", "C13_real", "C13_imag",
               "C22", "C23_real", "C23_imag", "C33"],
        "C4": ["C11", "C12_real", "C12_imag", "C13_real", "C13_imag",
               "C14_real", "C14_imag", "C22", "C23_real", "C23_imag",
               "C24_real", "C24_imag", "C33", "C34_real", "C34_imag", "C44"],
        "T4": ["T11", "T12_real", "T12_imag", "T13_real", "T13_imag",
               "T14_real", "T14_imag", "T22", "T23_real", "T23_imag",
               "T24_real", "T24_imag", "T33", "T34_real", "T34_imag", "T44"],
        "C2HX": ["C11", "C12_real", "C12_imag", "C22"],
        "C2VX": ["C11", "C12_real", "C12_imag", "C22"],
        "C2HV": ["C11", "C12_real", "C12_imag", "C22"],
        "T2HV": ["T11", "T12_real", "T12_imag", "T22",],
        "C2": ["C11", "C12_real", "C12_imag", "C22"],
        "T2": ["T11", "T12_real", "T12_imag", "T22"]
    }

    if matrix not in matrix_keys:
        raise ValueError(f"Invalid matrix type '{matrix}'")

    ext = ".bin" if fmt == "bin" else ".tif"
    if out_dir is None:
        out_dir = os.path.join(in_dir, matrix)
    os.makedirs(out_dir, exist_ok=True)

    return [os.path.join(out_dir, f"{name}{ext}") for name in matrix_keys[matrix]]

@time_it
def convert_S(in_dir, mat='T3', azlks=4,rglks=2,  
                  fmt="tif", cog=False,ovr = [2, 4, 8, 16],comp=False,
                  recip=True,
                  cf = 1, out_dir=None,
                  max_workers=None,block_size=(512, 512),
                  progress_callback=None,  # for QGIS plugin
                  ):
    """
    Convert full/dual-polarimetric scattering (S2,Sxy) matrix into multi-looked
    coherency (T4, T3, T2) or covariance (C4, C3, C2) matrices.
    It supports both GeoTIFF and PolSARpro-compatible output.

    Examples
    --------
    >>> # Convert to C3 matrix with 10x5 multi-looking
    >>> convert_S("/path/to/S_data", mat="C3", azlks=10, rglks=5)

    >>> # Output as tiled GeoTIFF with Cloud Optimized overviews
    >>> convert_S("/data/S_data", mat="C2", cog=True)

    Parameters
    ----------
    in_dir : str
        Path to the input folder containing S11, S12, S21, S22 scattering components.
    mat : str, default='T3'
        Output matrix format. Supported values:
        - 'T4', 'T3', 'T2HV' (Coherency)
        - 'C4', 'C3', 'C2HX', 'C2VX', 'C2HV' (Covariance)
    azlks : int, default=4
        Number of looks in azimuth direction.
    rglks : int, default=2
        Number of looks in range direction.
    fmt : {'tif', 'bin'}, default='tif'
        Output format type.
    cog : bool, default=False
        If True, creates Cloud Optimized GeoTIFF (COG).
    ovr : list[int], default=[2, 4, 8, 16]
        Levels of pyramid overviews for COG generation.
    comp : bool, default=False
        If True, applies LZW compression to output GeoTIFF files.
    recip : bool, default=True
        If True, scattering matrix reciprocal symmetry is applied, i.e, S_HV = S_VH.
    cf : float, default=1
        Calibration factor (linear) to adjust the amplitude of S2 data.
    out_dir : str | None, default=None
        Path to the output folder. If None, uses the input folder.
    max_workers : int | None, default=None
        Number of parallel worker threads.
    block_size : tuple[int, int], default=(512, 512)
        Size of chunks for processing.
    """
    
    window_size=None
    write_flag=True
    
    input_filepaths =  get_s_input_filepaths(in_dir)
    output_filepaths = get_output_filepaths(in_dir, out_dir,mat, fmt)
    
    if len(input_filepaths) not in [2, 4]:
        raise Exception("Invalid S folder: must contain either 2 (dual/compact-pol) or 4 (full-pol) S-matrix files")
    if len(input_filepaths) == 2 and mat not in {'C2', 'T2'}:
        raise Exception(f"Invalid matrix type '{mat}' for dual-pol input - please choose one of 'C2', 'T2'")
    if len(input_filepaths) == 4 and mat not in {'C4', 'T4', 'C3', 'T3', 'C2HX', 'C2VX', 'C2HV', 'T2HV'}:
        raise Exception(f"Invalid matrix type '{mat}' for full-pol input - please choose one of 'C4', 'T4', 'C3', 'T3', 'C2HX', 'C2VX', 'C2HV', 'T2HV',")
    
    # VALID_MATRICES = {'C4', 'T4', 'C3', 'T3', 'C2HX', 'C2VX', 'C2HV', 'T2HV', 'C2', 'T2'}
    # if matrix not in VALID_MATRICES:
    #     raise Exception(f"Invalid matrix type '{matrix}' - please choose one of {', '.join(VALID_MATRICES)}")
    
    
    
        
    """
    GET MULTI-LOOKED RASTER PROPERTIES
       
    """
    
    dataset = gdal.Open(input_filepaths[0], gdal.GA_ReadOnly)
    if dataset is None:
        raise FileNotFoundError(f"Cannot open {input_filepaths[0]}")

    in_cols = dataset.RasterXSize
    in_rows = dataset.RasterYSize
    in_geotransform = dataset.GetGeoTransform()
    in_projection = dataset.GetProjection()

    # Calculate output size after multilooking
    out_x_size = in_cols // rglks
    out_y_size = in_rows // azlks

    # Calculate new geotransform with updated pixel size
    out_geotransform = list(in_geotransform)
    
    if in_geotransform[0]==0.0 and in_geotransform[1]==1.0 and in_geotransform[2]==-0.0 and in_geotransform[3]==0.0 and in_geotransform[4]==-0.0 and in_geotransform[5]==-1.0:
        out_geotransform[1] *= 1 
        out_geotransform[5] *= 1 
    else:
        out_geotransform[1] = (in_geotransform[1] * in_cols) / out_x_size
        out_geotransform[5] = (in_geotransform[5] * in_rows) / out_y_size
        
        # out_geotransform[1] *= rglks 
        # out_geotransform[5] *= azlks
        
    out_geotransform = tuple(out_geotransform)
    dataset = None 
    


    def closest_multiple(value, base):
        return round(value / base) * base

    # Calculate closest multiples of rglks and azlks to 512
    block_x_size = closest_multiple(block_size[0] , rglks)
    block_y_size = closest_multiple(block_size[1] , azlks)

    # print(block_x_size,block_y_size)
    block_size = (block_x_size, block_y_size)
    
    if len(input_filepaths) == 2:
        # print("Processing dual-pol S-matrix...")
        process_chunks_parallel(input_filepaths, list(output_filepaths), 
                             window_size,
                            write_flag,
                            process_chunk_sxyct,
                            *[cf, mat, azlks, rglks],
                            block_size=block_size, 
                            max_workers=max_workers,  
                            num_outputs=len(output_filepaths),
                            cog=cog,
                            ovr=ovr,
                            comp=comp,
                            out_x_size=out_x_size,
                            out_y_size=out_y_size,
                            out_geotransform=out_geotransform,
                            out_projection=in_projection,
                            azlks=azlks,
                            rglks=rglks,
                            progress_callback=progress_callback
                            )
    elif len(input_filepaths) == 4:
        # print("Processing full-pol S-matrix...")
        process_chunks_parallel(input_filepaths, list(output_filepaths), 
                                window_size,
                                write_flag,
                                process_chunk_s2ct,
                                *[cf, mat, recip, azlks, rglks],
                                block_size=block_size, 
                                max_workers=max_workers,  
                                num_outputs=len(output_filepaths),
                                cog=cog,
                                ovr=ovr,
                                comp=comp,
                                out_x_size=out_x_size,
                                out_y_size=out_y_size,
                                out_geotransform=out_geotransform,
                                out_projection=in_projection,
                                azlks=azlks,
                                rglks=rglks,
                                progress_callback=progress_callback
                                )






def process_chunk_sxyct(chunks, *args, **kwargs):
    # print(args[-1],args[-2],args[-3])
    abs_cf = args[-4]
    matrix=args[-3]
    azlks=args[-2]
    rglks=args[-1]
    
    s11 = np.array(chunks[0])*abs_cf
    s12 = np.array(chunks[1])*abs_cf

    if matrix=='C2':
        C11 = mlook_arr(np.abs(s11)**2,azlks,rglks).astype(np.float32)
        C22 = mlook_arr(np.abs(s12)**2,azlks,rglks).astype(np.float32)    
        C12 = mlook_arr(s11*np.conjugate(s12),azlks,rglks).astype(np.complex64)
        
        return np.real(C11),np.real(C12),np.imag(C12),np.real(C22)

    elif matrix=='T2':
        C11 = mlook_arr(np.abs(s11+s12)**2,azlks,rglks).astype(np.float32)
        C22 = mlook_arr(np.abs(s11-s12)**2,azlks,rglks).astype(np.float32)    
        C12 = mlook_arr((s11+s12)*np.conjugate(s11-s12),azlks,rglks).astype(np.complex64)
        
        return np.real(C11),np.real(C12),np.imag(C12),np.real(C22)         
    
    else:
        raise('Invalid matrix type !!')

def process_chunk_s2ct(chunks, *args, **kwargs):
    # print(args[-1],args[-2],args[-3])
    abs_cf = args[-5]
    matrix =args[-4]
    recip=args[-3]
    azlks=args[-2]
    rglks=args[-1]

    
    s11 = np.array(chunks[0])*abs_cf
    s12 = np.array(chunks[1])*abs_cf
    s21 = np.array(chunks[2])*abs_cf
    s22 = np.array(chunks[3])*abs_cf
    
    if recip:
        # print("Applying reciprocal symmetry, i.e, S_HV = S_VH")
        s12 = (s12 + s21) / 2
        s21 = s12.copy()
    
    if matrix == 'C4':
        Kl = np.array([s11, s12, s21, s22])

        del s11,s12,s21,s22
        
        C11 = mlook_arr(np.abs(Kl[0])**2,azlks,rglks).astype(np.float32)
        C22 = mlook_arr(np.abs(Kl[1])**2,azlks,rglks).astype(np.float32)
        C33 = mlook_arr(np.abs(Kl[2])**2,azlks,rglks).astype(np.float32)
        C44 = mlook_arr(np.abs(Kl[3])**2,azlks,rglks).astype(np.float32)

        C12 = mlook_arr(Kl[0]*np.conj(Kl[1]),azlks,rglks).astype(np.complex64)
        C13 = mlook_arr(Kl[0]*np.conj(Kl[2]),azlks,rglks).astype(np.complex64)
        C14 = mlook_arr(Kl[0]*np.conj(Kl[3]),azlks,rglks).astype(np.complex64)
        C23 = mlook_arr(Kl[1]*np.conj(Kl[2]),azlks,rglks).astype(np.complex64)
        C24 = mlook_arr(Kl[1]*np.conj(Kl[3]),azlks,rglks).astype(np.complex64)
        C34 = mlook_arr(Kl[2]*np.conj(Kl[3]),azlks,rglks).astype(np.complex64)
        
        del Kl
        return np.real(C11),np.real(C12),np.imag(C12),np.real(C13),np.imag(C13),np.real(C14),np.imag(C14), np.real(C22),np.real(C23),np.imag(C23),np.real(C24),np.imag(C24), np.real(C33),np.real(C34),np.imag(C34), np.real(C44)
        
        
    elif matrix == "T4":
        Kp = (1/np.sqrt(2))*np.array([s11+s22, s11-s22, s12+s21, 1j*(s12-s21)])

        del s11,s12,s21,s22

        # 4x4 Pauli Coherency Matrix elements
        T11 = mlook_arr(np.abs(Kp[0])**2,azlks,rglks).astype(np.float32)
        T22 = mlook_arr(np.abs(Kp[1])**2,azlks,rglks).astype(np.float32)
        T33 = mlook_arr(np.abs(Kp[2])**2,azlks,rglks).astype(np.float32)
        T44 = mlook_arr(np.abs(Kp[3])**2,azlks,rglks).astype(np.float32)

        T12 = mlook_arr(Kp[0]*np.conj(Kp[1]),azlks,rglks).astype(np.complex64)
        T13 = mlook_arr(Kp[0]*np.conj(Kp[2]),azlks,rglks).astype(np.complex64)
        T14 = mlook_arr(Kp[0]*np.conj(Kp[3]),azlks,rglks).astype(np.complex64)
        T23 = mlook_arr(Kp[1]*np.conj(Kp[2]),azlks,rglks).astype(np.complex64)
        T24 = mlook_arr(Kp[1]*np.conj(Kp[3]),azlks,rglks).astype(np.complex64)
        T34 = mlook_arr(Kp[2]*np.conj(Kp[3]),azlks,rglks).astype(np.complex64)  

        del Kp
        return np.real(T11),np.real(T12),np.imag(T12),np.real(T13),np.imag(T13), np.real(T14),np.imag(T14),np.real(T22),np.real(T23),np.imag(T23), np.real(T24),np.imag(T24),np.real(T33),np.real(T34),np.imag(T34),np.real(T44)    


    elif matrix == "T3":
        Kp = (1/np.sqrt(2))*np.array([s11+s22, s11-s22, s12+s21])

        del s11,s12,s21,s22

        # 3x3 Pauli Coherency Matrix elements
        T11 = mlook_arr(np.abs(Kp[0])**2,azlks,rglks).astype(np.float32)
        T22 = mlook_arr(np.abs(Kp[1])**2,azlks,rglks).astype(np.float32)
        T33 = mlook_arr(np.abs(Kp[2])**2,azlks,rglks).astype(np.float32)

        T12 = mlook_arr(Kp[0]*np.conj(Kp[1]),azlks,rglks).astype(np.complex64)
        T13 = mlook_arr(Kp[0]*np.conj(Kp[2]),azlks,rglks).astype(np.complex64)
        T23 = mlook_arr(Kp[1]*np.conj(Kp[2]),azlks,rglks).astype(np.complex64)

        del Kp
        return np.real(T11),np.real(T12),np.imag(T12),np.real(T13),np.imag(T13),np.real(T22),np.real(T23),np.imag(T23),np.real(T33)

        
    elif matrix=='C3':
        # Kl- 3-D Lexicographic feature vector
        Kl = np.array([s11, np.sqrt(2)*0.5*(s12+s21), s22])
        del s11,s12,s22

        # 3x3 COVARIANCE Matrix elements

        C11 = mlook_arr(np.abs(Kl[0])**2,azlks,rglks).astype(np.float32)
        C22 = mlook_arr(np.abs(Kl[1])**2,azlks,rglks).astype(np.float32)
        C33 = mlook_arr(np.abs(Kl[2])**2,azlks,rglks).astype(np.float32)

        C12 = mlook_arr(Kl[0]*np.conj(Kl[1]),azlks,rglks).astype(np.complex64)
        C13 = mlook_arr(Kl[0]*np.conj(Kl[2]),azlks,rglks).astype(np.complex64)
        C23 = mlook_arr(Kl[1]*np.conj(Kl[2]),azlks,rglks).astype(np.complex64)
        return np.real(C11),np.real(C12),np.imag(C12),np.real(C13),np.imag(C13),np.real(C22),np.real(C23),np.imag(C23),np.real(C33)

    elif matrix=='C2HX':
        C11 = mlook_arr(np.abs(s11)**2,azlks,rglks).astype(np.float32)
        C22 = mlook_arr(np.abs(s12)**2,azlks,rglks).astype(np.float32)    
        C12 = mlook_arr(s11*np.conjugate(s12),azlks,rglks).astype(np.complex64)
        
        return np.real(C11),np.real(C12),np.imag(C12),np.real(C22)
    elif matrix=='C2VX':
        C11 = mlook_arr(np.abs(s22)**2,azlks,rglks).astype(np.float32)
        C22 = mlook_arr(np.abs(s21)**2,azlks,rglks).astype(np.float32)    
        C12 = mlook_arr(s22*np.conjugate(s21),azlks,rglks).astype(np.complex64)
        
        return np.real(C11),np.real(C12),np.imag(C12),np.real(C22)
    elif matrix=='C2HV':
        C11 = mlook_arr(np.abs(s11)**2,azlks,rglks).astype(np.float32)
        C22 = mlook_arr(np.abs(s22)**2,azlks,rglks).astype(np.float32)    
        C12 = mlook_arr(s11*np.conjugate(s22),azlks,rglks).astype(np.complex64)
        
        return np.real(C11),np.real(C12),np.imag(C12),np.real(C22)      
        
    elif matrix=='T2HV':
        C11 = mlook_arr(np.abs(s11+s22)**2,azlks,rglks).astype(np.float32)
        C22 = mlook_arr(np.abs(s11-s22)**2,azlks,rglks).astype(np.float32)    
        C12 = mlook_arr((s11+s22)*np.conjugate(s11-s22),azlks,rglks).astype(np.complex64)
        
        return np.real(C11),np.real(C12),np.imag(C12),np.real(C22)         
    
    else:
        raise('Invalid matrix type !!')