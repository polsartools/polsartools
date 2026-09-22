

from polsartools.utils.utils import time_it
from polsartools.utils.io_utils import write_T3, write_C3,mlook_arr,read_bin
from polsartools.utils.proc_utils import process_chunks_parallel
import numpy as np
import os
from osgeo import gdal
from polsartools.utils.convert_matrices import C3_T3_mat
gdal.UseExceptions()

def find_file(base, in_dir):
    for ext in [".bin", ".tif"]:
        path = os.path.join(in_dir, f"{base}{ext}")
        if os.path.isfile(path):
            return path
    return None

def get_c_input_filepaths(in_dir):
    keys = ["C11", "C12_real", "C12_imag", "C13_real", "C13_imag", "C14_real", "C14_imag",
             "C22", "C23_real", "C23_imag", "C24_real", "C24_imag",
             "C33", "C34_real", "C34_imag", "C44"]
    found_files = {k: find_file(k, in_dir) for k in keys}
    available = [v for v in found_files.values() if v is not None]
    if len(available) in [4, 9, 16]:
        return available
    else:
        raise FileNotFoundError(f"Only found {len(available)} C-matrix files; need C2 (4), C3 (9), or C4 (16).")
def get_in_matrix_type(in_dir):
    """
    Determines the input matrix type based on the number of covariance component files.
    Returns 'C2', 'C3', or 'C4'.
    """
    input_filepaths = get_c_input_filepaths(in_dir)
    num_files = len(input_filepaths)
    if num_files == 4:
        return "C2"
    elif num_files == 9:
        return "C3"
    elif num_files == 16:
        return "C4"
    else:
        raise ValueError(f"Invalid number of covariance component files: {num_files}. Expected 4, 9, or 16.")



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
        "T4": ["T11", "T12_real", "T12_imag", "T13_real", "T13_imag",
               "T14_real", "T14_imag", "T22", "T23_real", "T23_imag",
               "T24_real", "T24_imag", "T33", "T34_real", "T34_imag", "T44"],
        "C2HX": ["C11", "C12_real", "C12_imag", "C22"],
        "C2VX": ["C11", "C12_real", "C12_imag", "C22"],
        "C2HV": ["C11", "C12_real", "C12_imag", "C22"],
        "C2RH": ["C11", "C12_real", "C12_imag", "C22"],
        "C2LH": ["C11", "C12_real", "C12_imag", "C22"],
        "C2RR": ["C11", "C12_real", "C12_imag", "C22"],
        "C2LL": ["C11", "C12_real", "C12_imag", "C22"],
        "T2HV": ["T11", "T12_real", "T12_imag", "T22",],
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
def convert_C(in_dir, mat='T3', azlks=1,rglks=1,  
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
    >>> # Convert C3 to T3 matrix
    >>> convert_C("/path/to/C_data", mat="T3")

    >>> # Output as tiled GeoTIFF with Cloud Optimized overviews
    >>> convert_C("/data/C_data", mat="C2", cog=True)

    Parameters
    ----------
    in_dir : str
        Path to the input folder containing covariance components 
        {C3: C11, C12_real, C12_imag, C13_real, C13_imag, C22, C13_real, C23_imag, C33; 
         C4: C11, C12_real, C12_imag, C13_real, C13_imag, C14_real, C14_imag, C22, C23_real, C23_imag, C24_real, C24_imag, C33, C34_real, C34_imag,}.
    mat : str, default='T3'
        Output matrix format. Supported values:
        - 'T4', 'T3', 'T2HV' (Coherency)
        - 'C3', 'C2HX', 'C2VX', 'C2HV' (Covariance)
    azlks : int, default=1
        Number of looks in azimuth direction.
    rglks : int, default=1
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
    
    input_filepaths =  get_c_input_filepaths(in_dir)
    output_filepaths = get_output_filepaths(in_dir, out_dir, mat, fmt)
    # print(f"Input files: {input_filepaths}")
    in_mat = get_in_matrix_type(in_dir)
    print(f"Detected {in_mat} matrix and converting to {mat} matrix")

    if len(input_filepaths) not in [4, 9,16]:
        raise Exception("Invalid C folder: must contain either 4, 9, or 16 covariance component files")
    if len(input_filepaths) == 4 and mat not in {'C4', 'T4', 'C3', 'T3', 'C2HX', 'C2VX', 'C2HV', 'T2HV','M4'}:
        raise Exception(f"Invalid matrix type '{mat}' for full-pol input - please choose one of 'C4', 'T4', 'C3', 'T3', 'C2HX', 'C2VX', 'C2HV', 'T2HV','M4'")
    
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
    normalized = tuple(round(float(x), 6) for x in in_geotransform)
    if normalized in [
        (0.0, 1.0, 0.0, 0.0, 0.0, -1.0),
        (0.0, 1.0, 0.0, 0.0, 0.0,  1.0)
                    ]:
        out_geotransform[1] *= 1 
        out_geotransform[5] *= 1 
    else:
        out_geotransform[1] = (in_geotransform[1] * in_cols) / out_x_size
        out_geotransform[5] = (in_geotransform[5] * in_rows) / out_y_size
        
        
    out_geotransform = tuple(out_geotransform)
    dataset = None 
    


    def closest_multiple(value, base):
        return round(value / base) * base

    # Calculate closest multiples of rglks and azlks to 512
    block_x_size = closest_multiple(block_size[0] , rglks)
    block_y_size = closest_multiple(block_size[1] , azlks)

    # print(block_x_size,block_y_size)
    block_size = (block_x_size, block_y_size)


    # print("matrix:", mat)
    if len(input_filepaths) == 4:
        # print("Processing dual-pol S-matrix...")
        process_chunks_parallel(input_filepaths, list(output_filepaths), 
                             window_size,
                            write_flag,
                            process_chunk_c2,
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
    elif len(input_filepaths) == 9:
        # print("Processing full-pol S-matrix...")
        process_chunks_parallel(input_filepaths, list(output_filepaths), 
                                window_size,
                                write_flag,
                                process_chunk_c3,
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
    elif len(input_filepaths) == 16:
        # print("Processing full-pol C4-matrix...")
        process_chunks_parallel(input_filepaths, list(output_filepaths), 
                                window_size,
                                write_flag,
                                process_chunk_c4,
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

    else:
        raise Exception("Invalid C folder: must contain either 4, 9, or 16 covariance component files")



def process_chunk_c3(chunks, *args, **kwargs):
    # print(args[-1],args[-2],args[-3])
    abs_cf = args[-5]
    matrix=args[-4]
    recip = args[-3]
    azlks=args[-2]
    rglks=args[-1]

    if matrix=='T3':
        C11 = np.array(chunks[0])
        C12 = np.array(chunks[1])+1j*np.array(chunks[2])
        C13 = np.array(chunks[3])+1j*np.array(chunks[4])
        C21 = np.conj(C12)
        C22 = np.array(chunks[5])
        C23 = np.array(chunks[6])+1j*np.array(chunks[7])
        C31 = np.conj(C13)
        C32 = np.conj(C23)
        C33 = np.array(chunks[8])
        T_T1 = np.array([[C11, C12, C13], 
                        [C21, C22, C23], 
                        [C31, C32, C33]])
        T_T1 = C3_T3_mat(T_T1)
    else:
        raise Exception(f"Invalid C3 folder!!")

    # if window_size>1:
    #     kernel = np.ones((window_size,window_size),np.float32)/(window_size*window_size)

    #     t11f = conv2d(T_T1[0,0,:,:],kernel)
    #     t12f = conv2d(np.real(T_T1[0,1,:,:]),kernel)+1j*conv2d(np.imag(T_T1[0,1,:,:]),kernel)
    #     t13f = conv2d(np.real(T_T1[0,2,:,:]),kernel)+1j*conv2d(np.imag(T_T1[0,2,:,:]),kernel)
        
    #     t21f = np.conj(t12f) 
    #     t22f = conv2d(T_T1[1,1,:,:],kernel)
    #     t23f = conv2d(np.real(T_T1[1,2,:,:]),kernel)+1j*conv2d(np.imag(T_T1[1,2,:,:]),kernel)

    #     t31f = np.conj(t13f) 
    #     t32f = np.conj(t23f) 
    #     t33f = conv2d(T_T1[2,2,:,:],kernel)

    #     T_T1 = np.array([[t11f, t12f, t13f], [t21f, t22f, t23f], [t31f, t32f, t33f]])


    return T_T1[0,0,:,:].astype(np.float32), T_T1[0,1,:,:].real.astype(np.float32), T_T1[0,1,:,:].imag.astype(np.float32), T_T1[0,2,:,:].real.astype(np.float32), T_T1[0,2,:,:].imag.astype(np.float32), T_T1[1,1,:,:].astype(np.float32), T_T1[1,2,:,:].real.astype(np.float32), T_T1[1,2,:,:].imag.astype(np.float32), T_T1[2,2,:,:].astype(np.float32)
    
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

def process_chunk_c2(chunks, *args, **kwargs):
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


def process_chunk_c4(chunks, *args, **kwargs):
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
