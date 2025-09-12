import os
import numpy as np
from polsartools.utils.proc_utils import process_chunks_parallel
from polsartools.utils.utils import time_it, mlook_arr
from polsartools.preprocess.pre_utils import get_filter_io_paths
from osgeo import gdal
gdal.UseExceptions()

@time_it
def mlook(in_dir,  azlks=2, rglks=2, fmt="tif",sub_dir=True, 
          cog=False, ovr = [2, 4, 8, 16], comp=False,
          max_workers=None,block_size=(512, 512),
          progress_callback=None,  # for QGIS plugin
        ):
    
    """
    Generate multilooked polarimetric matrix from C4, T4, C3, T3, C2, or T2 formats.

    This function applies multilooking (spatial averaging) to reduce speckle noise 
    and improve radiometric stability in polarimetric SAR datasets.

    Examples
    --------
    >>> # Basic usage with default look factors
    >>> mlook("/path/to/polSAR_data")

    >>> # With 3 azimuth and 2 range looks, and COG GeoTIFF output
    >>> mlook("/path/to/polSAR_data", azlks=3, rglks=2, fmt="tif", cog=True)

    Parameters
    ----------
    in_dir : str
        Path to the input folder containing a supported polarimetric matrix.
    azlks : int, default=2
        Number of looks in azimuth (vertical) direction.
    rglks : int, default=2
        Number of looks in range (horizontal) direction.
    fmt : {'tif', 'bin'}, default='tif'
        Output format:
        - 'tif': Cloud-optimized GeoTIFF (if cog_flag is True)
        - 'bin': Raw binary format
    cog : bool, default=False
        Enable Cloud Optimized GeoTIFF output with internal overviews and tiling.
    ovr : list[int], default=[2, 4, 8, 16]
        Overview levels for pyramid generation (used with COGs).
    comp : bool, default=False
        If True, uses LZW compression for GeoTIFF outputs.
    max_workers : int | None, default=None
        Maximum number of parallel worker threads (defaults to all available CPUs).
    block_size : tuple[int, int], default=(512, 512)
        Size of processing blocks for chunked and parallel execution.

    Returns
    -------
    None
        The multilooked output matrix is saved to disk. Number and names of output
        files depend on matrix type (e.g., C3 → C11.tif, C12_real.tif, etc.).

    Notes
    -----
    - Supported polarimetric matrices: 'C4', 'T4', 'C3', 'T3', 'C2', 'T2'
    - Automatically detects matrix type and file extension (.bin or .tif)
    - Handles real and complex-valued data appropriately (e.g., multilooks real and imag separately)
    - Output pixel spacing is updated in geotransform metadata based on look factors
    """
    write_flag=True
    if azlks <= 0 or rglks <= 0:
        raise ValueError("azlks and rglks must be positive integers")
  
    input_filepaths, output_filepaths = get_filter_io_paths(in_dir, [azlks, rglks],
                                                            fmt=fmt, 
                                                            filter_type="ml",sub_dir=sub_dir)

    window_size=None

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
        out_geotransform[1] *= rglks 
        out_geotransform[5] *= azlks
        
    out_geotransform = tuple(out_geotransform)  # back to immutable
    # print(in_geotransform,out_geotransform)
    dataset = None  # Close GDAL dataset
    


    def closest_multiple(value, base):
        return round(value / base) * base

    # Calculate closest multiples of rglks and azlks to 512
    block_x_size = closest_multiple(block_size[0] , rglks)
    block_y_size = closest_multiple(block_size[1] , azlks)

    # print(block_x_size,block_y_size)
    block_size = (block_x_size, block_y_size)


    process_chunks_parallel(input_filepaths, list(output_filepaths), 
                             window_size,
                            write_flag,
                            process_chunk_mlook,
                            *[azlks, rglks],
                            block_size=block_size, 
                            max_workers=max_workers,  
                            num_outputs=len(output_filepaths),
                            cog=cog, ovr=ovr, comp=comp,
                            out_x_size=out_x_size,
                            out_y_size=out_y_size,
                            out_geotransform=out_geotransform,
                            out_projection=in_projection,
                            azlks=azlks,
                            rglks=rglks,
                            progress_callback=progress_callback
                            )

def process_chunk_mlook(chunks, window_size, *args, **kwargs):
    azlks=args[-2]
    rglks=args[-1]
    # print("mlook",azlks,rglks)
    mlook_chunks = []

    for chunk in chunks:
        arr = np.array(chunk)
        if np.iscomplexobj(arr):
            real_part = mlook_arr(arr.real, azlks, rglks)
            imag_part = mlook_arr(arr.imag, azlks, rglks)
            processed = (real_part + 1j * imag_part).astype(np.complex64)
        else:
            processed = mlook_arr(arr, azlks, rglks).astype(np.float32)
        mlook_chunks.append(processed)
        
    return mlook_chunks