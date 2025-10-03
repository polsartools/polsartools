import os
import numpy as np
from polsartools.utils.proc_utils import process_chunks_parallel
from polsartools.utils.utils import conv2d,time_it
from polsartools.preprocess.pre_utils import get_filter_io_paths

from polsartools.preprocess.rflee_filter import process_chunk_refined_lee
from polsartools.rflee import process_chunk_rfleecpp
@time_it
def boxcar(in_dir,  win=3, fmt="tif", sub_dir=True,
           cog=False, ovr = [2, 4, 8, 16], comp=False,
           max_workers=None,block_size=(512, 512),
           progress_callback=None,  # for QGIS plugin
           ):
    """
    Apply a Boxcar speckle filter to polarimetric SAR data to reduce speckle noise.

    The Boxcar filter performs a uniform mean filter over a square window, effectively 
    smoothing the polarimetric matrix to enhance interpretability. It is a fast, 
    straightforward speckle reduction technique suitable for preprocessing and 
    visualization.

    Examples
    --------
    >>> # Basic usage
    >>> boxcar("/path/to/polSAR_data")

    >>> # With custom window size and output as GeoTIFF
    >>> boxcar("/path/to/polSAR_data", win=5, cog=True)

    Parameters
    ----------
    in_dir : str
        Input folder containing C4, T4, C3, T3, C2, or T2 matrix files (.bin or .tif).
    win : int, default=3
        Size of the spatial smoothing window (e.g., 3x3).
    fmt : {'tif', 'bin'}, default='tif'
        Output file format type.
    cog : bool, default=False
        If True, generates a Cloud Optimized GeoTIFF.
    ovr : list[int], default=[2, 4, 8, 16]
        Overview levels for COG pyramids.
    sub_dir : bool, default=True
        If True, creates a subdirectory for the output files based on filter type and window size in the input folder. Else saves to a output folder at same level as input folder.
    comp : bool, default=False
        If True, applies LZW compression to the output GeoTIFF files.
    max_workers : int | None, default=None
        Maximum number of parallel workers.
    block_size : tuple[int, int], default=(512, 512)
        Block size for chunked processing.

    Returns
    -------
    None
        Writes filtered output matrix files to disk.
    """
    
    write_flag = True  # Always write output files
    input_filepaths, output_filepaths = get_filter_io_paths(in_dir, 
                                                            [win, win], 
                                                            fmt=fmt, 
                                                            filter_type="boxcar", 
                                                            sub_dir=sub_dir)

    # Process chunks in parallel
    num_outputs = len(output_filepaths)

    process_chunks_parallel(input_filepaths, list(output_filepaths), window_size=win, write_flag=write_flag,
                            processing_func=process_chunk_boxcar,block_size=block_size, max_workers=max_workers,  num_outputs=num_outputs,
                            cog=cog,
                            ovr=ovr,comp=comp,
                            progress_callback=progress_callback
                            )

def process_chunk_boxcar(chunks, window_size, *args):

    filtered_chunks = []
    for i in range(len(chunks)):
        img = np.array(chunks[i])
        kernel = np.ones((window_size, window_size), np.float32) / (window_size * window_size)
        filtered_chunks.append(conv2d(img, kernel))
        
    filtered_chunks = np.array(filtered_chunks).astype(np.float32)
    return filtered_chunks


@time_it
def rlee(in_dir,  win=3, fmt="tif",sub_dir=True, 
         cog=False, ovr = [2, 4, 8, 16], comp=False,
         max_workers=None,block_size=(512, 512),
         progress_callback=None,  # for QGIS plugin
         ):

    """
    Apply Refined Lee (RLee) speckle filter to polarimetric SAR data.

    The Refined Lee filter is an adaptive speckle filter that preserves edges and 
    structural details while reducing noise. Unlike Boxcar, RLee dynamically adjusts 
    based on local statistics, making it more suitable for classification, 
    segmentation, and biophysical retrieval tasks.

    Examples
    --------
    >>> # Basic usage with default parameters
    >>> rlee("/path/to/polSAR_data")

    >>> # Custom usage with large window and tiled COG output
    >>> rlee("/path/to/polSAR_data", win=7, cog=True)

    Parameters
    ----------
    in_dir : str
        Path to the folder with C4, T4, C3, T3, C2, or T2 matrix files.
    win : int, default=3
        Size of the adaptive filtering window.
    fmt : {'tif', 'bin'}, default='tif'
        Desired output file format.
    sub_dir : bool, default=True
        If True, creates a subdirectory for the output files based on filter type and window size in the input folder. Else saves to a output folder at same level as input folder.
    cog : bool, default=False
        Create Cloud Optimized GeoTIFF with overviews and internal tiling.
    ovr : list[int], default=[2, 4, 8, 16]
        Overview pyramid levels for zoomable image access.
    comp : bool, default=False
        If True, applies LZW compression to the output GeoTIFF files.
    max_workers : int | None, default=None
        Number of threads for parallel processing.
    block_size : tuple[int, int], default=(512, 512)
        Block size used during chunked filtering.

    Returns
    -------
    None
        Output files are written to disk with the applied RLee filter.
    """
    write_flag=True
    input_filepaths, output_filepaths = get_filter_io_paths(in_dir, 
                                                            [win, win], 
                                                            fmt=fmt, 
                                                            filter_type="rlee",
                                                            sub_dir=sub_dir)
    num_outputs = len(output_filepaths)
    
    ### Python implementation
    # process_chunks_parallel(input_filepaths, output_filepaths, win=win, write_flag=write_flag,
    #                         processing_func=process_chunk_refined_lee, block_size=(512, 512), max_workers=max_workers,
    #                         num_outputs=num_outputs)
    
    #### Uncomment below to use C++ implementation 

    process_chunks_parallel(input_filepaths, list(output_filepaths), window_size=win, write_flag=write_flag,
                            processing_func=process_chunk_rfl,block_size=block_size, max_workers=max_workers,  num_outputs=num_outputs,
                            cog=cog,
                            ovr=ovr,
                            progress_callback=progress_callback
                            )


def process_chunk_rfl(chunks, window_size, *args):

    # print('before pad',np.shape(chunks[0]))
    
    for i in range(len(chunks)):
        pad_top_left = window_size // 2 
        pad_bottom_right = window_size // 2 +1
        chunks[i] = np.pad(chunks[i], 
                            ((pad_top_left, pad_bottom_right), 
                            (pad_top_left, pad_bottom_right)), 
                            mode='constant', constant_values=0)
    
    
    # print("after pad",np.shape(chunks[0]))
    chunk_arrays = [np.array(ch) for ch in chunks]  
    vi_c_raw = process_chunk_rfleecpp(chunk_arrays, window_size)
    
    proc_chunks=[]
    for chunk in vi_c_raw:
        filt_data = np.array(chunk)
        # filt_data[filt_data == 0] = np.nan
        proc_chunks.append(filt_data)
        # print('mean %0.3f'%np.nanmean(np.array(chunk)),'std %0.3f'%np.nanstd(np.array(chunk)))
        
        
    del vi_c_raw,filt_data,chunk
    
    # print("proc_chunks pad",np.shape(proc_chunks[0]))
    for i in range(len(proc_chunks)):
        # Calculate the padding size
        pad_top_left = window_size // 2 
        pad_bottom_right = window_size // 2 +1
        proc_chunks[i] = proc_chunks[i][pad_top_left:-pad_bottom_right, pad_top_left:-pad_bottom_right]
    
    
    def shift_array(arr,shift):
        # Get the number of rows and columns in the array
        rows, cols = arr.shape

        # Step 1: Move the rightmost 3 columns to the left
        right_columns = arr[:, -shift:]  # Last 3 columns
        remaining_columns = arr[:, :-shift]  # All but last 3 columns

        # Step 2: Move the bottom 3 rows to the top
        bottom_rows = arr[-shift:, :]  # Last 3 rows
        remaining_rows = arr[:-shift, :]  # All but last 3 rows

        # Combine the shifted rows and columns
        shifted_array = np.vstack((bottom_rows, remaining_rows))  # Stack bottom rows to the top
        shifted_array = np.hstack((right_columns, shifted_array[:, :-shift]))  # Stack right columns to the left

        return shifted_array
    
    
    for i in range(len(proc_chunks)):
        # Remove 'window_size' rows from the top and 'window_size' columns from the left
        proc_chunks[i] = shift_array(proc_chunks[i],window_size//2)
    
    
    # print("proc_chunks unpad",np.shape(proc_chunks[0]))

    
    num_chunks = len(proc_chunks) // 2
    out_chunks = []

    for i in range(num_chunks):
        real_part = proc_chunks[2 * i]       # Get the real part from the even indices
        # real_part[real_part == 0] = np.nan
        imag_part = proc_chunks[2 * i + 1]   # Get the imaginary part from the odd indices
        # imag_part[imag_part == 0] = np.nan
        complex_array = real_part + 1j * imag_part  # Create a complex number
        
        out_chunks.append(complex_array)
        # print(np.nanmean(real_part),' ' ,np.nanmean(imag_part))
    filtered_chunks = []

    if len(chunks)==9:
        # print("out_chunks shape:", np.shape(out_chunks))
        filtered_chunks.append(np.real(out_chunks[0]))
        filtered_chunks.append(np.real(out_chunks[1]))
        filtered_chunks.append(np.imag(out_chunks[1]))
        filtered_chunks.append(np.real(out_chunks[2]))
        filtered_chunks.append(np.imag(out_chunks[2]))
        filtered_chunks.append(np.real(out_chunks[4]))
        filtered_chunks.append(np.real(out_chunks[5]))
        filtered_chunks.append(np.imag(out_chunks[5]))
        filtered_chunks.append(np.real(out_chunks[8]))
    if len(chunks)==4:
        filtered_chunks.append(np.real(out_chunks[0]))
        filtered_chunks.append(np.real(out_chunks[1]))
        filtered_chunks.append(np.imag(out_chunks[1]))
        filtered_chunks.append(np.real(out_chunks[3]))

    filtered_chunks = np.array(filtered_chunks).astype(np.float32)

    return filtered_chunks
    