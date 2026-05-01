
import numpy as np
from osgeo import gdal
import os, glob
import tempfile,shutil
from tqdm import tqdm
from polsartools.utils.utils import time_it, mlook_arr
# from polsartools.utils.io_utils import write_T3, write_C3
from polsartools.preprocess.convert_S2 import convert_S
gdal.UseExceptions()


def load_a4(input_file, output_file, calfac_linear,mat=None,
            driver='GTiff', 
            cog=False, ovr=[2, 4, 8, 16], comp=False,
            block_size=1000):
    
    with open(input_file, mode='rb') as fp:
        # 2. Extract Metadata for dimensions
        fp.seek(236)
        nline = int(fp.read(8))
        fp.seek(248)
        npixel = int(fp.read(8))
        
        prefix_len = 800 
        nrec_bytes = prefix_len + (npixel * 8)
        row_width_floats = nrec_bytes // 4
        pixel_start_idx = prefix_len // 4

        # print(f"Image Dimensions: {nline} lines x {npixel} pixels")
        # print(f"Processing in blocks of {block_size} lines...")

        # 3. Initialize GDAL Output File
        if driver =='ENVI':
            driver = gdal.GetDriverByName('ENVI')
            dataset = driver.Create(output_file, npixel, nline, 1, gdal.GDT_CFloat32)
        else:
            driver = gdal.GetDriverByName("GTiff")
            options = ['BIGTIFF=IF_SAFER']
            if comp:
                # options += ['COMPRESS=DEFLATE', 'PREDICTOR=2', 'ZLEVEL=9']
                options += ['COMPRESS=LZW']
            if cog:
                options += ['TILED=YES', 'BLOCKXSIZE=512', 'BLOCKYSIZE=512']
            
            dataset = driver.Create(
                output_file,
                npixel, nline,     
                1,                   
                gdal.GDT_CFloat32,
                options    
            )
        
        
        
        out_band = dataset.GetRasterBand(1)

        # 4. Block-by-Block Loop
        fp.seek(720) # Skip CEOS header
        print(f"Extracting single-look elements...")
        for start_line in tqdm(range(0, nline, block_size)):
            # Calculate how many lines to read (don't overflow the end)
            current_block_height = min(block_size, nline - start_line)
            
            # Read block of bytes
            raw_bytes = fp.read(nrec_bytes * current_block_height)
            if not raw_bytes:
                break
                
            # Convert to numpy array
            data = np.frombuffer(raw_bytes, dtype='>f4').reshape(current_block_height, row_width_floats)
            
            # Extract Pixels (strip 800-byte prefix)
            pixel_data = data[:, pixel_start_idx:]
            
            # Convert to Complex and Apply Calibration + Absolute Value
            # Process: Real/Imag -> Complex -> Magnitude -> Scale
            # real = pixel_data[:, ::2]
            # imag = pixel_data[:, 1::2]
            
            slc = (pixel_data[:, ::2] + 1j*pixel_data[:, 1::2]).astype(np.float64)* calfac_linear
            # Logic: abs(complex) * calfac = sqrt(re^2 + im^2) * calfac
            # We do this in one step to save memory
            # mag_block = np.sqrt(real**2 + imag**2).astype(np.float32) * calfac_linear
            
            # 5. Write block to disk
            # WriteArray(array, x_offset, y_offset)
            out_band.WriteArray(slc, 0, start_line)
            
            # if start_line % (block_size * 5) == 0:
            #     print(f"Progress: {start_line}/{nline} lines processed.")
        
        if cog:
            dataset.BuildOverviews("NEAREST", ovr)
        # Finalize
        dataset.FlushCache()
        dataset = None 
    if mat == 'S2' or mat == 'Sxy':
        print(f"Saved file: {output_file}")

def write_a2_rst(out_file,data,
                driver='GTiff', out_dtype=gdal.GDT_CFloat32,
                mat=None,
               cog=False, ovr=[2, 4, 8, 16], comp=False
                 ):

    if driver =='ENVI':
        # Create GDAL dataset
        driver = gdal.GetDriverByName(driver)
        dataset = driver.Create(
            out_file,
            data.shape[1],      
            data.shape[0],      
            1,                   
            out_dtype    
        )


    else:
        driver = gdal.GetDriverByName("GTiff")
        options = ['BIGTIFF=IF_SAFER']
        if comp:
            # options += ['COMPRESS=DEFLATE', 'PREDICTOR=2', 'ZLEVEL=9']
            options += ['COMPRESS=LZW']
        if cog:
            options += ['TILED=YES', 'BLOCKXSIZE=512', 'BLOCKYSIZE=512']
        
        dataset = driver.Create(
            out_file,
            data.shape[1],      
            data.shape[0],      
            1,                   
            out_dtype,
            options    
        )

        
    dataset.GetRasterBand(1).WriteArray(data)
    # outdata.GetRasterBand(1).SetNoDataValue(0)##if you want these values transparent
    dataset.FlushCache() ##saves to disk!!
    
    if cog:
        dataset.BuildOverviews("NEAREST", ovr)
    dataset = None
    if mat == 'S2' or mat == 'Sxy':
        print(f"Saved file: {out_file}")

#################################################################################################

@time_it    
def import_alos4_uwd_l11(in_dir,mat='C2', azlks=25,rglks=5, 
                 fmt='tif', cog=False,ovr = [2, 4, 8, 16],comp=False,
                 out_dir=None,
                  cf_dB=-83):
    """
    Extracts the C2 matrix elements (C11, C22, and C12) from ALOS-2 Wide Beam Dual-Pol (WBD) CEOS data 
    and saves them into respective binary files.

    Example:
    --------
    >>> import_alos4_uwd_l11("path_to_folder", azlks=25, rglks=5)
    This will extract the C2 matrix elements from the ALOS-2 Wide Beam Dual-Pol data 
    in the specified folder and save them in the 'C2' directory.
    
    Parameters:
    -----------
    in_dir : str
        The path to the folder containing the ALOS-2 Wide Beam Dual-Pol CEOS data files.
    mat : str, optional (default = 'S2' or 'Sxy)
        Type of matrix to extract. Valid options: 'Sxy','C2', 'T2'.
    azlks : int, optional (default=25)
        The number of azimuth looks for multi-looking.

    rglks : int, optional (default=5)
        The number of range looks for multi-looking.
    
    swath : int, optional (default=1)
        The swath number [1,2,3,4,5].

    fmt : {'tif', 'bin'}, optional (default='tif')
        Output format:
        - 'tif': GeoTIFF 
        - 'bin': Raw binary format

    cog : bool, optional (default=False)
        If True, outputs will be saved as Cloud Optimized GeoTIFFs with internal tiling and overviews.

    ovr : list of int, optional (default=[2, 4, 8, 16])
        Overview levels for COG generation. Ignored if cog=False.

    comp : bool, optional (default=False)
        If True, applies LZW compression to GeoTIFF outputs.

    out_dir : str or None, optional (default=None)
        Directory to save output files. If None, a folder named after the matrix type will be created
        in the same location as the input file.
                
    cf_dB : float, optional (default=-83)
        The calibration factor in dB used to scale the raw radar data. It is applied to 
        the HH and HV polarization data before matrix computation.

    Returns:
    --------
    None
        The function does not return any value. Instead, it creates a folder named `C2` 
        (if not already present) and saves the following binary files:

        - `C11.bin`: Contains the C11 matrix elements.
        - `C22.bin`: Contains the C22 matrix elements.
        - `C12_real.bin`: Contains the real part of the C12 matrix.
        - `C12_imag.bin`: Contains the imaginary part of the C12 matrix.
        - `config.txt`: A text file containing grid dimensions and polarimetric configuration.

    Raises:
    -------
    FileNotFoundError
        If the required ALOS-2 data files (e.g., `IMG-HH` and `IMG-HV`) cannot be found in the specified folder.

    ValueError
        If the calibration factor is invalid or if the files are not in the expected format.


    """
    
    
    
    valid_dual_pol = ['Sxy', 'C2', 'T2']
    valid_matrices = valid_dual_pol

    if mat not in valid_matrices:
        raise ValueError(f"Invalid matrix type '{mat}'. \n Supported types are:\n"
                        f"  Dual-pol: {sorted(valid_dual_pol)}")
    
    temp_dir = None
    ext = 'bin' if fmt == 'bin' else 'tif'
    driver = 'ENVI' if fmt == 'bin' else None

    # Final output directory
    if out_dir is None:
        final_out_dir = os.path.join(in_dir, mat)
    else:
        final_out_dir = os.path.join(out_dir, mat)
    os.makedirs(final_out_dir, exist_ok=True)

    # Intermediate output directory
    if mat in ['Sxy']:
        base_out_dir = final_out_dir
    else:
        temp_dir = tempfile.mkdtemp(prefix='temp_S2_')
        base_out_dir = temp_dir
        

    hh_file = glob.glob(os.path.join(in_dir,f'IMG-HH-*')) [0]

    hv_file = glob.glob(os.path.join(in_dir,f'IMG-HV-*')) [0]

    calfac_linear = np.sqrt(10 ** ((cf_dB) / 10))
    load_a4(hh_file, os.path.join(base_out_dir, f's11.{ext}'), 
                calfac_linear, mat=mat, driver=driver, cog=cog, ovr=ovr, comp=comp,
                block_size=1000)

    load_a4(hv_file, os.path.join(base_out_dir, f's12.{ext}'), 
                calfac_linear, mat=mat, driver=driver, cog=cog, ovr=ovr, comp=comp,
                block_size=1000)
    
    # Matrix conversion if needed
    if mat in ['C2', 'T2']:
        convert_S(base_out_dir, mat=mat, azlks=azlks, rglks=rglks, cf=1,
                  fmt=fmt, out_dir=final_out_dir, cog=cog, ovr=ovr, comp=comp)

        # Clean up temp directory
        if temp_dir:
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Warning: Could not delete temporary directory {temp_dir}: {e}")