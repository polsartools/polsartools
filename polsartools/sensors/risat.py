
import os,glob,re
import struct
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from osgeo import gdal
gdal.UseExceptions()
from polsartools.utils.utils import conv2d,time_it, mlook_arr

def get_bandmeta(filepath):
    metadata_dict = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line:
                key, value = line.split('=', 1) 
                metadata_dict[key.strip()] = value.strip()
    return metadata_dict

def extract_lengths(leader_file, data_file):
    def htonl(val):
        """Convert unsigned int from host to network byte order"""
        return struct.unpack(">I", struct.pack("<I", val))[0]
    tmp = bytearray(32768)

    # --- Read Leader File ---
    with open(leader_file, "rb") as f:
        for _ in range(9):  # Skipping 9 known records
            f.read(8)
            length = htonl(struct.unpack("<I", f.read(4))[0])
            f.read(length - 12)

        f.read(8332)
        calib_factor = f.read(16).decode().strip()

    # --- Read Image File ---
    with open(data_file, "rb") as f:
        f.seek(8)
        header_length = htonl(struct.unpack("<I", f.read(4))[0])

        f.seek(186)
        data_record_length = int(f.read(6).decode())

        f.seek(276)
        prefix_record_length = int(f.read(6).decode())

    # print("header_length:",header_length)
    # print("prefix_record_length:",prefix_record_length)
    # print("data_record_length:",data_record_length)

    return header_length,prefix_record_length,data_record_length

def load_data(filepath, NoScans, header_length, record_length, prefix_length,
                                 bytes_per_pixel=4, dtype=np.dtype('>i2')):
    
    prefix_length += 12
    
    pixels_per_line = (record_length - prefix_length) // bytes_per_pixel
    # image_lines = NoScans
    # payload_bytes_per_line = pixels_per_line * bytes_per_pixel

    with open(filepath, 'rb') as f:
        f.seek(header_length)
        full_data = np.frombuffer(f.read(NoScans * record_length), dtype=np.uint8)
    
    full_data = full_data.reshape((NoScans, record_length))
    image_payload = full_data[:, prefix_length:]


    raw_pixels = image_payload.reshape(-1).view(dtype)
    image = raw_pixels.reshape(NoScans, pixels_per_line, 2)

    return image

def write_rst(file,wdata,dtype):
    [cols, rows] = wdata.shape
    if '.bin' in file:
        driver = gdal.GetDriverByName("ENVI")
        outdata = driver.Create(file, rows, cols, 1, dtype)
    else:
        driver = gdal.GetDriverByName("GTiff")
        outdata = driver.Create(file, rows, cols, 1, dtype,
                                options=['COMPRESS=DEFLATE','PREDICTOR=2','ZLEVEL=9',])
    
    outdata.SetDescription(file)
    outdata.GetRasterBand(1).WriteArray(wdata)
    outdata.FlushCache() 
    outdata = None


def inrp_inc(inc_array, target_shape):
    input_shape = inc_array.shape
    y = np.linspace(0, 1, input_shape[0])  # scan direction
    x = np.linspace(0, 1, input_shape[1])  # pixel direction

    # Create interpolator
    interpolator = RegularGridInterpolator((y, x), inc_array, bounds_error=False, fill_value=None)
    target_y = np.linspace(0, 1, target_shape[0])
    target_x = np.linspace(0, 1, target_shape[1])
    yy, xx = np.meshgrid(target_y, target_x, indexing='ij')

    # Stack points as (N, 2) for interpolation
    points = np.stack([yy.ravel(), xx.ravel()], axis=-1)
    interpolated = interpolator(points).reshape(target_shape)

    return interpolated.astype(np.float32)

def get_inc(file_path):
    inc_angles = []
    num_records = None
    num_samples = None

    with open(file_path, 'r') as f:
        for line in f:
            # Extract shape from header lines
            if "#Number of Records in Grid:" in line:
                num_records = int(re.search(r'\d+', line).group())
            elif "#Number of Samples in Grid:" in line:
                num_samples = int(re.search(r'\d+', line).group())

            elif line.strip().startswith("#") or line.strip() == "":
                continue  

            else:
                parts = line.strip().split()
                if len(parts) >= 4:
                    try:
                        inc_angles.append(float(parts[3]))
                    except ValueError:
                        pass

    inc_array = np.array(inc_angles)

    if num_records and num_samples:
        inc_array = inc_array.reshape((num_records, num_samples))


    return inc_array.astype(np.float32)

@time_it
def risat_l11(prod_dir,matrixType='C2',
              azlks=10,rglks=7,outType='tif',
              out_dir = None):

    """
    Extracts the Sxy/C2 matrix elements from a RISAT-1A compact-pol SLC data 
    
    Example:
    --------
    >>> risat_l11("path_to_folder", matrixType='C2', azlks=10, rglks=7, outType='tif')
    This will extract the C2 from compact-pol RISAT-1A SLC data and save them as geotiff files.
    Additionally, it will also extract the incidence angle map and save it as a geotiff file.
    
    Parameters:
    -----------
    inFile : str
        The path to the RISAT-1A SLCfolder containing the data and metadata.
    matrixType : str, optional (default = 'C2')
        Type of matrix to extract. Valid options 'Sxy','C2'.
    azlks : int, optional (default=10)
        The number of azimuth looks for multi-looking. 
    rglks : int, optional (default=7)
        The number of range looks for multi-looking. 
    outType : str, optional (default='tif')
        The type of output file to save the matrix elements. Valid options are 'tif' and 'bin'.
    out_dir : str, optional (default=None)
        The path to the output directory. If not provided, the function will create a matrix folder at the inFile directory.

    Returns:
    --------
    None
        The function does not return any value. Instead, it creates a folder 
        named `Sxy/C2` (if not already present) and saves the geotiff/binary files:



    """   


    band_meta = os.path.join(prod_dir,'BAND_META.txt')
    metadata_dict = get_bandmeta(band_meta)
    ImagingMode = str(metadata_dict['ImagingMode'])
    NoOfPolarizations=int(metadata_dict['NoOfPolarizations'])
    
    polList = []
    for ii in range(NoOfPolarizations):
        polList.append(metadata_dict[f'TxRxPol{ii+1}'])
    
    print(f"Detected {ImagingMode}: {polList} acquired on {metadata_dict['DateOfPass']}")
    
    
    sr_grid = glob.glob(os.path.join(prod_dir,f'*{polList[0]}_L1_SlantRange_grid.txt'))[0]
    leader_file = glob.glob(os.path.join(prod_dir,f'scene_{polList[0]}/lea_01.001'))[0]
    data_file = glob.glob(os.path.join(prod_dir,f'scene_{polList[0]}/dat_01.001'))[0]
    
    header_length,prefix_length,record_length = extract_lengths(leader_file, data_file)
    
    # print(header_length,prefix_length,record_length)
    
    
    inc_array = get_inc(sr_grid)
    
    target_shape = (int(metadata_dict['NoScans']), int(metadata_dict['NoPixels']))
    resized_inc = inrp_inc(inc_array, target_shape)
    
    if out_dir is None:
        out_dir = prod_dir
    else:
        os.makedirs(out_dir, exist_ok=True)
    
    # print("Resized incidence angle shape:", resized_inc.shape)
    if outType=='bin':
        write_rst(os.path.join(out_dir,'inc.bin'),resized_inc,gdal.GDT_Float32)
        print(f"Saved file: {os.path.join(out_dir,'inc.bin')}")
    else:
        write_rst(os.path.join(out_dir,'inc.tif'),resized_inc,gdal.GDT_Float32)
        print(f"Saved file: {os.path.join(out_dir,'inc.tif')}")
        
    del inc_array
    
    
    if 'RH' in polList and 'RV' in polList:
        inFile = glob.glob(os.path.join(prod_dir,'scene_RH/dat_01.001'))[0]
        s11 = load_data(inFile, int(metadata_dict['NoScans']), 
                         header_length, record_length, prefix_length,
                         bytes_per_pixel=int(metadata_dict['BytesPerPixel']), dtype=np.dtype('>i2'))
        
        cc_dB = float(metadata_dict['Calibration_Constant_RH'])
        cc_lin_rh = np.sqrt(10**(cc_dB/10))
        inc_center = float(metadata_dict['IncidenceAngle'])
        # print(cc_dB,inc_center)
        s11 = ((s11[:,:,0]*np.sqrt(np.sin(resized_inc*np.pi/180)/np.sin(inc_center*np.pi/180))/cc_lin_rh)+\
                 1j*(s11[:,:,1]*np.sqrt(np.sin(resized_inc*np.pi/180)/np.sin(inc_center*np.pi/180))/cc_lin_rh)).astype(np.complex64)
    
        
        inFile = glob.glob(os.path.join(prod_dir,'scene_RV/dat_01.001'))[0]
        s21 = load_data(inFile, int(metadata_dict['NoScans']), 
                         header_length, record_length, prefix_length,
                         bytes_per_pixel=int(metadata_dict['BytesPerPixel']), dtype=np.dtype('>i2'))
        cc_dB = float(metadata_dict['Calibration_Constant_RV'])
        # print(cc_dB)
        cc_lin_rv = np.sqrt(10**(cc_dB/10))
        s21 = ((s21[:,:,0]*np.sqrt(np.sin(resized_inc*np.pi/180)/np.sin(inc_center*np.pi/180))/cc_lin_rv)+\
                 1j*(s21[:,:,1]*np.sqrt(np.sin(resized_inc*np.pi/180)/np.sin(inc_center*np.pi/180))/cc_lin_rv)).astype(np.complex64)
    
        if matrixType == 'Sxy':
            outFolder = os.path.join(out_dir,'Sxy')
            os.makedirs(outFolder,  exist_ok=True)
            
            if outType=='bin':
                write_rst(os.path.join(outFolder,'s11.bin'),s11,gdal.GDT_CFloat32)
                print(f"Saved file: {os.path.join(outFolder,'s11.bin')}")
                write_rst(os.path.join(outFolder,'s21.bin'),s21,gdal.GDT_CFloat32)
                print(f"Saved file: {os.path.join(outFolder,'s21.bin')}")
            else:
                write_rst(os.path.join(outFolder,'s11.tif'),s11,gdal.GDT_CFloat32)
                print(f"Saved file: {os.path.join(outFolder,'s11.tif')}")
                write_rst(os.path.join(outFolder,'s21.tif'),s21,gdal.GDT_CFloat32)
                print(f"Saved file: {os.path.join(outFolder,'s21.tif')}")
                
        elif matrixType == 'C2':
            outFolder = os.path.join(out_dir,'C2')
            os.makedirs(outFolder,  exist_ok=True)
            
            write_rst(os.path.join(outFolder,'inc.tif'),mlook_arr(resized_inc,azlks,rglks).astype(np.float32),
                      gdal.GDT_Float32)
            
            del resized_inc
            
            if outType=='bin':
                write_rst(os.path.join(outFolder,'C11.bin'),mlook_arr(np.abs(s11)**2,azlks,rglks).astype(np.float32),gdal.GDT_Float32)
                print(f"Saved file: {os.path.join(outFolder,'C11.bin')}")
                write_rst(os.path.join(outFolder,'C22.bin'),mlook_arr(np.abs(s21)**2,azlks,rglks).astype(np.float32),gdal.GDT_Float32)
                print(f"Saved file: {os.path.join(outFolder,'C22.bin')}")            
            else:
                write_rst(os.path.join(outFolder,'C11.tif'),mlook_arr(np.abs(s11)**2,azlks,rglks).astype(np.float32),gdal.GDT_Float32)
                print(f"Saved file: {os.path.join(outFolder,'C11.tif')}")
                write_rst(os.path.join(outFolder,'C22.tif'),mlook_arr(np.abs(s21)**2,azlks,rglks).astype(np.float32),gdal.GDT_Float32)
                print(f"Saved file: {os.path.join(outFolder,'C22.tif')}")

            
            C12 = mlook_arr(s11*np.conjugate(s21),azlks,rglks).astype(np.complex64)
            

            rows,cols = C12.shape
            if outType=='bin':
                write_rst(os.path.join(outFolder,'C12_real.bin'),np.real(C12).astype(np.float32),gdal.GDT_Float32)
                print(f"Saved file: {os.path.join(outFolder,'C12_real.bin')}")
                write_rst(os.path.join(outFolder,'C12_imag.bin'),np.imag(C12).astype(np.float32),gdal.GDT_Float32)
                print(f"Saved file: {os.path.join(outFolder,'C12_imag.bin')}")
            else:
                write_rst(os.path.join(outFolder,'C12_real.tif'),np.real(C12).astype(np.float32),gdal.GDT_Float32)
                print(f"Saved file: {os.path.join(outFolder,'C12_real.tif')}")
                write_rst(os.path.join(outFolder,'C12_imag.tif'),np.imag(C12).astype(np.float32),gdal.GDT_Float32)
                print(f"Saved file: {os.path.join(outFolder,'C12_imag.tif')}")


            file = open(outFolder +'/config.txt',"w+")
            file.write('Nrow\n%d\n---------\nNcol\n%d\n---------\nPolarCase\nmonostatic\n---------\nPolarType\npp1'%(rows,cols))

            file.close()  
        else:
            print("Only Sxy, C2 matrices are supported for RISAT compact-pol data.\
                  Please check the matrixType argument.\
                  Example matrixType='C2' or 'Sxy'")
