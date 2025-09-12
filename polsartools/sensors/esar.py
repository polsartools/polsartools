

import os,glob,time
import numpy as np
from osgeo import gdal, osr
gdal.UseExceptions()

from polsartools.utils.utils import time_it,mlook_arr
from polsartools.preprocess.convert_S2 import convert_S
# import gc
def get_geometa(txt_path):
    metadata = {}
    in_section = False

    with open(txt_path, 'r') as file:
        for line in file:
            if line.strip().startswith("VII.)-GEOCODED IMAGE PARAMETERS:"):
                in_section = True
                continue
            if in_section and line.strip().startswith("="):  # End of section
                break
            if in_section and ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower().replace(" ", "_").replace("(", "").replace(")", "")
                value = value.strip().split()[0]
                try:
                    value = float(value)
                except ValueError:
                    pass
                metadata[key] = value

    return metadata

def parse_slc_geo_info(filepath):
    geo_info = {}              # filename → {frequency, polarization}
    polarization_map = {}      # polarization → filename
    frequency = None           # assumed to be consistent across all entries

    with open(filepath, "r") as file:
        lines = file.readlines()

    in_section_11 = False
    for line in lines:
        line = line.strip()

        # Detect start of section [11]
        if line.startswith("[11]"):
            in_section_11 = True
            continue

        # Stop parsing when section [12] begins
        if in_section_11 and line.startswith("[12]"):
            break

        if in_section_11:
            parts = line.split()
            for i in range(len(parts)):
                part = parts[i]
                if part.endswith(".dat") and i + 1 < len(parts):
                    pol_part = parts[i + 1]
                    if pol_part.startswith("(") and "-" in pol_part and pol_part.endswith(")"):
                        freq, pol = pol_part.strip("()").split("-")
                        geo_info[part] = {
                            "frequency": freq,
                            "polarization": pol
                        }
                        polarization_map[pol] = part
                        frequency = freq  # assumes same frequency across all

    return geo_info, polarization_map, frequency

def get_size(filename):
    found_title = False
    records = words = None

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()

            # Detect the start of the desired block
            if line.startswith("[11]  TITLE:") and "SingleLook Complex Image" in line:
                found_title = True

            # Extract the ARRAY line once inside the correct block
            elif found_title and "ARRAY:" in line and "Records with" in line:
                parts = line.split()
                try:
                    records = int(parts[1])  
                    words = int(parts[4])    
                    break  # Stop after finding the needed info
                except (IndexError, ValueError):
                    raise ValueError("Couldn't parse records and words from ARRAY line.")

    if records is not None and words is not None:
        return records, words
    else:
        raise ValueError("Specified TITLE block or ARRAY line not found.")

def get_meta(in_dir):
    
    txt_path_list = glob.glob(os.path.join(in_dir, "*_README_GTC.txt"))
    if len(txt_path_list)>0:
        txt_path = txt_path_list[0]
    else:
        raise ValueError("Metadata file not found!")
    
    metadata = get_geometa(txt_path)
    records, words = get_size(txt_path)
    
    metadata['slc records'] = records
    metadata['slc words'] = words
    
    
    
    geo_dict, files, radar_freq = parse_slc_geo_info(txt_path)


    return metadata, files, radar_freq

def read_data(filename,num_records,words_per_record,calfactor=1000):
    with open(filename, 'rb') as f:
        # Skip the header (2 LONGs = 8 bytes)
        f.seek(8)
        # Read the rest as signed 16-bit integers
        raw = np.fromfile(f, dtype=np.int16)

    # Convert to complex numbers: real, imag, real, ...
    complex_data = raw[::2]/calfactor + 1j * raw[1::2]/calfactor

    complex_per_record = words_per_record // 2
    complex_data = complex_data.reshape((num_records, complex_per_record))
    
    complex_data[complex_data==-9.998-1j*9.998] = np.nan
    complex_data = np.flipud(complex_data)
    
    return complex_data 


def get_incmap(filename, num_records, words_per_record):
    # Skip the first 8 bytes of header (2 LONGs)
    with open(filename, 'rb') as f:
        f.seek(8)  # Skip header
        data = np.fromfile(f, dtype=np.int16, count=num_records * words_per_record)

    # Reshape and scale to radians
    angle_array = data.reshape((num_records, words_per_record)) / 1000.0

    return angle_array

def write_data(array, metadata, out_file, 
               driver='GTiff', out_dtype=gdal.GDT_CFloat32,
               mat=None,
               cog=False, ovr=[2, 4, 8, 16], comp=False):
    # Extract spatial info
    pixel_size_x = metadata['pixel_spacing_east__slc_[m]']
    pixel_size_y = metadata['pixel_spacing_north_slc_[m]']
    
    min_x = metadata['minimum_easting_slc']
    max_y = metadata['maximum_northing_slc']
    zone = int(metadata['projection_zone'])

    # UTM projection setup
    srs = osr.SpatialReference()
    srs.SetUTM(zone, True)  # True for northern hemisphere
    srs.SetWellKnownGeogCS("WGS84")
    if driver =='ENVI':
        # Create GDAL dataset
        driver = gdal.GetDriverByName(driver)
        dataset = driver.Create(
            out_file,
            array.shape[1],      # width
            array.shape[0],      # height
            1,                   # number of bands
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
            array.shape[1],      # width
            array.shape[0],      # height
            1,                   # number of bands
            out_dtype,
            options    
        )
        

    geotransform = (min_x, pixel_size_x, 0, max_y, 0, -pixel_size_y)
    dataset.SetGeoTransform(geotransform)
    dataset.SetProjection(srs.ExportToWkt())
    dataset.GetRasterBand(1).WriteArray(array)
    
    if cog:
        dataset.BuildOverviews("NEAREST", ovr)

    # Cleanup
    dataset.FlushCache()
    dataset = None
    del dataset
    
    if mat == 'S2' or mat == 'Sxy':
        print(f"Saved file: {out_file}")



@time_it
def esar_gtc(in_dir,mat='S2',
             azlks=3,rglks=3,
             fmt='tif', cog=False,ovr = [2, 4, 8, 16],comp=False,
             out_dir=None,
             recip=False,
             ):
    
    """
    Extracts the Sxy/C2/T2 (for dual-pol), S2/C4/T4/C3/T3/C2/T2 (for full-pol) matrix elements from a ESAR GTC SLC data 
    
    Example:
    --------
    >>> esar_gtc("path_to_folder", mat='C3', azlks=3, rglks=3)
    This will extract the C3 from full-pol ESAR GTC SLC data and save them as geotiff files.
    Additionally, it will also extract the incidence angle map and save it as a geotiff file.
    
    Parameters:
    -----------
    in_dir : str
        The path to the ESAR GTC folder containing the data and metadata.
        
    mat : str, optional (default = 'S2' or 'Sxy)
        Type of matrix to extract. Valid options for Full-pol: 'S2',  'C4, 'C3', 'T4', 
        'T3', 'C2HX', 'C2VX', 'C2HV','T2HV'and Dual-pol: 'Sxy','C2', 'T2'.

    azlks : int, optional (default=3)
        The number of azimuth looks for multi-looking. 

    rglks : int, optional (default=3)
        The number of range looks for multi-looking. 
    
    recip : bool, default=False
        If True, scattering matrix reciprocal symmetry is applied, i.e, S_HV = S_VH.
    
    Returns:
    --------
    None
        The function does not return any value. Instead, it creates a folder 
        named `S2/C3/T3/C2/T2` (if not already present) and saves the geotiff/binary files:



    """    
    
    

    metadata, files, radar_freq = get_meta(in_dir)
    
    print(f"Detected {radar_freq}-band {list(files.keys())}")
    
    
    if len(files) == 4:

        s11File = glob.glob(os.path.join(in_dir, files['HH']))[0]
        s12File = glob.glob(os.path.join(in_dir, files['HV']))[0]
        s21File = glob.glob(os.path.join(in_dir, files['VH']))[0]
        s22File = glob.glob(os.path.join(in_dir, files['VV']))[0]
        

        
        s11  = read_data(s11File,metadata['slc records'],metadata['slc words'])
        s12  = read_data(s12File,metadata['slc records'],metadata['slc words'])
        s21  = read_data(s21File,metadata['slc records'],metadata['slc words'])
        
        if recip:
            s12 = (s12 + s21) / 2
            s21 = s12.copy()
        
        s22  = read_data(s22File,metadata['slc records'],metadata['slc words'])

        
        inc_file = glob.glob(os.path.join(in_dir, "incmap*.dat"))[0]
        inc = get_incmap(inc_file, metadata['slc records'],metadata['slc words']//2)
        inc[inc<0]=np.nan
        inc = np.flipud(inc)
        
   
        out_dir = os.path.join(in_dir, 'S2')
        # else:
        #     out_dir = os.path.join(in_dir, 'temp_S2')
        os.makedirs(out_dir, exist_ok=True)
                
        if fmt == 'bin':
            write_data(s11, metadata, os.path.join(out_dir,'s11.bin'),driver='ENVI',mat=mat)
            write_data(s12, metadata, os.path.join(out_dir,'s12.bin'),driver='ENVI',mat=mat)
            write_data(s21, metadata, os.path.join(out_dir,'s21.bin'),driver='ENVI',mat=mat)
            write_data(s22, metadata, os.path.join(out_dir,'s22.bin'),driver='ENVI',mat=mat)
        else:
            write_data(s11, metadata, os.path.join(out_dir,'s11.tif'),mat=mat,cog=cog, ovr=ovr, comp=comp)
            write_data(s12, metadata, os.path.join(out_dir,'s12.tif'),mat=mat,cog=cog, ovr=ovr, comp=comp)
            write_data(s21, metadata, os.path.join(out_dir,'s21.tif'),mat=mat,cog=cog, ovr=ovr, comp=comp)
            write_data(s22, metadata, os.path.join(out_dir,'s22.tif'),mat=mat,cog=cog, ovr=ovr, comp=comp)

        write_data(inc, metadata,  os.path.join(out_dir,'inc.tif'), out_dtype=gdal.GDT_Float32,mat=mat,cog=cog, ovr=ovr, comp=comp)

        if mat == 'S2': 
            pass

        elif mat == 'T3':
            T3_dir = os.path.join(in_dir, 'T3')
            # convert_S()
            convert_S(out_dir, mat='T3', azlks=azlks,rglks=rglks, cf = 1, 
                  fmt=fmt,out_dir=T3_dir,
                  cog=cog, ovr=ovr, comp=comp
                #   write_flag=True, max_workers=None,block_size=(512, 512)
                  )
            
            in_rows, in_cols = inc.shape
            out_x_size = in_cols // rglks
            out_y_size = in_rows // azlks
            
            metadata['pixel_spacing_east__slc_[m]'] = (metadata['pixel_spacing_east__slc_[m]']* in_cols) / out_x_size
            metadata['pixel_spacing_north_slc_[m]'] = (metadata['pixel_spacing_north_slc_[m]']* in_rows) / out_y_size
            
            inc = mlook_arr(inc,azlks,rglks).astype(np.float32)
            write_data(inc, metadata,  os.path.join(T3_dir,'inc.tif'),  
                       out_dtype=gdal.GDT_Float32,cog=cog, ovr=ovr, comp=comp)
            
            # remove temp directory
            # if os.path.exists(os.path.join(in_dir, 'temp_S2')):
                
            #     gc.collect()
            #     time.sleep(1)
            #     os.remove(os.path.join(in_dir, 'temp_S2'))
            
            
        elif mat == 'C3':
            C3_dir = os.path.join(in_dir, 'C3')
            # convert_S()
            convert_S(out_dir, mat='C3', azlks=azlks,rglks=rglks, cf = 1, 
                  fmt=fmt,out_dir=C3_dir,cog=cog, ovr=ovr, comp=comp
                #   cog_flag=False, cog_overviews = [2, 4, 8, 16], 
                #   write_flag=True, max_workers=None,block_size=(512, 512)
                  )
            
            in_rows, in_cols = inc.shape
            out_x_size = in_cols // rglks
            out_y_size = in_rows // azlks
            
            metadata['pixel_spacing_east__slc_[m]'] = (metadata['pixel_spacing_east__slc_[m]']* in_cols) / out_x_size
            metadata['pixel_spacing_north_slc_[m]'] = (metadata['pixel_spacing_north_slc_[m]']* in_rows) / out_y_size
            
            inc = mlook_arr(inc,azlks,rglks).astype(np.float32)
            write_data(inc, metadata,  os.path.join(C3_dir,'inc.tif'),  
                       out_dtype=gdal.GDT_Float32,cog=cog, ovr=ovr, comp=comp)


        elif mat == 'C4':
            C4_dir = os.path.join(in_dir, 'C4')
            # convert_S()
            convert_S(out_dir, mat='C4', azlks=azlks,rglks=rglks, cf = 1, 
                  fmt=fmt,out_dir=C4_dir,cog=cog, ovr=ovr, comp=comp
                  )
            
            in_rows, in_cols = inc.shape
            out_x_size = in_cols // rglks
            out_y_size = in_rows // azlks
            
            metadata['pixel_spacing_east__slc_[m]'] = (metadata['pixel_spacing_east__slc_[m]']* in_cols) / out_x_size
            metadata['pixel_spacing_north_slc_[m]'] = (metadata['pixel_spacing_north_slc_[m]']* in_rows) / out_y_size
            
            inc = mlook_arr(inc,azlks,rglks).astype(np.float32)
            write_data(inc, metadata,  os.path.join(C4_dir,'inc.tif'),  
                       out_dtype=gdal.GDT_Float32,cog=cog, ovr=ovr, comp=comp)


        elif mat == 'T4':
            T4_dir = os.path.join(in_dir, 'T4')
            # convert_S()
            convert_S(out_dir, mat='T4', azlks=azlks,rglks=rglks, cf = 1, 
                  fmt=fmt,out_dir=T4_dir,cog=cog, ovr=ovr, comp=comp
                  )
            
            in_rows, in_cols = inc.shape
            out_x_size = in_cols // rglks
            out_y_size = in_rows // azlks
            
            metadata['pixel_spacing_east__slc_[m]'] = (metadata['pixel_spacing_east__slc_[m]']* in_cols) / out_x_size
            metadata['pixel_spacing_north_slc_[m]'] = (metadata['pixel_spacing_north_slc_[m]']* in_rows) / out_y_size
            
            inc = mlook_arr(inc,azlks,rglks).astype(np.float32)
            write_data(inc, metadata,  os.path.join(T4_dir,'inc.tif'),  
                       out_dtype=gdal.GDT_Float32,cog=cog, ovr=ovr, comp=comp)



        elif mat == 'C2HX':
            c2_dir = os.path.join(in_dir, 'C2HX')
            # convert_S()
            convert_S(out_dir, mat='C2HX', azlks=azlks,rglks=rglks, cf = 1, 
                  fmt=fmt,out_dir=c2_dir,cog=cog, ovr=ovr, comp=comp
                #   cog_flag=False, cog_overviews = [2, 4, 8, 16], 
                #   write_flag=True, max_workers=None,block_size=(512, 512)
                  )
            
            in_rows, in_cols = inc.shape
            out_x_size = in_cols // rglks
            out_y_size = in_rows // azlks
            
            metadata['pixel_spacing_east__slc_[m]'] = (metadata['pixel_spacing_east__slc_[m]']* in_cols) / out_x_size
            metadata['pixel_spacing_north_slc_[m]'] = (metadata['pixel_spacing_north_slc_[m]']* in_rows) / out_y_size
            
            inc = mlook_arr(inc,azlks,rglks).astype(np.float32)
            write_data(inc, metadata,  os.path.join(c2_dir,'inc.tif'),  
                       out_dtype=gdal.GDT_Float32,cog=cog, ovr=ovr, comp=comp)


        elif mat == 'C2VX':
            c2_dir = os.path.join(in_dir, 'C2VX')
            # convert_S()
            convert_S(out_dir, mat='C2VX', azlks=azlks,rglks=rglks, cf = 1, 
                  fmt=fmt,out_dir=c2_dir,cog=cog, ovr=ovr, comp=comp
                #   cog_flag=False, cog_overviews = [2, 4, 8, 16], 
                #   write_flag=True, max_workers=None,block_size=(512, 512)
                  )
            
            in_rows, in_cols = inc.shape
            out_x_size = in_cols // rglks
            out_y_size = in_rows // azlks
            
            metadata['pixel_spacing_east__slc_[m]'] = (metadata['pixel_spacing_east__slc_[m]']* in_cols) / out_x_size
            metadata['pixel_spacing_north_slc_[m]'] = (metadata['pixel_spacing_north_slc_[m]']* in_rows) / out_y_size
            
            inc = mlook_arr(inc,azlks,rglks).astype(np.float32)
            write_data(inc, metadata,  os.path.join(c2_dir,'inc.tif'),  
                       out_dtype=gdal.GDT_Float32,cog=cog, ovr=ovr, comp=comp)


        elif mat == 'C2HV':
            c2_dir = os.path.join(in_dir, 'C2HV')
            # convert_S()
            convert_S(out_dir, mat='C2HV', azlks=azlks,rglks=rglks, cf = 1, 
                  fmt=fmt,out_dir=c2_dir,cog=cog, ovr=ovr, comp=comp
                #   cog_flag=False, cog_overviews = [2, 4, 8, 16], 
                #   write_flag=True, max_workers=None,block_size=(512, 512)
                  )
            
            in_rows, in_cols = inc.shape
            out_x_size = in_cols // rglks
            out_y_size = in_rows // azlks
            
            metadata['pixel_spacing_east__slc_[m]'] = (metadata['pixel_spacing_east__slc_[m]']* in_cols) / out_x_size
            metadata['pixel_spacing_north_slc_[m]'] = (metadata['pixel_spacing_north_slc_[m]']* in_rows) / out_y_size
            
            inc = mlook_arr(inc,azlks,rglks).astype(np.float32)
            write_data(inc, metadata,  os.path.join(c2_dir,'inc.tif'),  
                       out_dtype=gdal.GDT_Float32,cog=cog, ovr=ovr, comp=comp)



        elif mat == 'T2HV':
            t2_dir = os.path.join(in_dir, 'T2HV')
            # convert_S()
            convert_S(out_dir, mat='T2HV', azlks=azlks,rglks=rglks, cf = 1, 
                  fmt=fmt,out_dir=t2_dir,cog=cog, ovr=ovr, comp=comp
                #   cog_flag=False, cog_overviews = [2, 4, 8, 16], 
                #   write_flag=True, max_workers=None,block_size=(512, 512)
                  )
            
            in_rows, in_cols = inc.shape
            out_x_size = in_cols // rglks
            out_y_size = in_rows // azlks
            
            metadata['pixel_spacing_east__slc_[m]'] = (metadata['pixel_spacing_east__slc_[m]']* in_cols) / out_x_size
            metadata['pixel_spacing_north_slc_[m]'] = (metadata['pixel_spacing_north_slc_[m]']* in_rows) / out_y_size
            
            inc = mlook_arr(inc,azlks,rglks).astype(np.float32)
            write_data(inc, metadata,  os.path.join(t2_dir,'inc.tif'),  
                       out_dtype=gdal.GDT_Float32,cog=cog, ovr=ovr, comp=comp)

        else:
            raise ValueError(f"Unsupported matrix type: {mat}. \n\
                Supported types are: S2, T3, C3, C4, T4, C2HX, C2VX, C2HV, T2HV.")



    elif len(files) == 2:
        
        if 'HH' in list(files.keys()) and 'HV' in list(files.keys()):
            s11File = glob.glob(os.path.join(in_dir, files['HH']))[0]
            s12File = glob.glob(os.path.join(in_dir, files['HV']))[0]
        elif 'VH' in list(files.keys()) and 'VV' in list(files.keys()):
            s11File = glob.glob(os.path.join(in_dir, files['VV']))[0]
            s12File = glob.glob(os.path.join(in_dir, files['VH']))[0]
        elif 'HH' in list(files.keys()) and 'VV' in list(files.keys()):
            s11File = glob.glob(os.path.join(in_dir, files['HH']))[0]
            s12File = glob.glob(os.path.join(in_dir, files['VV']))[0]
        
        # s11File = glob.glob(os.path.join(in_dir, "*_ch1_t01_int_slc_geo.dat"))[0]
        # s12File = glob.glob(os.path.join(in_dir, "*_ch2_t01_int_slc_geo.dat"))[0]
        
        out_dir = os.path.join(in_dir, 'Sxy')
        os.makedirs(out_dir, exist_ok=True)
        
        s11  = read_data(s11File,metadata['slc records'],metadata['slc words'])
        s12  = read_data(s12File,metadata['slc records'],metadata['slc words'])
        
        if fmt == 'bin':
            write_data(s11, metadata, os.path.join(out_dir,'s11.bin'),driver='ENVI',mat=mat)
            write_data(s12, metadata, os.path.join(out_dir,'s12.bin'),driver='ENVI',mat=mat)
        else:
            write_data(s11, metadata, os.path.join(out_dir,'s11.tif'),mat=mat,cog=cog, ovr=ovr, comp=comp)
            write_data(s12, metadata, os.path.join(out_dir,'s12.tif'),mat=mat,cog=cog, ovr=ovr, comp=comp)
    
        inc_file = glob.glob(os.path.join(in_dir, "incmap*.dat"))[0]
        inc = get_incmap(inc_file, metadata['slc records'],metadata['slc words']//2)
        inc[inc<0]=np.nan
        inc = np.flipud(inc)
        write_data(inc, metadata,  os.path.join(out_dir,'inc.tif'), out_dtype=gdal.GDT_Float32,mat=mat,cog=cog, ovr=ovr, comp=comp)
        
        
        
        if mat == 'Sxy': 
            pass
        
        elif mat == 'C2':
            c2_dir = os.path.join(in_dir, 'C2')
            # convert_S()
            convert_S(out_dir, mat='C2', azlks=azlks,rglks=rglks, cf = 1, 
                  fmt=fmt,out_dir=c2_dir,cog=cog, ovr=ovr, comp=comp
                #   cog_flag=False, cog_overviews = [2, 4, 8, 16], 
                #   write_flag=True, max_workers=None,block_size=(512, 512)
                  )
            
            in_rows, in_cols = inc.shape
            out_x_size = in_cols // rglks
            out_y_size = in_rows // azlks
            
            metadata['pixel_spacing_east__slc_[m]'] = (metadata['pixel_spacing_east__slc_[m]']* in_cols) / out_x_size
            metadata['pixel_spacing_north_slc_[m]'] = (metadata['pixel_spacing_north_slc_[m]']* in_rows) / out_y_size
            
            inc = mlook_arr(inc,azlks,rglks).astype(np.float32)
            write_data(inc, metadata,  os.path.join(c2_dir,'inc.tif'),  
                       out_dtype=gdal.GDT_Float32,cog=cog, ovr=ovr, comp=comp)
        
        
        elif mat == 'T2':
            t2_dir = os.path.join(in_dir, 'T2')
            # convert_S()
            convert_S(out_dir, mat='T2', azlks=azlks,rglks=rglks, cf = 1, 
                  fmt=fmt,out_dir=t2_dir,cog=cog, ovr=ovr, comp=comp
                #   cog_flag=False, cog_overviews = [2, 4, 8, 16], 
                #   write_flag=True, max_workers=None,block_size=(512, 512)
                  )
            
            in_rows, in_cols = inc.shape
            out_x_size = in_cols // rglks
            out_y_size = in_rows // azlks
            
            metadata['pixel_spacing_east__slc_[m]'] = (metadata['pixel_spacing_east__slc_[m]']* in_cols) / out_x_size
            metadata['pixel_spacing_north_slc_[m]'] = (metadata['pixel_spacing_north_slc_[m]']* in_rows) / out_y_size
            
            inc = mlook_arr(inc,azlks,rglks).astype(np.float32)
            write_data(inc, metadata,  os.path.join(t2_dir,'inc.tif'),  
                       out_dtype=gdal.GDT_Float32,cog=cog, ovr=ovr, comp=comp)        
        else:
            raise ValueError(f"Unsupported matrix type: {mat}. \n\
                        Expected 'Sxy' or 'C2' or 'T2'.")
        
        
    else:
        raise ValueError("Unsupported number of channels detected. Expected 2 or 4 files for SLC data.")
    
