import glob,os,tempfile,shutil
import numpy as np
from osgeo import gdal 
gdal.UseExceptions()
from polsartools.utils.utils import time_it
# from polsartools.utils.io_utils import  write_T3, write_C3
from polsartools.preprocess.convert_S2 import convert_S

def read_rs2_tif(file):
    ds = gdal.Open(file)
    band1 = ds.GetRasterBand(1).ReadAsArray()
    band2 = ds.GetRasterBand(2).ReadAsArray()
    ds=None
    return np.dstack((band1,band2))

def write_rst(out_file,data,
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
    
    if cog:
        dataset.BuildOverviews("NEAREST", ovr)
        
    dataset.GetRasterBand(1).WriteArray(data)
    # outdata.GetRasterBand(1).SetNoDataValue(0)##if you want these values transparent
    dataset.FlushCache() ##saves to disk!!
    dataset = None
    if mat == 'S2' or mat == 'Sxy':
        print(f"Saved file: {out_file}")



def write_s2_bin(file,wdata):
    [cols, rows] = wdata.shape
    driver = gdal.GetDriverByName("ENVI")
    outdata = driver.Create(file, rows, cols, 1, gdal.GDT_CFloat32)
    outdata.SetDescription(file)
    outdata.GetRasterBand(1).WriteArray(wdata)
    outdata.FlushCache()


def read_bin(file):
    ds = gdal.Open(file,gdal.GA_ReadOnly)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    return arr


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

@time_it
def chyaan2_fp(in_dir,mat='T3',azlks=None,rglks=None,
               fmt='tif', cog=False,ovr = [2, 4, 8, 16],comp=False,
             out_dir=None,
             recip=False):
    
    """
    Extracts specified matrix elements (S2, T3, or C3) from Chandrayaan-II DFSAR Full-Pol data 
    and saves them into respective directories.

    Example:
    --------
    >>> chyaan2_fp("path_to_folder", mat='T3', azlks=50, rglks=2)
    This will extract the T3 matrix elements from the Chandrayaan-II DFSAR Full-Pol data 
    in the specified folder and save them in the 'T3' directory.
    
    Parameters:
    -----------
    in_dir : str
        The path to the folder containing the Chandrayaan-II DFSAR Full-Pol data files.

    mat : str, optional (default='T3')
        Type of matrix to extract. Valid options: 'S2',  'C4, 'C3', 'T4', 
        'T3', 'C2HX', 'C2VX', 'C2HV','T2HV'

    azlks : int, optional (default=None)
        The number of azimuth looks for multi-looking. If not specified, the value is derived from 
        the ground range and output line spacing.

    rglks : int, optional (default=None)
        The number of range looks for multi-looking. If not specified, the value is set to 1.

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
        
    recip : bool, optional (default=False)
        If True, scattering matrix reciprocal symmetry is applied, i.e, S_HV = S_VH.
        
    """
    #%%
    valid_full_pol = ['S2', 'C4', 'C3', 'T4', 'T3', 'C2HX', 'C2VX', 'C2HV', 'T2HV']
    # valid_dual_pol = ['Sxy', 'C2', 'T2']
    valid_matrices = valid_full_pol

    if mat not in valid_matrices:
        raise ValueError(f"Invalid matrix type '{mat}'. \n Supported types are:\n"
                        f"  Full-pol: {sorted(valid_full_pol)}\n")


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
    if mat in ['S2', 'Sxy']:
        base_out_dir = final_out_dir
    else:
        temp_dir = tempfile.mkdtemp(prefix='temp_S2_')
        base_out_dir = temp_dir
        
    
    xmlFile = glob.glob(in_dir+'/data/calibrated/*/*sli*.xml')[0]
    fxml = open(xmlFile, 'r')
    for line in fxml:
        if "output_line_spacing" in line:
            # print("output_line_spacing: ", line.split('>')[1].split('<')[0])
            ols = float(line.split('>')[1].split('<')[0])
        if "output_pixel_spacing" in line:
            # print("output_pixel_spacing: ", line.split('>')[1].split('<')[0])
            ops = float(line.split('>')[1].split('<')[0])
        if "isda:incidence_angle" in line:
            # print("incidence_angle: ", line.split('>')[1].split('<')[0])
            inc= float( line.split('>')[1].split('<')[0])
        if "isda:calibration_constant" in line:
            cc = float( line.split('>')[1].split('<')[0])
        if "isda:pulse_bandwidth" in line:
            bw = float( line.split('>')[1].split('<')[0])/1000000
    fxml.close() 
    gRange = ops/np.sin(inc*np.pi/180)
    # multi-llok factor 
    mlf = int(np.round(gRange/ols,0))

    ds = gdal.Open(glob.glob(in_dir+'/data/calibrated/*/*sli*_hh_*.tif')[0])
    cols = ds.RasterXSize  
    rows = ds.RasterYSize 


    lines = ['output_line_spacing '+ str(ols)+'\n',
            'output_pixel_spacing '+ str(ops)+'\n',
            'ground_range '+ str(gRange)+'\n',
            'mlook_factor '+ str(mlf)+'\n',
            'incidence_angle '+ str(inc)+'\n',
            'calibration_constant '+str(cc)+'\n',
            'pulse_bandwidth '+str(bw)+'\n',
            'lines '+ str(rows)+'\n',
            'samples '+str(cols)+'\n'
            
            ]

    calFactor = 1/np.sqrt(10**(cc/10))
    
    inFile = glob.glob(os.path.join(in_dir, 'data/calibrated/*/*sli*_hh_*.tif'))[0]
    S11 = read_rs2_tif(inFile)
    write_rst(os.path.join(base_out_dir, f's11.{ext}'),
              S11[:,:,0]*calFactor+1j*(S11[:,:,1]*calFactor),   
              driver=driver, mat=mat, cog=cog, ovr=ovr, comp=comp)
    del S11
    # write_s2_bin(out_file,data[:,:,0]*calFactor+1j*(data[:,:,1]*calFactor))
    
    inFile = glob.glob(os.path.join(in_dir, 'data/calibrated/*/*sli*_hv_*.tif'))[0]
    # data_xy = read_rs2_tif(inFile)
    S12 = read_rs2_tif(inFile)


    inFile = glob.glob(os.path.join(in_dir, 'data/calibrated/*/*sli*_vh_*.tif'))[0]
    # data_yx = read_rs2_tif(inFile)    
    S21 = read_rs2_tif(inFile)
    
    if recip:
        S12 = (S12 + S21)/2
        S21 = S12    
        
    write_rst(os.path.join(base_out_dir, f's12.{ext}'),
              S12[:,:,0]*calFactor+1j*(S12[:,:,1]*calFactor),   
              driver=driver, mat=mat, cog=cog, ovr=ovr, comp=comp)
    del S12
    write_rst(os.path.join(base_out_dir, f's21.{ext}'),
              S21[:,:,0]*calFactor+1j*(S21[:,:,1]*calFactor),   
              driver=driver, mat=mat, cog=cog, ovr=ovr, comp=comp)
    del S21    
    
    inFile = glob.glob(os.path.join(in_dir, 'data/calibrated/*/*sli*_vv_*.tif'))[0]
    S22 = read_rs2_tif(inFile)
    
    write_rst(os.path.join(base_out_dir, f's22.{ext}'),
              S22[:,:,0]*calFactor+1j*(S22[:,:,1]*calFactor),   
              driver=driver, mat=mat, cog=cog, ovr=ovr, comp=comp)
    del S22    
    
    with open(base_out_dir+'/multilook_info.txt', 'w+') as f:
        f.writelines(lines)
    f.close()

    if azlks == None and rglks == None:
        azlks = mlf
        rglks = 1

    # Matrix conversion if needed
    if mat not in ['S2', 'Sxy']:
        convert_S(base_out_dir, mat=mat, azlks=azlks, rglks=rglks, cf=1,
                  fmt=fmt, out_dir=final_out_dir, cog=cog, ovr=ovr, comp=comp)

        # Clean up temp directory
        if temp_dir:
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Warning: Could not delete temporary directory {temp_dir}: {e}")




    # if mat == 'S2':

    #     out_dir = os.path.join(in_dir,"S2")
    #     os.makedirs(out_dir,exist_ok=True)

    #     print("Considering S12 = S21")

    #     inFile = glob.glob(os.path.join(in_dir, 'data/calibrated/*/*sli*_hh_*.tif'))[0]
    #     data = read_rs2_tif(inFile)
    #     out_file = os.path.join(out_dir,'s11.bin')
    #     write_s2_bin(out_file,data[:,:,0]*calFactor+1j*(data[:,:,1]*calFactor))
    #     print("Saved file "+out_file)
        
    #     rows, cols, _ = data.shape

    #     inFile = glob.glob(os.path.join(in_dir, 'data/calibrated/*/*sli*_hv_*.tif'))[0]
    #     data_xy = read_rs2_tif(inFile)


    #     inFile = glob.glob(os.path.join(in_dir, 'data/calibrated/*/*sli*_vh_*.tif'))[0]
    #     data_yx = read_rs2_tif(inFile)

    #     data = (data_xy+data_yx)*0.5
    #     del data_xy,data_yx

    #     out_file = os.path.join(out_dir,'s12.bin')
        
    #     write_s2_bin(out_file,data[:,:,0]*calFactor+1j*(data[:,:,1]*calFactor))
    #     print("Saved file "+out_file)
    #     out_file = os.path.join(out_dir,'s21.bin')
    #     write_s2_bin(out_file,data[:,:,0]*calFactor+1j*(data[:,:,1]*calFactor))
    #     print("Saved file "+out_file)

    #     inFile = glob.glob(os.path.join(in_dir, 'data/calibrated/*/*sli*_vv_*.tif'))[0]
    #     data = read_rs2_tif(inFile)
    #     out_file = os.path.join(out_dir,'s22.bin')
    #     write_s2_bin(out_file,data[:,:,0]*calFactor+1j*(data[:,:,1]*calFactor))
    #     print("Saved file "+out_file)
        
    #     file = open(out_dir +'/config.txt',"w+")
    #     file.write('Nrow\n%d\n---------\nNcol\n%d\n---------\nPolarCase\nmonostatic\n---------\nPolarType\nfull'%(rows,cols))
    #     file.close() 
        
    #     with open(base_out_dir+'/multilook_info.txt', 'w+') as f:
    #         f.writelines(lines)
    #     f.close()
        
        
        
    # elif mat == 'T3':
    #     # print("Considering S12 = S21")
        
    #     inFile = glob.glob(os.path.join(in_dir, 'data/calibrated/*/*sli*_hh_*.tif'))[0]
    #     data = read_rs2_tif(inFile)
    #     s11 = data[:,:,0]*calFactor+1j*(data[:,:,1]*calFactor)

    #     inFile = glob.glob(os.path.join(in_dir, 'data/calibrated/*/*sli*_hv_*.tif'))[0]
    #     data_xy = read_rs2_tif(inFile)

    #     inFile = glob.glob(os.path.join(in_dir, 'data/calibrated/*/*sli*_vh_*.tif'))[0]
    #     data_yx = read_rs2_tif(inFile)
        
    #     # Symmetry assumption
    #     data = (data_xy+data_yx)*0.5
    #     del data_xy,data_yx
    #     s12 = data[:,:,0]*calFactor+1j*(data[:,:,1]*calFactor)

    #     inFile = glob.glob(os.path.join(in_dir, 'data/calibrated/*/*sli*_vv_*.tif'))[0]
    #     data = read_rs2_tif(inFile)
    #     s22 = data[:,:,0]*calFactor+1j*(data[:,:,1]*calFactor)
        
    #     # Kp- 3-D Pauli feature vector
    #     Kp = (1/np.sqrt(2))*np.array([s11+s22, s11-s22, 2*s12])

    #     del s11,s12,s22


    #     if azlks == None and rglks == None:
    #         azlks = mlf
    #         rglks = 1
                
    #     print(f'Using multi-look factor: azlks = {azlks}, rglks = {rglks}')

    #     # 3x3 Pauli Coherency Matrix elements
    #     T11 = mlook_arr(np.abs(Kp[0])**2,azlks,rglks).astype(np.float32)
    #     T22 = mlook_arr(np.abs(Kp[1])**2,azlks,rglks).astype(np.float32)
    #     T33 = mlook_arr(np.abs(Kp[2])**2,azlks,rglks).astype(np.float32)

    #     T12 = mlook_arr(Kp[0]*np.conj(Kp[1]),azlks,rglks).astype(np.complex64)
    #     T13 = mlook_arr(Kp[0]*np.conj(Kp[2]),azlks,rglks).astype(np.complex64)
    #     T23 = mlook_arr(Kp[1]*np.conj(Kp[2]),azlks,rglks).astype(np.complex64)

    #     del Kp
    #     T3Folder = os.path.join(in_dir,'T3')

    #     if not os.path.isdir(T3Folder):
    #         print("T3 folder does not exist. \nCreating folder {}".format(T3Folder))
    #         os.mkdir(T3Folder)
            
    #     # write_T3(np.dstack([T11,T12,T13,np.conjugate(T12),T22,T23,np.conjugate(T13),np.conjugate(T23),T33]),T3Folder)
    #     write_T3([np.real(T11),np.real(T12),np.imag(T12),np.real(T13),np.imag(T13),
    #               np.real(T22),np.real(T23),np.imag(T23),
    #               np.real(T33)],T3Folder)
        
    #     with open(T3Folder+'/multilook_info.txt', 'w+') as f:
    #         f.writelines(lines)
    #     f.close()
        
    # elif mat == 'C3':
    #     # print("Considering S12 = S21")

    #     inFile = glob.glob(os.path.join(in_dir, 'data/calibrated/*/*sli*_hh_*.tif'))[0]
    #     data = read_rs2_tif(inFile)
    #     s11 = data[:,:,0]*calFactor+1j*(data[:,:,1]*calFactor)

    #     inFile = glob.glob(os.path.join(in_dir, 'data/calibrated/*/*sli*_hv_*.tif'))[0]
    #     data_xy = read_rs2_tif(inFile)

    #     inFile = glob.glob(os.path.join(in_dir, 'data/calibrated/*/*sli*_vh_*.tif'))[0]
    #     data_yx = read_rs2_tif(inFile)
        
    #     # Symmetry assumption
    #     data = (data_xy+data_yx)*0.5
    #     del data_xy,data_yx
    #     s12 = data[:,:,0]*calFactor+1j*(data[:,:,1]*calFactor)

    #     inFile = glob.glob(os.path.join(in_dir, 'data/calibrated/*/*sli*_vv_*.tif'))[0]
    #     data = read_rs2_tif(inFile)
    #     s22 = data[:,:,0]*calFactor+1j*(data[:,:,1]*calFactor)

    #     # Kl- 3-D Lexicographic feature vector
    #     Kl = np.array([s11, np.sqrt(2)*s12, s22])
    #     del s11,s12,s22


    #     if azlks == None and rglks == None:
    #         azlks = mlf
    #         rglks = 1
                
    #     print(f'Using multi-look factor: azlks = {azlks}, rglks = {rglks}')

    #     # 3x3 COVARIANCE Matrix elements

    #     C11 = mlook_arr(np.abs(Kl[0])**2,azlks,rglks).astype(np.float32)
    #     C22 = mlook_arr(np.abs(Kl[1])**2,azlks,rglks).astype(np.float32)
    #     C33 = mlook_arr(np.abs(Kl[2])**2,azlks,rglks).astype(np.float32)

    #     C12 = mlook_arr(Kl[0]*np.conj(Kl[1]),azlks,rglks).astype(np.complex64)
    #     C13 = mlook_arr(Kl[0]*np.conj(Kl[2]),azlks,rglks).astype(np.complex64)
    #     C23 = mlook_arr(Kl[1]*np.conj(Kl[2]),azlks,rglks).astype(np.complex64)

    #     C3Folder = os.path.join(in_dir,'C3')

    #     if not os.path.isdir(C3Folder):
    #         print("C3 folder does not exist. \nCreating folder {}".format(C3Folder))
    #         os.mkdir(C3Folder)
        
    #     # write_C3(np.dstack([C11,C12,C13,np.conjugate(C12),C22,C23,np.conjugate(C13),np.conjugate(C23),C33]),C3Folder)
    #     write_C3([np.real(C11),np.real(C12),np.imag(C12),np.real(C13),np.imag(C13),
    #               np.real(C22),np.real(C23),np.imag(C23),
    #               np.real(C33)],C3Folder) 
        
    #     with open(C3Folder+'/multilook_info.txt', 'w+') as f:
    #         f.writelines(lines)
    #     f.close()
    # else:
    #     raise ValueError('Invalid matrix type. Valid types are "S2", "T3" and "C3"')

