import numpy as np
from osgeo import gdal
gdal.UseExceptions()
import os,tempfile,shutil
import xml.etree.ElementTree as ET
from polsartools.utils.utils import time_it
# from polsartools.utils.io_utils import write_T3, write_C3
from polsartools.preprocess.convert_S2 import convert_S

def read_rs2_tif(file):
    ds = gdal.Open(file)
    band1 = ds.GetRasterBand(1).ReadAsArray()
    band2 = ds.GetRasterBand(2).ReadAsArray()
    ds=None
    return np.dstack((band1,band2))

# def write_s2_bin(file,wdata):
#     [cols, rows] = wdata.shape
#     driver = gdal.GetDriverByName("ENVI")
#     outdata = driver.Create(file, rows, cols, 1, gdal.GDT_CFloat32)
#     outdata.SetDescription(file)
#     outdata.GetRasterBand(1).WriteArray(wdata)
#     outdata.FlushCache()

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
        
    dataset.GetRasterBand(1).WriteArray(data)
    # outdata.GetRasterBand(1).SetNoDataValue(0)##if you want these values transparent
    dataset.FlushCache() ##saves to disk!!
    
    if cog:
        dataset.BuildOverviews("NEAREST", ovr)
    
    dataset = None
    if mat == 'S2' or mat == 'Sxy':
        print(f"Saved file: {out_file}")

# def write_rst(file,wdata,dtype,cog=False, ovr=[2, 4, 8, 16], comp=False):
#     [cols, rows] = wdata.shape
#     if '.bin' in file:
#         driver = gdal.GetDriverByName("ENVI")
#         outdata = driver.Create(file, rows, cols, 1, dtype)
#     else:
#         driver = gdal.GetDriverByName("GTiff")
#         options = ['BIGTIFF=IF_SAFER']
#         if comp:
#         # options += ['COMPRESS=DEFLATE', 'PREDICTOR=2', 'ZLEVEL=9']
#             options += ['COMPRESS=LZW']
#         if cog:
#             options += ['TILED=YES', 'BLOCKXSIZE=512', 'BLOCKYSIZE=512']
#         outdata = driver.Create(file, rows, cols, 1, dtype, options)
    

#     outdata.SetDescription(file)
#     outdata.GetRasterBand(1).WriteArray(wdata)
#     outdata.FlushCache() 
#     outdata=None


@time_it
def rs2_fp(in_dir,mat='T3',
           azlks=8,rglks=2,fmt='tif',
            cog=False,ovr = [2, 4, 8, 16],comp=False,
           bsc='sigma0', out_dir = None,
           recip=False,
           ):
    """
    Process radarsat-2 image data and generate the specified matrix (S2, T3, or C3) from the input imagery files.

    This function reads radarsat-2 image data in the form of .tif files (HH, HV, VH, VV) from the input folder (`in_dir`) 
    and calculates either the S2, T3, or C3 matrix. The resulting matrix is then saved in a corresponding directory
    (`S2`, `T3`, or `C3`). The function uses lookup tables (`lutSigma.xml`, `lutGamma.xml`, `lutBeta.xml`) for scaling 
    the data based on the chosen backscatter coefficient `bsc` (sigma0, gamma0, or beta0). The processed data is written into binary files 
    in the output folder.

    Example Usage:
    --------------
    To process imagery and generate a T3 matrix:
    
    .. code-block:: python

        rs2_fp("/path/to/data", mat="T3", bsc="sigma0")

    To process imagery and generate a C3 matrix:

    .. code-block:: python

        rs2_fp("/path/to/data", mat="C3", bsc="beta0", azlks=10, rglks=3)
        
    Parameters:
    -----------
    in_dir : str
        Path to the folder containing the Radarsat-2 files and the lookup tables (`lutSigma.xml`, `lutGamma.xml`, `lutBeta.xml`).
    
    mat : str, optional (default='T3')
        Type of matrix to extract. Valid options: 'S2',  'C4, 'C3', 'T4', 
        'T3', 'C2HX', 'C2VX', 'C2HV','T2HV'
    
    azlks : int, optional (default=8)
        The number of azimuth looks to apply during the C/T processing.

    rglks : int, optional (default=2)
        The number of range looks to apply during the C/Tprocessing.
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

    bsc : str, optional (default='sigma0')
        The type of radar cross-section to use for scaling. Available options:
        
        - 'sigma0' : Uses `lutSigma.xml` for scaling.
        - 'gamma0' : Uses `lutGamma.xml` for scaling.
        - 'beta0' : Uses `lutBeta.xml` for scaling.
        
    out_dir : str or None, optional (default=None)
        Directory to save output files. If None, a folder named after the matrix type will be created
        in the same location as the input file.
        
    recip : bool, optional (default=False)
        If True, scattering matrix reciprocal symmetry is applied, i.e, S_HV = S_VH.
                
    """
    valid_full_pol = ['S2', 'C4', 'C3', 'T4', 'T3', 'C2HX', 'C2VX', 'C2HV', 'T2HV']
    # valid_dual_pol = ['Sxy', 'C2', 'T2']
    valid_matrices = valid_full_pol
    
    if mat not in valid_matrices:
        raise ValueError(f"Invalid matrix type '{mat}'. \n Supported types are:\n"
                        f"  Full-pol: {sorted(valid_full_pol)}\n")
    
    if bsc == 'sigma0':
        tree = ET.parse(os.path.join(in_dir,"lutSigma.xml"))
        root = tree.getroot()
        lut = root.find('gains').text.strip()
        lut = np.fromstring(lut, sep=' ')
    elif bsc == 'gamma0':
        tree = ET.parse(os.path.join(in_dir,"lutGamma.xml"))
        root = tree.getroot()
        lut = root.find('gains').text.strip()
        lut = np.fromstring(lut, sep=' ')
    elif bsc=='beta0':
        tree = ET.parse(os.path.join(in_dir,"lutBeta.xml"))
        root = tree.getroot()
        lut = root.find('gains').text.strip()
        lut = np.fromstring(lut, sep=' ')
    else:
        raise ValueError(f'Unknown type {bsc} \n Available bsc: sigma0,gamma0,beta0')

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

    inFile = os.path.join(in_dir,"imagery_HH.tif")
    S11 = read_rs2_tif(inFile)
    write_rst(os.path.join(base_out_dir, f's11.{ext}'),
              S11[:,:,0]/lut+1j*(S11[:,:,1]/lut),   
              driver=driver, mat=mat, cog=cog, ovr=ovr, comp=comp)
    
    del S11
    
    inFile = os.path.join(in_dir,"imagery_HV.tif")
    S12 = read_rs2_tif(inFile)

    inFile = os.path.join(in_dir,"imagery_VH.tif")
    S21 = read_rs2_tif(inFile)
    
    if recip:
        S12 = (S12 + S21)/2
        S21 = S12

    write_rst(os.path.join(base_out_dir, f's12.{ext}'),
              S12[:,:,0]/lut+1j*(S12[:,:,1]/lut),   
              driver=driver, mat=mat, cog=cog, ovr=ovr, comp=comp)
    del S12
    write_rst(os.path.join(base_out_dir, f's21.{ext}'),
              S21[:,:,0]/lut+1j*(S21[:,:,1]/lut),   
              driver=driver, mat=mat, cog=cog, ovr=ovr, comp=comp)
    del S21
    
    inFile = os.path.join(in_dir,"imagery_VV.tif")
    S22 = read_rs2_tif(inFile)
    write_rst(os.path.join(base_out_dir, f's22.{ext}'),
              S22[:,:,0]/lut+1j*(S22[:,:,1]/lut),   
              driver=driver, mat=mat, cog=cog, ovr=ovr, comp=comp)
    del S22
    
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
    #     if out_dir is None:
    #         out_dir = os.path.join(in_dir,"S2")
    #     else:
    #         out_dir = os.path.join(out_dir,"S2")
            
    #     os.makedirs(out_dir,exist_ok=True)

    #     print("Considering S12 = S21")
    #     inFile = os.path.join(in_dir,"imagery_HH.tif")
    #     data = read_rs2_tif(inFile)
    #     if fmt=='bin':
    #         out_file = os.path.join(out_dir,'s11.bin')
    #         write_rst(out_file,data[:,:,0]/lut+1j*(data[:,:,1]/lut),gdal.GDT_CFloat32)
    #     else:
    #         out_file = os.path.join(out_dir,'s11.tif')
    #         write_rst(out_file,data[:,:,0]/lut+1j*(data[:,:,1]/lut),gdal.GDT_CFloat32,cog= cog, ovr=ovr, comp=comp)
    #     print("Saved file "+out_file)
        
    #     rows,cols,_ = data.shape
        
    #     inFile = os.path.join(in_dir,"imagery_HV.tif")
    #     data_xy = read_rs2_tif(inFile)

    #     inFile = os.path.join(in_dir,"imagery_VH.tif")
    #     data_yx = read_rs2_tif(inFile)

    #     if recip:
    #         data = (data_xy+data_yx)*0.5
    #         del data_xy,data_yx

    #         if fmt=='bin':
    #             out_file = os.path.join(out_dir,'s12.bin')
    #             write_rst(out_file,data[:,:,0]/lut+1j*(data[:,:,1]/lut),gdal.GDT_CFloat32)
    #             print("Saved file "+out_file)
    #             out_file = os.path.join(out_dir,'s21.bin')
    #             write_rst(out_file,data[:,:,0]/lut+1j*(data[:,:,1]/lut),gdal.GDT_CFloat32,cog= cog, ovr=ovr, comp=comp)
    #             print("Saved file "+out_file)
    #         else:
    #             out_file = os.path.join(out_dir,'s12.tif')
    #             write_rst(out_file,data[:,:,0]/lut+1j*(data[:,:,1]/lut),gdal.GDT_CFloat32)
    #             print("Saved file "+out_file)
    #             out_file = os.path.join(out_dir,'s21.tif')
    #             write_rst(out_file,data[:,:,0]/lut+1j*(data[:,:,1]/lut),gdal.GDT_CFloat32,cog= cog, ovr=ovr, comp=comp)
    #             print("Saved file "+out_file)
    #     else:
    #         if fmt=='bin':
    #             out_file = os.path.join(out_dir,'s12.bin')
    #             write_rst(out_file,data_xy[:,:,0]/lut+1j*(data_xy[:,:,1]/lut),gdal.GDT_CFloat32)
    #             print("Saved file "+out_file)
    #             out_file = os.path.join(out_dir,'s21.bin')
    #             write_rst(out_file,data_yx[:,:,0]/lut+1j*(data_yx[:,:,1]/lut),gdal.GDT_CFloat32,cog= cog, ovr=ovr, comp=comp)
    #             print("Saved file "+out_file)
    #         else:
    #             out_file = os.path.join(out_dir,'s12.tif')
    #             write_rst(out_file,data_xy[:,:,0]/lut+1j*(data_xy[:,:,1]/lut),gdal.GDT_CFloat32)
    #             print("Saved file "+out_file)
    #             out_file = os.path.join(out_dir,'s21.tif')
    #             write_rst(out_file,data_yx[:,:,0]/lut+1j*(data_yx[:,:,1]/lut),gdal.GDT_CFloat32,cog= cog, ovr=ovr, comp=comp)
    #             print("Saved file "+out_file)

    #     inFile = os.path.join(in_dir,"imagery_VV.tif")
    #     data = read_rs2_tif(inFile)
        
    #     if fmt=='bin':
    #         out_file = os.path.join(out_dir,'s22.bin')
    #         write_rst(out_file,data[:,:,0]/lut+1j*(data[:,:,1]/lut),gdal.GDT_CFloat32)
    #         print("Saved file "+out_file)
    #     else:
    #         out_file = os.path.join(out_dir,'s22.tif')
    #         write_rst(out_file,data[:,:,0]/lut+1j*(data[:,:,1]/lut),gdal.GDT_CFloat32,cog= cog, ovr=ovr, comp=comp)
    #         print("Saved file "+out_file)
    #     # out_file = os.path.join(out_dir,'s22.bin')
    #     # write_s2_bin(out_file,data[:,:,0]/lut+1j*(data[:,:,1]/lut))
    #     # print("Saved file "+out_file)
        
    #     file = open(out_dir +'/config.txt',"w+")
    #     file.write('Nrow\n%d\n---------\nNcol\n%d\n---------\nPolarCase\nmonostatic\n---------\nPolarType\nfull'%(rows,cols))
    #     file.close() 
        
    # elif mat == 'T3':
    #     # print("Considering S12 = S21")
    #     # Kp- 3-D Pauli feature vector
    #     # Kp = (1/np.sqrt(2))*np.array([S2[0,0]+S2[1,1], S2[0,0]-S2[1,1], S2[1,0]])
    #     # Kp = (1/np.sqrt(2))*np.array([S2[0,0]+S2[1,1], S2[0,0]-S2[1,1], S2[0,1]])
        
    #     inFile = os.path.join(in_dir,"imagery_HH.tif")
    #     data = read_rs2_tif(inFile)
    #     s11 = data[:,:,0]/lut+1j*(data[:,:,1]/lut)

    #     inFile = os.path.join(in_dir,"imagery_HV.tif")
    #     data_xy = read_rs2_tif(inFile)

    #     inFile = os.path.join(in_dir,"imagery_VH.tif")
    #     data_yx = read_rs2_tif(inFile)
    #     # Symmetry assumption
    #     data = (data_xy+data_yx)*0.5
    #     del data_xy,data_yx
    #     s12 = data[:,:,0]/lut+1j*(data[:,:,1]/lut)

    #     inFile = os.path.join(in_dir,"imagery_VV.tif")
    #     data = read_rs2_tif(inFile)
    #     s22 = data[:,:,0]/lut+1j*(data[:,:,1]/lut)

    #     Kp = (1/np.sqrt(2))*np.array([s11+s22, s11-s22, 2*s12])

    #     del s11,s12,s22

    #     # 3x3 Pauli Coherency Matrix elements
    #     T11 = mlook_arr(np.abs(Kp[0])**2,azlks,rglks).astype(np.float32)
    #     T22 = mlook_arr(np.abs(Kp[1])**2,azlks,rglks).astype(np.float32)
    #     T33 = mlook_arr(np.abs(Kp[2])**2,azlks,rglks).astype(np.float32)

    #     T12 = mlook_arr(Kp[0]*np.conj(Kp[1]),azlks,rglks).astype(np.complex64)
    #     T13 = mlook_arr(Kp[0]*np.conj(Kp[2]),azlks,rglks).astype(np.complex64)
    #     T23 = mlook_arr(Kp[1]*np.conj(Kp[2]),azlks,rglks).astype(np.complex64)

    #     del Kp
    #     if out_dir is None:
    #         out_dir = os.path.join(in_dir,"T3")
    #     else:
    #         out_dir = os.path.join(out_dir,"T3")
    #     os.makedirs(out_dir,exist_ok=True)
    #     # T3Folder = os.path.join(in_dir,'T3')

    #     # if not os.path.isdir(T3Folder):
    #     #     print("T3 folder does not exist. \nCreating folder {}".format(T3Folder))
    #     #     os.mkdir(T3Folder)
            
    #     # write_T3(np.dstack([T11,T12,T13,np.conjugate(T12),T22,T23,np.conjugate(T13),np.conjugate(T23),T33]),T3Folder)
    #     write_T3([np.real(T11),np.real(T12),np.imag(T12),np.real(T13),np.imag(T13),
    #               np.real(T22),np.real(T23),np.imag(T23),
    #               np.real(T33)],out_dir)
        
        
    # elif mat == 'C3':
    #     # print("Considering S12 = S21")
    #     inFile = os.path.join(in_dir,"imagery_HH.tif")
    #     data = read_rs2_tif(inFile)
    #     s11 = data[:,:,0]/lut+1j*(data[:,:,1]/lut)

    #     inFile = os.path.join(in_dir,"imagery_HV.tif")
    #     data_xy = read_rs2_tif(inFile)

    #     inFile = os.path.join(in_dir,"imagery_VH.tif")
    #     data_yx = read_rs2_tif(inFile)
    #     # Symmetry assumption
    #     data = (data_xy+data_yx)*0.5
    #     del data_xy,data_yx
    #     s12 = data[:,:,0]/lut+1j*(data[:,:,1]/lut)

    #     inFile = os.path.join(in_dir,"imagery_VV.tif")
    #     data = read_rs2_tif(inFile)
    #     s22 = data[:,:,0]/lut+1j*(data[:,:,1]/lut)

    #     # Kl- 3-D Lexicographic feature vector
    #     Kl = np.array([s11, np.sqrt(2)*s12, s22])
    #     del s11,s12,s22

    #     # 3x3 COVARIANCE Matrix elements

    #     C11 = mlook_arr(np.abs(Kl[0])**2,azlks,rglks).astype(np.float32)
    #     C22 = mlook_arr(np.abs(Kl[1])**2,azlks,rglks).astype(np.float32)
    #     C33 = mlook_arr(np.abs(Kl[2])**2,azlks,rglks).astype(np.float32)

    #     C12 = mlook_arr(Kl[0]*np.conj(Kl[1]),azlks,rglks).astype(np.complex64)
    #     C13 = mlook_arr(Kl[0]*np.conj(Kl[2]),azlks,rglks).astype(np.complex64)
    #     C23 = mlook_arr(Kl[1]*np.conj(Kl[2]),azlks,rglks).astype(np.complex64)

    #     if out_dir is None:
    #         out_dir = os.path.join(in_dir,"C3")
    #     else:
    #         out_dir = os.path.join(out_dir,"C3")
    #     os.makedirs(out_dir,exist_ok=True)
        
    #     # C3Folder = os.path.join(in_dir,'C3')

    #     # if not os.path.isdir(C3Folder):
    #     #     print("C3 folder does not exist. \nCreating folder {}".format(C3Folder))
    #     #     os.mkdir(C3Folder)
        
    #     # write_C3(np.dstack([C11,C12,C13,np.conjugate(C12),C22,C23,np.conjugate(C13),np.conjugate(C23),C33]),C3Folder)
    #     write_C3([np.real(C11),np.real(C12),np.imag(C12),np.real(C13),np.imag(C13),
    #               np.real(C22),np.real(C23),np.imag(C23),
    #               np.real(C33)],out_dir)
        
        
    # else:
    #     raise ValueError('Invalid matrix type. Valid types are "S2", "T3" and "C3"')