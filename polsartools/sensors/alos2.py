
import numpy as np
from osgeo import gdal
import os, glob
import tempfile,shutil
from polsartools.utils.utils import time_it, mlook_arr
# from polsartools.utils.io_utils import write_T3, write_C3
from polsartools.preprocess.convert_S2 import convert_S
gdal.UseExceptions()
def read_bin(file):
    ds = gdal.Open(file)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    return arr

def read_a2(file):
    
    fp = open(file,mode='rb')
    fp.seek(232)
    ch = int(fp.read(4))
    # print(ch)
    
    fp.seek(236)
    nline = int(fp.read(8))
    # print(nline)
    fp.seek(248)
    npixel = int(fp.read(8))
    # print(npixel)
    
    nrec = 544 + npixel*8
    # print(nrec)
    fp.seek(720)
    data = np.frombuffer(fp.read(int(nrec * nline)), dtype='>f4')
    data = np.array(data).reshape(-1,int(nrec/4)) 
    # print(np.shape(data))
    
    data = data[:,int(544/4):int(nrec/4)] 
    slc = data[:,::2] + 1j*data[:,1::2]
    # print(np.shape(slc))
    del data
    
    return slc

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
    
    if cog:
        dataset.BuildOverviews("NEAREST", ovr)
        
    dataset.GetRasterBand(1).WriteArray(data)
    # outdata.GetRasterBand(1).SetNoDataValue(0)##if you want these values transparent
    dataset.FlushCache() ##saves to disk!!
    dataset = None
    if mat == 'S2' or mat == 'Sxy':
        print(f"Saved file: {out_file}")

@time_it    
def alos2_fbd_l11(in_dir,mat='C2', azlks=3,rglks=2,
                 fmt='tif', cog=False,ovr = [2, 4, 8, 16],comp=False,
                 out_dir=None,
                  cf_dB=-83):
    """
    Extracts the C2 matrix elements (C11, C22, and C12) from ALOS-2 Fine Beam Dual-Pol (FBD) CEOS data 
    and saves them into respective binary files.

    Example:
    --------
    >>> alos2_fbd_l11("path_to_folder", azlks=5, rglks=3)
    This will extract the C2 matrix elements from the ALOS-2 Fine Beam Dual-Pol data 
    in the specified folder and save them in the 'C2' directory.
    
    Parameters:
    -----------
    in_dir : str
        The path to the folder containing the ALOS-2 Fine Beam Dual-Pol CEOS data files.
    mat : str, optional (default = 'S2' or 'Sxy)
        Type of matrix to extract. Valid options: 'Sxy','C2', 'T2'.
    azlks : int, optional (default=3)
        The number of azimuth looks for multi-looking.

    rglks : int, optional (default=2)
        The number of range looks for multi-looking.
    
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
        
    
    hh_file = list(glob.glob(os.path.join(in_dir,'IMG-HH-*-FBDR1.1__A')) + \
        glob.glob(os.path.join(in_dir, 'IMG-HH-*-FBDR1.1__D')))[0]

    hv_file = list(glob.glob(os.path.join(in_dir,'IMG-HV-*-FBDR1.1__A')) + \
        glob.glob(os.path.join(in_dir, 'IMG-HV-*-FBDR1.1__D')))[0]

    calfac_linear = np.sqrt(10 ** ((cf_dB - 32) / 10))

    S11 = read_a2(hh_file).astype(np.complex64)*calfac_linear 
    write_a2_rst(os.path.join(base_out_dir, f's11.{ext}'),S11,   driver=driver, mat=mat, cog=cog, ovr=ovr, comp=comp)
    del S11
    S12 = read_a2(hv_file).astype(np.complex64)*calfac_linear 
    write_a2_rst(os.path.join(base_out_dir, f's12.{ext}'),S12,   driver=driver, mat=mat, cog=cog, ovr=ovr, comp=comp)
    del S12
    
    
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

    

    # if mat=='Sxy':
    #     if out_dir is None:
    #         SxyFolder = os.path.join(in_dir,os.path.basename(hh_file).split('-HH-')[1].split('1.1')[0],'Sxy')
    #     else:
    #         SxyFolder = out_dir
        
    #     os.makedirs(SxyFolder,exist_ok=True)
    #     if fmt=='bin':
    #         write_a2_rst( os.path.join(SxyFolder,'s11.bin'), S11,out_dtype=gdal.GDT_CFloat32, driver='ENVI')
    #         print(f"Saved file {SxyFolder}/s11.bin")
    #         write_a2_rst( os.path.join(SxyFolder,'s12.bin'), S12,out_dtype=gdal.GDT_CFloat32, driver='ENVI')
    #         print(f"Saved file {SxyFolder}/s12.bin")
    #     else:
    #         out_file = os.path.join(SxyFolder,'s11.tif')
    #         write_a2_rst(out_file, S11,cog=cog, ovr=ovr, comp=comp, out_dtype=gdal.GDT_CFloat32)
    #         print("Saved file "+out_file)
    #         out_file = os.path.join(SxyFolder,'s12.tif')
    #         write_a2_rst(out_file, S12,cog=cog, ovr=ovr, comp=comp, out_dtype=gdal.GDT_CFloat32)
    #         print("Saved file "+out_file)
        
        
    # else:
    #     C11 = mlook_arr(np.abs(S11)**2,azlks,rglks).astype(np.float32)
    #     C22 = mlook_arr(np.abs(S12)**2,azlks,rglks).astype(np.float32)
        
    #     C12 = mlook_arr(S11*np.conjugate(S12),azlks,rglks).astype(np.complex64)
    #     rows,cols = C11.shape
        
    #     if out_dir is None:
    #         in_dir = os.path.dirname(hh_file)   
    #         C2Folder = os.path.join(in_dir,os.path.basename(hh_file).split('-HH-')[1].split('1.1')[0],'C2')
    #     else:
    #         C2Folder = out_dir
    #     del S11,S12
        
    #     os.makedirs(C2Folder,exist_ok=True)
        
    #     if fmt=='bin':
        
    #         write_a2_rst( os.path.join(C2Folder,'C11.bin'), C11, driver='ENVI')
    #         print(f"Saved file {C2Folder}/C11.bin")
    #         del C11  
    #         write_a2_rst( os.path.join(C2Folder,'C22.bin'), C22,driver='ENVI')
    #         print(f"Saved file {C2Folder}/C22.bin")
    #         del C22
    #         write_a2_rst( os.path.join(C2Folder,'C12_real.bin'), np.real(C12), driver='ENVI')
    #         print(f"Saved file {C2Folder}/C12_real.bin")
    #         write_a2_rst( os.path.join(C2Folder,'C12_imag.bin'), np.imag(C12), driver='ENVI')
    #         print(f"Saved file {C2Folder}/C12_imag.bin")
    #         del C12

    #     elif fmt=='tif':
    #         out_file = os.path.join(C2Folder,'C11.tif')
    #         write_a2_rst(out_file, C11,cog=cog, ovr=ovr, comp=comp)
    #         print("Saved file "+out_file)
    #         del C11  
    #         out_file = os.path.join(C2Folder,'C22.tif')
    #         write_a2_rst(out_file, C22,cog=cog, ovr=ovr, comp=comp)
    #         print("Saved file "+out_file)
    #         del C22
    #         out_file = os.path.join(C2Folder,'C12_real.tif')
    #         write_a2_rst(out_file, np.real(C12),cog=cog, ovr=ovr, comp=comp)
    #         print("Saved file "+out_file)
    #         out_file = os.path.join(C2Folder,'C12_imag.tif')
    #         write_a2_rst(out_file, np.imag(C12),cog=cog, ovr=ovr, comp=comp)
    #         print("Saved file "+out_file)
    #         del C12        

    #     file = open(C2Folder +'/config.txt',"w+")
    #     file.write('Nrow\n%d\n---------\nNcol\n%d\n---------\nPolarCase\nmonostatic\n---------\nPolarType\npp1'%(rows,cols))
    #     file.close()  
#################################################################################################

@time_it    
def alos2_hbq_l11(in_dir,mat='T3', azlks=8,rglks=4,
                  fmt='tif', cog=False,ovr = [2, 4, 8, 16],comp=False,
                  out_dir=None,
                  recip=False,cf_dB=-83):

    """
    Extracts single look S2 or Multi-look T3/C3 matrix elements from ALOS-2 Quad-Pol (HBQ) CEOS data 
    and saves them into respective binary files.

    Example:
    --------
    >>> alos2_hbq_l11("path_to_folder", azlks=5, rglks=3)
    This will extract the T3 matrix elements from the ALOS-2 Fine Beam Dual-Pol data 
    in the specified folder and save them in the 'C2' directory.
    
    Parameters:
    -----------
    in_dir : str
        The path to the folder containing the ALOS-2 Quad-Pol (HBQ) CEOS data folder.
    
    mat : str, optional (default='T3')
        Type of matrix to extract. Valid options: 'S2',  'C4, 'C3', 'T4', 
        'T3', 'C2HX', 'C2VX', 'C2HV','T2HV'
        
    azlks : int, optional (default=8)
        The number of azimuth looks for multi-looking.

    rglks : int, optional (default=4)
        The number of range looks for multi-looking.

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
    cf_dB : float, optional (default=-83)
        The calibration factor in dB used to scale the raw radar data. It is applied to 
        the HH and HV polarization data before matrix computation.

    Returns:
    --------
    None
        The function does not return any value. Instead, it creates a folders named 'S2` or 'C3` or 'T3` 
        (depending on the chosen matrix) and saves the corresponding binary files.

    """
    
    hh_file = list(glob.glob(os.path.join(in_dir,'IMG-HH-*-HBQR1.1__A')) + \
        glob.glob(os.path.join(in_dir, 'IMG-HH-*-HBQR1.1__D')))[0]

    hv_file = list(glob.glob(os.path.join(in_dir,'IMG-HV-*-HBQR1.1__A')) + \
        glob.glob(os.path.join(in_dir, 'IMG-HV-*-HBQR1.1__D')))[0]


    vh_file = list(glob.glob(os.path.join(in_dir,'IMG-VH-*-HBQR1.1__A')) + \
        glob.glob(os.path.join(in_dir, 'IMG-VH-*-HBQR1.1__D')))[0]

    vv_file = list(glob.glob(os.path.join(in_dir,'IMG-VV-*-HBQR1.1__A')) + \
        glob.glob(os.path.join(in_dir, 'IMG-VV-*-HBQR1.1__D')))[0]

    valid_full_pol = ['S2', 'C4', 'C3', 'T4', 'T3', 'C2HX', 'C2VX', 'C2HV', 'T2HV']
    valid_matrices = valid_full_pol

    if mat not in valid_matrices:
        raise ValueError(f"Invalid matrix type '{mat}'. \n Supported types are:\n"
                        f"  Full-pol: {sorted(valid_full_pol)}")



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



    calfac_linear = np.sqrt(10 ** ((cf_dB - 32) / 10))
    
    S11 = read_a2(hh_file).astype(np.complex64)*calfac_linear 
    S21 = read_a2(hv_file).astype(np.complex64)*calfac_linear 
    S12 = read_a2(vh_file).astype(np.complex64)*calfac_linear 
    S22 = read_a2(vv_file).astype(np.complex64)*calfac_linear 
    
    if recip:
        S12 = (S12 + S21)/2
        S21 = S12
    
    
    write_a2_rst(os.path.join(base_out_dir, f's11.{ext}'),S11,   driver=driver, mat=mat, cog=cog, ovr=ovr, comp=comp)
    write_a2_rst(os.path.join(base_out_dir, f's12.{ext}'),S12,   driver=driver, mat=mat, cog=cog, ovr=ovr, comp=comp)
    write_a2_rst(os.path.join(base_out_dir, f's21.{ext}'),S21,  driver=driver, mat=mat, cog=cog, ovr=ovr, comp=comp)
    write_a2_rst(os.path.join(base_out_dir, f's22.{ext}'),S22,  driver=driver, mat=mat, cog=cog, ovr=ovr, comp=comp)

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



    # if mat=='S2':
    #     rows,cols = S11.shape

    #     if out_dir is None:
    #         out_dir = os.path.join(in_dir,'S2')
    #         os.makedirs(out_dir,exist_ok=True)
    #     else:
    #         os.makedirs(out_dir,exist_ok=True)

    #     if fmt=='bin':        
    #         out_file = os.path.join(out_dir,'s11.bin')
    #         write_a2_rst(out_file,S11,out_dtype=gdal.GDT_CFloat32, driver='ENVI')
    #         print("Saved file "+out_file)

    #         out_file = os.path.join(out_dir,'s12.bin')
    #         write_a2_rst(out_file,S12,out_dtype=gdal.GDT_CFloat32, driver='ENVI')
    #         print("Saved file "+out_file)
            
    #         out_file = os.path.join(out_dir,'s21.bin')
    #         write_a2_rst(out_file,S21,out_dtype=gdal.GDT_CFloat32, driver='ENVI')
    #         print("Saved file "+out_file)            
            
    #         out_file = os.path.join(out_dir,'s22.bin')
    #         write_a2_rst(out_file,S22,out_dtype=gdal.GDT_CFloat32, driver='ENVI')
    #         print("Saved file "+out_file)
            
    #         file = open(out_dir +'/config.txt',"w+")
    #         file.write('Nrow\n%d\n---------\nNcol\n%d\n---------\nPolarCase\nmonostatic\n---------\nPolarType\nfull'%(rows,cols))
    #         file.close() 
    #     else:
    #         out_file = os.path.join(out_dir,'s11.tif')
    #         write_a2_rst(out_file, S11,cog=cog, ovr=ovr, comp=comp, out_dtype=gdal.GDT_CFloat32)
    #         print("Saved file "+out_file)

    #         out_file = os.path.join(out_dir,'s12.tif')
    #         write_a2_rst(out_file, S12,cog=cog, ovr=ovr, comp=comp, out_dtype=gdal.GDT_CFloat32)
    #         print("Saved file "+out_file)
            
    #         out_file = os.path.join(out_dir,'s21.tif')
    #         write_a2_rst(out_file, S21,cog=cog, ovr=ovr, comp=comp, out_dtype=gdal.GDT_CFloat32)
    #         print("Saved file "+out_file)            
            
    #         out_file = os.path.join(out_dir,'s22.tif')
    #         write_a2_rst(out_file, S22,cog=cog, ovr=ovr, comp=comp, out_dtype=gdal.GDT_CFloat32)
    #         print("Saved file "+out_file)
            
    #         file = open(out_dir +'/config.txt',"w+")
    #         file.write('Nrow\n%d\n---------\nNcol\n%d\n---------\nPolarCase\nmonostatic\n---------\nPolarType\nfull'%(rows,cols))
    #         file.close()
        
        
        
        
    # elif mat=='T3':
    #     # 3x1 Pauli Coherency Matrix
    #     Kp = (1/np.sqrt(2))*np.array([S11+S22, S11-S22, S12+S21])

    #     del S11,S12,S21,S22

    #     # 3x3 Pauli Coherency Matrix elements
    #     T11 = mlook_arr(np.abs(Kp[0])**2,azlks,rglks).astype(np.float32)
    #     T22 = mlook_arr(np.abs(Kp[1])**2,azlks,rglks).astype(np.float32)
    #     T33 = mlook_arr(np.abs(Kp[2])**2,azlks,rglks).astype(np.float32)

    #     T12 = mlook_arr(Kp[0]*np.conj(Kp[1]),azlks,rglks).astype(np.complex64)
    #     T13 = mlook_arr(Kp[0]*np.conj(Kp[2]),azlks,rglks).astype(np.complex64)
    #     T23 = mlook_arr(Kp[1]*np.conj(Kp[2]),azlks,rglks).astype(np.complex64)

    #     del Kp
        
        
    #     if out_dir is None:
    #         T3Folder = os.path.join(in_dir,'T3')
    #         os.makedirs(T3Folder,exist_ok=True)
    #     else:
    #         T3Folder = out_dir
    #         os.makedirs(T3Folder,exist_ok=True)
            
            
    #     # write_T3(np.dstack([T11,T12,T13,np.conjugate(T12),T22,T23,np.conjugate(T13),np.conjugate(T23),T33]),T3Folder)
    #     write_T3([np.real(T11),np.real(T12),np.imag(T12),np.real(T13),np.imag(T13),
    #               np.real(T22),np.real(T23),np.imag(T23),
    #               np.real(T33)],T3Folder,fmt)
    # elif mat=='C3':
    #     # Kl- 3-D Lexicographic feature vector
    #     Kl = np.array([S11, np.sqrt(2)*0.5*(S12+S21), S22])
    #     del S11, S12, S21, S22

    #     # 3x3 COVARIANCE Matrix elements

    #     C11 = mlook_arr(np.abs(Kl[0])**2,azlks,rglks).astype(np.float32)
    #     C22 = mlook_arr(np.abs(Kl[1])**2,azlks,rglks).astype(np.float32)
    #     C33 = mlook_arr(np.abs(Kl[2])**2,azlks,rglks).astype(np.float32)

    #     C12 = mlook_arr(Kl[0]*np.conj(Kl[1]),azlks,rglks).astype(np.complex64)
    #     C13 = mlook_arr(Kl[0]*np.conj(Kl[2]),azlks,rglks).astype(np.complex64)
    #     C23 = mlook_arr(Kl[1]*np.conj(Kl[2]),azlks,rglks).astype(np.complex64)

    #     if out_dir is None:
    #         C3Folder = os.path.join(in_dir,'C3')
    #         os.makedirs(C3Folder,exist_ok=True)
    #     else:
    #         C3Folder = out_dir
    #         os.makedirs(out_dir,exist_ok=True)
        
        
    #     # write_C3(np.dstack([C11,C12,C13,np.conjugate(C12),C22,C23,np.conjugate(C13),np.conjugate(C23),C33]),C3Folder)
    #     write_C3([np.real(C11),np.real(C12),np.imag(C12),np.real(C13),np.imag(C13),
    #               np.real(C22),np.real(C23),np.imag(C23),
    #               np.real(C33)],C3Folder,fmt)
        
        
    # else:
    #     raise ValueError('Invalid matrix type. Valid types are "S2", "T3" and "C3"')
        