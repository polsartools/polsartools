import numpy as np
from osgeo import gdal
gdal.UseExceptions()
import os
import xml.etree.ElementTree as ET
from polsartools.utils.utils import time_it,mlook_arr
from polsartools.utils.io_utils import write_T3, write_C3

def read_rs2_tif(file):
    ds = gdal.Open(file)
    band1 = ds.GetRasterBand(1).ReadAsArray()
    band2 = ds.GetRasterBand(2).ReadAsArray()
    ds=None
    return np.dstack((band1,band2))

def write_s2_bin(file,wdata):
    [cols, rows] = wdata.shape
    driver = gdal.GetDriverByName("ENVI")
    outdata = driver.Create(file, rows, cols, 1, gdal.GDT_CFloat32)
    outdata.SetDescription(file)
    outdata.GetRasterBand(1).WriteArray(wdata)
    outdata.FlushCache()

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
    outdata=None


@time_it
def rs2_fp(inFolder,matrixType='T3',
           azlks=8,rglks=2,outType='tif',
           type='sigma0', out_dir = None):
    """
    Process radarsat-2 image data and generate the specified matrix (S2, T3, or C3) from the input imagery files.

    This function reads radarsat-2 image data in the form of .tif files (HH, HV, VH, VV) from the input folder (`inFolder`) 
    and calculates either the S2, T3, or C3 matrix. The resulting matrix is then saved in a corresponding directory
    (`S2`, `T3`, or `C3`). The function uses lookup tables (`lutSigma.xml`, `lutGamma.xml`, `lutBeta.xml`) for scaling 
    the data based on the chosen `type` (sigma0, gamma0, or beta0). The processed data is written into binary files 
    in the output folder.

    Example Usage:
    --------------
    To process imagery and generate a T3 matrix:
    
    .. code-block:: python

        rs2_fp("/path/to/data", matrix="T3", type="sigma0")

    To process imagery and generate a C3 matrix:

    .. code-block:: python

        rs2_fp("/path/to/data", matrix="C3", type="beta0", azlks=10, rglks=3)
        
    Parameters:
    -----------
    inFolder : str
        Path to the folder containing the radar imagery files and the lookup tables (`lutSigma.xml`, `lutGamma.xml`, `lutBeta.xml`).
    
    matrixType : str, optional (default='T3')
        The type of matrix to generate. Can be:
        
        - 'S2' : Generates S2 matrix from the input imagery (S12 = S21 assumption).
        - 'T3' : Generates T3 matrix from the input imagery, based on Pauli decomposition.
        - 'C3' : Generates C3 matrix from the input imagery, based on Lexicographic decomposition.

    type : str, optional (default='sigma0')
        The type of radar cross-section to use for scaling. Available options:
        
        - 'sigma0' : Uses `lutSigma.xml` for scaling.
        - 'gamma0' : Uses `lutGamma.xml` for scaling.
        - 'beta0' : Uses `lutBeta.xml` for scaling.
    
    azlks : int, optional (default=8)
        The number of azimuth looks to apply during the C3/T3 processing.

    rglks : int, optional (default=2)
        The number of range looks to apply during the C3/T3processing.

    Raises:
    -------
    ValueError
        If an invalid `matrix` or `type` is provided, a `ValueError` will be raised with a descriptive message.
    
    FileNotFoundError
        If any of the required input files (such as the .tif imagery or the lookup tables) cannot be found in the `inFolder`.

    Notes:
    ------
    - The function assumes that the input imagery is stored as "imagery_HH.tif", "imagery_HV.tif", "imagery_VH.tif", and "imagery_VV.tif" in the `inFolder`.
    - The function uses the `read_rs2_tif` and `write_*` helper functions to read and write image data and binary files.
    - The generated matrices (S2, T3, or C3) are saved in subdirectories within `inFolder` named accordingly.
    """
    
    if type == 'sigma0':
        tree = ET.parse(os.path.join(inFolder,"lutSigma.xml"))
        root = tree.getroot()
        lut = root.find('gains').text.strip()
        lut = np.fromstring(lut, sep=' ')
    elif type == 'gamma0':
        tree = ET.parse(os.path.join(inFolder,"lutGamma.xml"))
        root = tree.getroot()
        lut = root.find('gains').text.strip()
        lut = np.fromstring(lut, sep=' ')
    elif type=='beta0':
        tree = ET.parse(os.path.join(inFolder,"lutBeta.xml"))
        root = tree.getroot()
        lut = root.find('gains').text.strip()
        lut = np.fromstring(lut, sep=' ')
    else:
        raise ValueError(f'Unknown type {type} \n Available types: sigma0,gamma0,beta0')

    if matrixType == 'S2':
        if out_dir is None:
            out_dir = os.path.join(inFolder,"S2")
        else:
            out_dir = os.path.join(out_dir,"S2")
            
        os.makedirs(out_dir,exist_ok=True)

        print("Considering S12 = S21")
        inFile = os.path.join(inFolder,"imagery_HH.tif")
        data = read_rs2_tif(inFile)
        if outType=='bin':
            out_file = os.path.join(out_dir,'s11.bin')
            write_rst(out_file,data[:,:,0]/lut+1j*(data[:,:,1]/lut),gdal.GDT_CFloat32)
        else:
            out_file = os.path.join(out_dir,'s11.tif')
            write_rst(out_file,data[:,:,0]/lut+1j*(data[:,:,1]/lut),gdal.GDT_CFloat32)
        print("Saved file "+out_file)
        
        rows,cols,_ = data.shape
        
        inFile = os.path.join(inFolder,"imagery_HV.tif")
        data_xy = read_rs2_tif(inFile)

        inFile = os.path.join(inFolder,"imagery_VH.tif")
        data_yx = read_rs2_tif(inFile)

        data = (data_xy+data_yx)*0.5
        del data_xy,data_yx

        if outType=='bin':
            out_file = os.path.join(out_dir,'s12.bin')
            write_rst(out_file,data[:,:,0]/lut+1j*(data[:,:,1]/lut),gdal.GDT_CFloat32)
            print("Saved file "+out_file)
            out_file = os.path.join(out_dir,'s21.bin')
            write_rst(out_file,data[:,:,0]/lut+1j*(data[:,:,1]/lut),gdal.GDT_CFloat32)
            print("Saved file "+out_file)
        else:
            out_file = os.path.join(out_dir,'s12.tif')
            write_rst(out_file,data[:,:,0]/lut+1j*(data[:,:,1]/lut),gdal.GDT_CFloat32)
            print("Saved file "+out_file)
            out_file = os.path.join(out_dir,'s21.tif')
            write_rst(out_file,data[:,:,0]/lut+1j*(data[:,:,1]/lut),gdal.GDT_CFloat32)
            print("Saved file "+out_file)


        inFile = os.path.join(inFolder,"imagery_VV.tif")
        data = read_rs2_tif(inFile)
        
        if outType=='bin':
            out_file = os.path.join(out_dir,'s22.bin')
            write_rst(out_file,data[:,:,0]/lut+1j*(data[:,:,1]/lut),gdal.GDT_CFloat32)
            print("Saved file "+out_file)
        else:
            out_file = os.path.join(out_dir,'s22.tif')
            write_rst(out_file,data[:,:,0]/lut+1j*(data[:,:,1]/lut),gdal.GDT_CFloat32)
            print("Saved file "+out_file)
        # out_file = os.path.join(out_dir,'s22.bin')
        # write_s2_bin(out_file,data[:,:,0]/lut+1j*(data[:,:,1]/lut))
        # print("Saved file "+out_file)
        
        file = open(out_dir +'/config.txt',"w+")
        file.write('Nrow\n%d\n---------\nNcol\n%d\n---------\nPolarCase\nmonostatic\n---------\nPolarType\nfull'%(rows,cols))
        file.close() 
        
    elif matrixType == 'T3':
        # print("Considering S12 = S21")
        # Kp- 3-D Pauli feature vector
        # Kp = (1/np.sqrt(2))*np.array([S2[0,0]+S2[1,1], S2[0,0]-S2[1,1], S2[1,0]])
        # Kp = (1/np.sqrt(2))*np.array([S2[0,0]+S2[1,1], S2[0,0]-S2[1,1], S2[0,1]])
        
        inFile = os.path.join(inFolder,"imagery_HH.tif")
        data = read_rs2_tif(inFile)
        s11 = data[:,:,0]/lut+1j*(data[:,:,1]/lut)

        inFile = os.path.join(inFolder,"imagery_HV.tif")
        data_xy = read_rs2_tif(inFile)

        inFile = os.path.join(inFolder,"imagery_VH.tif")
        data_yx = read_rs2_tif(inFile)
        # Symmetry assumption
        data = (data_xy+data_yx)*0.5
        del data_xy,data_yx
        s12 = data[:,:,0]/lut+1j*(data[:,:,1]/lut)

        inFile = os.path.join(inFolder,"imagery_VV.tif")
        data = read_rs2_tif(inFile)
        s22 = data[:,:,0]/lut+1j*(data[:,:,1]/lut)

        Kp = (1/np.sqrt(2))*np.array([s11+s22, s11-s22, 2*s12])

        del s11,s12,s22

        # 3x3 Pauli Coherency Matrix elements
        T11 = mlook_arr(np.abs(Kp[0])**2,azlks,rglks).astype(np.float32)
        T22 = mlook_arr(np.abs(Kp[1])**2,azlks,rglks).astype(np.float32)
        T33 = mlook_arr(np.abs(Kp[2])**2,azlks,rglks).astype(np.float32)

        T12 = mlook_arr(Kp[0]*np.conj(Kp[1]),azlks,rglks).astype(np.complex64)
        T13 = mlook_arr(Kp[0]*np.conj(Kp[2]),azlks,rglks).astype(np.complex64)
        T23 = mlook_arr(Kp[1]*np.conj(Kp[2]),azlks,rglks).astype(np.complex64)

        del Kp
        if out_dir is None:
            out_dir = os.path.join(inFolder,"T3")
        else:
            out_dir = os.path.join(out_dir,"T3")
        os.makedirs(out_dir,exist_ok=True)
        # T3Folder = os.path.join(inFolder,'T3')

        # if not os.path.isdir(T3Folder):
        #     print("T3 folder does not exist. \nCreating folder {}".format(T3Folder))
        #     os.mkdir(T3Folder)
            
        # write_T3(np.dstack([T11,T12,T13,np.conjugate(T12),T22,T23,np.conjugate(T13),np.conjugate(T23),T33]),T3Folder)
        write_T3([np.real(T11),np.real(T12),np.imag(T12),np.real(T13),np.imag(T13),
                  np.real(T22),np.real(T23),np.imag(T23),
                  np.real(T33)],out_dir)
        
        
    elif matrixType == 'C3':
        # print("Considering S12 = S21")
        inFile = os.path.join(inFolder,"imagery_HH.tif")
        data = read_rs2_tif(inFile)
        s11 = data[:,:,0]/lut+1j*(data[:,:,1]/lut)

        inFile = os.path.join(inFolder,"imagery_HV.tif")
        data_xy = read_rs2_tif(inFile)

        inFile = os.path.join(inFolder,"imagery_VH.tif")
        data_yx = read_rs2_tif(inFile)
        # Symmetry assumption
        data = (data_xy+data_yx)*0.5
        del data_xy,data_yx
        s12 = data[:,:,0]/lut+1j*(data[:,:,1]/lut)

        inFile = os.path.join(inFolder,"imagery_VV.tif")
        data = read_rs2_tif(inFile)
        s22 = data[:,:,0]/lut+1j*(data[:,:,1]/lut)

        # Kl- 3-D Lexicographic feature vector
        Kl = np.array([s11, np.sqrt(2)*s12, s22])
        del s11,s12,s22

        # 3x3 COVARIANCE Matrix elements

        C11 = mlook_arr(np.abs(Kl[0])**2,azlks,rglks).astype(np.float32)
        C22 = mlook_arr(np.abs(Kl[1])**2,azlks,rglks).astype(np.float32)
        C33 = mlook_arr(np.abs(Kl[2])**2,azlks,rglks).astype(np.float32)

        C12 = mlook_arr(Kl[0]*np.conj(Kl[1]),azlks,rglks).astype(np.complex64)
        C13 = mlook_arr(Kl[0]*np.conj(Kl[2]),azlks,rglks).astype(np.complex64)
        C23 = mlook_arr(Kl[1]*np.conj(Kl[2]),azlks,rglks).astype(np.complex64)

        if out_dir is None:
            out_dir = os.path.join(inFolder,"C3")
        else:
            out_dir = os.path.join(out_dir,"C3")
        os.makedirs(out_dir,exist_ok=True)
        
        # C3Folder = os.path.join(inFolder,'C3')

        # if not os.path.isdir(C3Folder):
        #     print("C3 folder does not exist. \nCreating folder {}".format(C3Folder))
        #     os.mkdir(C3Folder)
        
        # write_C3(np.dstack([C11,C12,C13,np.conjugate(C12),C22,C23,np.conjugate(C13),np.conjugate(C23),C33]),C3Folder)
        write_C3([np.real(C11),np.real(C12),np.imag(C12),np.real(C13),np.imag(C13),
                  np.real(C22),np.real(C23),np.imag(C23),
                  np.real(C33)],out_dir)
        
        
    else:
        raise ValueError('Invalid matrix type. Valid types are "S2", "T3" and "C3"')