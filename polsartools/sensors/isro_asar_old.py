import numpy as np
from osgeo import gdal
import os,h5py,glob
import xml.etree.ElementTree as ET
from polsartools.utils.utils import time_it, mlook_arr
from polsartools.utils.io_utils import write_T3, write_C3,write_C4,write_s2_bin
from polsartools.utils.geo_utils import geocode_grid, intp_grid, update_vrt, write_latlon
from polsartools.utils.proc_utils import process_chunks_parallel
from polsartools.utils.utils import conv2d,time_it


def asar_geo_T3(inFile,azlks,rglks,cc_linear):
    print("Considering S12 = S21")
           
    try:
        h5File = h5py.File(inFile,"r")
    except:
        raise('Invalid .h5 file !!')
    
    
    h5File = h5py.File(inFile,"r")
    if '/science/LSAR' in h5File:
        freq_band = 'L'            
        print("Detected L-band data ")
    elif '/science/SSAR' in h5File:
        freq_band = 'S'
        print("Detected S-band data")
    else:
        print("Neither LSAR nor SSAR data found in the file.")
        h5File.close()
        return
    
    S11 = np.array(h5File[f'/science/{freq_band}SAR/RSLC/swaths/frequencyA/HH'])/cc_linear
    S12 = np.array(h5File[f'/science/{freq_band}SAR/RSLC/swaths/frequencyA/HV'])/cc_linear
    S21 = np.array(h5File[f'/science/{freq_band}SAR/RSLC/swaths/frequencyA/VH'])/cc_linear
    S22 = np.array(h5File[f'/science/{freq_band}SAR/RSLC/swaths/frequencyA/VV'])/cc_linear
    
    # xCoordinateSpacing = np.array(h5File[f'/science/{freq_band}SAR/RSLC/swaths/frequencyA/xCoordinateSpacing'])
    # yCoordinateSpacing = np.array(h5File[f'/science/{freq_band}SAR/RSLC/swaths/frequencyA/yCoordinateSpacing'])
    # xCoordinates = np.array(h5File[f'/science/{freq_band}SAR/RSLC/swaths/frequencyA/xCoordinates'])
    # yCoordinates = np.array(h5File[f'/science/{freq_band}SAR/RSLC/swaths/frequencyA/yCoordinates'])
    # projection = np.array(h5File[f'/science/{freq_band}SAR/RSLC/metadata/radarGrid/projection'])

    sceneCenterAlongTrackSpacing = np.array(h5File[f'/science/{freq_band}SAR/RSLC/swaths/frequencyA/sceneCenterAlongTrackSpacing' ])
    sceneCenterGroundRangeSpacing = np.array(h5File[f'/science/{freq_band}SAR/RSLC/swaths/frequencyA/sceneCenterGroundRangeSpacing'])
    slantRange = np.array(h5File[f'/science/{freq_band}SAR/RSLC/swaths/frequencyA/slantRange'])
    slantRangeSpacing = np.array(h5File[f'/science/{freq_band}SAR/RSLC/swaths/frequencyA/slantRangeSpacing'])
    coordinateX = np.array(h5File[f'/science/{freq_band}SAR/RSLC/metadata/geolocationGrid/coordinateX' ])[1,:,:] # at 0 height
    coordinateY = np.array(h5File[f'/science/{freq_band}SAR/RSLC/metadata/geolocationGrid/coordinateY'])[1,:,:] # at 0 height
    epsg = np.array(h5File[f'/science/{freq_band}SAR/RSLC/metadata/geolocationGrid/epsg'])

    multi_look_factor = slantRangeSpacing/sceneCenterAlongTrackSpacing


    h5File.close()


    mlrows,mlcols = S11.shape[0]//azlks,S11.shape[1]//rglks


    inFolder = os.path.dirname(inFile)   
    if not inFolder:
        inFolder = "."
    T3Folder = os.path.join(inFolder,os.path.basename(inFile).split('.h5')[0],'T3')
    # os.makedirs(T3Folder,exist_ok=True)

    if not os.path.isdir(T3Folder):
        print("T3 folder does not exist. \nCreating folder {}".format(T3Folder))
        # os.mkdir(T3Folder)
        os.makedirs(T3Folder,exist_ok=True)
    
    os.chdir(T3Folder)
    ######################################################
    lat_interp = intp_grid(coordinateY,(mlrows,mlcols))
    write_latlon('lat.bin', lat_interp)
    long_interp = intp_grid(coordinateX,(mlrows,mlcols))
    write_latlon('lon.bin', long_interp)

    # print('lat lon saved')
    gdal.Translate('lat.vrt', 'lat.bin')
    gdal.Translate('lon.vrt', 'lon.bin')

    minX, maxX = np.nanmin(long_interp),np.nanmax(long_interp)
    minY,maxY = np.nanmin(lat_interp),np.nanmax(lat_interp)

    x_res_m = sceneCenterAlongTrackSpacing*azlks  # in meters
    y_res_m = slantRangeSpacing*rglks
    mean_lat = np.nanmean(lat_interp) 
    x_res_deg = x_res_m / (111320 * np.cos(np.deg2rad(mean_lat)))
    y_res_deg = y_res_m / 111320.0

    bbox = [minX, minY, maxX, maxY]

    #######################################################

    # 3x1 Pauli Coherency Matrix
    Kp = (1/np.sqrt(2))*np.array([S11+S22, S11-S22, S12+S21])

    del S11,S12,S21,S22


    def write_element(data,outname):
        write_latlon('data.bin', data)
        gdal.Translate('data.vrt', 'data.bin')
        update_vrt('data.vrt')
        input_vrt = "data.vrt"
        geocode_grid(input_vrt, outname, 'bin', x_res_deg, y_res_deg, bbox,resampleAlg="near")
        os.remove('data.vrt')
        os.remove('data.bin')
        os.remove('data.hdr')
        
    # 3x3 Pauli Coherency Matrix elements
    T11 = mlook_arr(np.abs(Kp[0])**2,azlks,rglks).astype(np.float32)
    write_element(T11,'T11.bin')
    print(f"Saved file {os.path.join(T3Folder,'T11.bin')}")
    del T11
    T22 = mlook_arr(np.abs(Kp[1])**2,azlks,rglks).astype(np.float32)
    write_element(T22,'T22.bin')
    print(f"Saved file {os.path.join(T3Folder,'T22.bin')}")
    del T22
    T33 = mlook_arr(np.abs(Kp[2])**2,azlks,rglks).astype(np.float32)
    write_element(T33,'T33.bin')
    print(f"Saved file {os.path.join(T3Folder,'T33.bin')}")
    del T33
    T12 = mlook_arr(Kp[0]*np.conj(Kp[1]),azlks,rglks).astype(np.complex64)
    write_element(np.real(T12),'T12_real.bin') 
    print(f"Saved file {os.path.join(T3Folder,'T12_real.bin')}")
    write_element(np.imag(T12),'T12_imag.bin')
    print(f"Saved file {os.path.join(T3Folder,'T12_imag.bin')}")
    del T12
    
    T13 = mlook_arr(Kp[0]*np.conj(Kp[2]),azlks,rglks).astype(np.complex64)
    write_element(np.real(T13),'T13_real.bin')
    print(f"Saved file {os.path.join(T3Folder,'T13_real.bin')}")
    write_element(np.imag(T13),'T13_imag.bin')
    print(f"Saved file {os.path.join(T3Folder,'T13_imag.bin')}")
    del T13
    
    T23 = mlook_arr(Kp[1]*np.conj(Kp[2]),azlks,rglks).astype(np.complex64)
    write_element(np.real(T23),'T23_real.bin')
    print(f"Saved file {os.path.join(T3Folder,'T23_real.bin')}")
    write_element(np.imag(T23),'T23_imag.bin')
    print(f"Saved file {os.path.join(T3Folder,'T23_imag.bin')}")

    del Kp,T23
    
    os.remove('lat.bin')
    os.remove('lat.vrt')
    os.remove('lat.hdr')
    os.remove('lon.bin')
    os.remove('lon.vrt')
    os.remove('lon.hdr')
    
def asar_T3(inFile,azlks,rglks,cc_linear):
    print("Considering S12 = S21")
           
    try:
        h5File = h5py.File(inFile,"r")
    except:
        raise('Invalid .h5 file !!')
    
    
    h5File = h5py.File(inFile,"r")
    if '/science/LSAR' in h5File:
        freq_band = 'L'            
        print("Detected L-band data ")
    elif '/science/SSAR' in h5File:
        freq_band = 'S'
        print("Detected S-band data")
    else:
        print("Neither LSAR nor SSAR data found in the file.")
        h5File.close()
        return
    
    S11 = np.array(h5File[f'/science/{freq_band}SAR/RSLC/swaths/frequencyA/HH'])/cc_linear
    S12 = np.array(h5File[f'/science/{freq_band}SAR/RSLC/swaths/frequencyA/HV'])/cc_linear
    S21 = np.array(h5File[f'/science/{freq_band}SAR/RSLC/swaths/frequencyA/VH'])/cc_linear
    S22 = np.array(h5File[f'/science/{freq_band}SAR/RSLC/swaths/frequencyA/VV'])/cc_linear

    h5File.close()

    # 3x1 Pauli Coherency Matrix
    Kp = (1/np.sqrt(2))*np.array([S11+S22, S11-S22, S12+S21])

    del S11,S12,S21,S22

    # 3x3 Pauli Coherency Matrix elements
    T11 = mlook_arr(np.abs(Kp[0])**2,azlks,rglks).astype(np.float32)
    T22 = mlook_arr(np.abs(Kp[1])**2,azlks,rglks).astype(np.float32)
    T33 = mlook_arr(np.abs(Kp[2])**2,azlks,rglks).astype(np.float32)

    T12 = mlook_arr(Kp[0]*np.conj(Kp[1]),azlks,rglks).astype(np.complex64)
    T13 = mlook_arr(Kp[0]*np.conj(Kp[2]),azlks,rglks).astype(np.complex64)
    T23 = mlook_arr(Kp[1]*np.conj(Kp[2]),azlks,rglks).astype(np.complex64)

    del Kp
    
    
    inFolder = os.path.dirname(inFile)   
    if not inFolder:
        inFolder = "."
    T3Folder = os.path.join(inFolder,os.path.basename(inFile).split('.h5')[0],'T3')
    os.makedirs(T3Folder,exist_ok=True)

    if not os.path.isdir(T3Folder):
        print("T3 folder does not exist. \nCreating folder {}".format(T3Folder))
        os.mkdir(T3Folder)
        
    # write_T3(np.dstack([T11,T12,T13,np.conjugate(T12),T22,T23,np.conjugate(T13),np.conjugate(T23),T33]),T3Folder)
    write_T3([np.real(T11),np.real(T12),np.imag(T12),np.real(T13),np.imag(T13),
                np.real(T22),np.real(T23),np.imag(T23),
                np.real(T33)],T3Folder)

def asar_geo_C3(inFile,azlks,rglks,cc_linear):
    print("Considering S12 = S21")
    try:
        h5File = h5py.File(inFile,"r")
    except:
        raise('Invalid .h5 file !!')
    
    
    h5File = h5py.File(inFile,"r")
    if '/science/LSAR' in h5File:
        freq_band = 'L'            
        print("Detected L-band data ")
    elif '/science/SSAR' in h5File:
        freq_band = 'S'
        print(" Detected S-band data")
    else:
        print("Neither LSAR nor SSAR data found in the file.")
        h5File.close()
        return
    
    S11 = np.array(h5File[f'/science/{freq_band}SAR/RSLC/swaths/frequencyA/HH'])/cc_linear
    S12 = np.array(h5File[f'/science/{freq_band}SAR/RSLC/swaths/frequencyA/HV'])/cc_linear
    S21 = np.array(h5File[f'/science/{freq_band}SAR/RSLC/swaths/frequencyA/VH'])/cc_linear
    S22 = np.array(h5File[f'/science/{freq_band}SAR/RSLC/swaths/frequencyA/VV'])/cc_linear
    
    sceneCenterAlongTrackSpacing = np.array(h5File[f'/science/{freq_band}SAR/RSLC/swaths/frequencyA/sceneCenterAlongTrackSpacing' ])
    sceneCenterGroundRangeSpacing = np.array(h5File[f'/science/{freq_band}SAR/RSLC/swaths/frequencyA/sceneCenterGroundRangeSpacing'])
    slantRange = np.array(h5File[f'/science/{freq_band}SAR/RSLC/swaths/frequencyA/slantRange'])
    slantRangeSpacing = np.array(h5File[f'/science/{freq_band}SAR/RSLC/swaths/frequencyA/slantRangeSpacing'])
    coordinateX = np.array(h5File[f'/science/{freq_band}SAR/RSLC/metadata/geolocationGrid/coordinateX' ])[1,:,:] # at 0 height
    coordinateY = np.array(h5File[f'/science/{freq_band}SAR/RSLC/metadata/geolocationGrid/coordinateY'])[1,:,:] # at 0 height
    epsg = np.array(h5File[f'/science/{freq_band}SAR/RSLC/metadata/geolocationGrid/epsg'])

    multi_look_factor = slantRangeSpacing/sceneCenterAlongTrackSpacing


    h5File.close()


    mlrows,mlcols = S11.shape[0]//azlks,S11.shape[1]//rglks


    inFolder = os.path.dirname(inFile)   
    if not inFolder:
        inFolder = "."
    C3Folder = os.path.join(inFolder,os.path.basename(inFile).split('.h5')[0],'C3')
    
    
    if not os.path.isdir(C3Folder):
        print("C3 folder does not exist. \nCreating folder {}".format(C3Folder))
        os.makedirs(C3Folder,exist_ok=True)
    
    os.chdir(C3Folder)
    ######################################################
    lat_interp = intp_grid(coordinateY,(mlrows,mlcols))
    write_latlon('lat.bin', lat_interp)
    long_interp = intp_grid(coordinateX,(mlrows,mlcols))
    write_latlon('lon.bin', long_interp)

    # print('lat lon saved')
    gdal.Translate('lat.vrt', 'lat.bin')
    gdal.Translate('lon.vrt', 'lon.bin')

    minX, maxX = np.nanmin(long_interp),np.nanmax(long_interp)
    minY,maxY = np.nanmin(lat_interp),np.nanmax(lat_interp)

    x_res_m = sceneCenterAlongTrackSpacing*azlks  # in meters
    y_res_m = slantRangeSpacing*rglks
    mean_lat = np.nanmean(lat_interp) 
    x_res_deg = x_res_m / (111320 * np.cos(np.deg2rad(mean_lat)))
    y_res_deg = y_res_m / 111320.0

    bbox = [minX, minY, maxX, maxY]

    #######################################################


    # Kl- 3-D Lexicographic feature vector
    Kl = np.array([S11, np.sqrt(2)*0.5*(S12+S21), S22])
    del S11, S12, S21, S22
    def write_element(data,outname):
        write_latlon('data.bin', data)
        gdal.Translate('data.vrt', 'data.bin')
        update_vrt('data.vrt')
        input_vrt = "data.vrt"
        geocode_grid(input_vrt, outname, 'bin', x_res_deg, y_res_deg, bbox,resampleAlg="near")
        os.remove('data.vrt')
        os.remove('data.bin')
        os.remove('data.hdr')
        
        
    # 3x3 COVARIANCE Matrix elements

    C11 = mlook_arr(np.abs(Kl[0])**2,azlks,rglks).astype(np.float32)
    write_element(C11,'C11.bin')
    print(f"Saved file {os.path.join(C3Folder,'C11.bin')}")
    del C11
    C22 = mlook_arr(np.abs(Kl[1])**2,azlks,rglks).astype(np.float32)
    write_element(C22,'C22.bin')
    print(f"Saved file {os.path.join(C3Folder,'C22.bin')}")
    del C22
    C33 = mlook_arr(np.abs(Kl[2])**2,azlks,rglks).astype(np.float32)
    write_element(C33,'C33.bin')
    print(f"Saved file {os.path.join(C3Folder,'C33.bin')}")
    del C33 

    C12 = mlook_arr(Kl[0]*np.conj(Kl[1]),azlks,rglks).astype(np.complex64)
    write_element(np.real(C12),'C12_real.bin')
    print(f"Saved file {os.path.join(C3Folder,'C12_real.bin')}")
    write_element(np.imag(C12),'C12_imag.bin')
    print(f"Saved file {os.path.join(C3Folder,'C12_imag.bin')}")
    del C12
    C13 = mlook_arr(Kl[0]*np.conj(Kl[2]),azlks,rglks).astype(np.complex64)
    write_element(np.real(C13),'C13_real.bin')
    print(f"Saved file {os.path.join(C3Folder,'C13_real.bin')}")
    write_element(np.imag(C13),'C13_imag.bin')
    print(f"Saved file {os.path.join(C3Folder,'C13_imag.bin')}")
    del C13
    C23 = mlook_arr(Kl[1]*np.conj(Kl[2]),azlks,rglks).astype(np.complex64)
    write_element(np.real(C23),'C23_real.bin')
    print(f"Saved file {os.path.join(C3Folder,'C23_real.bin')}")
    write_element(np.imag(C23),'C23_imag.bin')
    print(f"Saved file {os.path.join(C3Folder,'C23_imag.bin')}")
    del C23, Kl
    os.remove('lat.bin')
    os.remove('lat.vrt')
    os.remove('lat.hdr')
    os.remove('lon.bin')
    os.remove('lon.vrt')
    os.remove('lon.hdr')


def asar_C3(inFile,azlks,rglks,cc_linear):    
    
    print("Considering S12 = S21")
    try:
        h5File = h5py.File(inFile,"r")
    except:
        raise('Invalid .h5 file !!')
    
    
    h5File = h5py.File(inFile,"r")
    if '/science/LSAR' in h5File:
        freq_band = 'L'            
        print("Detected L-band data ")
    elif '/science/SSAR' in h5File:
        freq_band = 'S'
        print(" Detected S-band data")
    else:
        print("Neither LSAR nor SSAR data found in the file.")
        h5File.close()
        return
    
    S11 = np.array(h5File[f'/science/{freq_band}SAR/RSLC/swaths/frequencyA/HH'])/cc_linear
    S12 = np.array(h5File[f'/science/{freq_band}SAR/RSLC/swaths/frequencyA/HV'])/cc_linear
    S21 = np.array(h5File[f'/science/{freq_band}SAR/RSLC/swaths/frequencyA/VH'])/cc_linear
    S22 = np.array(h5File[f'/science/{freq_band}SAR/RSLC/swaths/frequencyA/VV'])/cc_linear
    
    # xCoordinateSpacing = np.array(h5File[f'/science/{freq_band}SAR/RSLC/swaths/frequencyA/xCoordinateSpacing'])
    # yCoordinateSpacing = np.array(h5File[f'/science/{freq_band}SAR/RSLC/swaths/frequencyA/yCoordinateSpacing'])
    # xCoordinates = np.array(h5File[f'/science/{freq_band}SAR/RSLC/swaths/frequencyA/xCoordinates'])
    # yCoordinates = np.array(h5File[f'/science/{freq_band}SAR/RSLC/swaths/frequencyA/yCoordinates'])
    # projection = np.array(h5File[f'/science/{freq_band}SAR/RSLC/metadata/radarGrid/projection'])

    h5File.close()


    # Kl- 3-D Lexicographic feature vector
    Kl = np.array([S11, np.sqrt(2)*0.5*(S12+S21), S22])
    del S11, S12, S21, S22

    # 3x3 COVARIANCE Matrix elements

    C11 = mlook_arr(np.abs(Kl[0])**2,azlks,rglks).astype(np.float32)
    C22 = mlook_arr(np.abs(Kl[1])**2,azlks,rglks).astype(np.float32)
    C33 = mlook_arr(np.abs(Kl[2])**2,azlks,rglks).astype(np.float32)

    C12 = mlook_arr(Kl[0]*np.conj(Kl[1]),azlks,rglks).astype(np.complex64)
    C13 = mlook_arr(Kl[0]*np.conj(Kl[2]),azlks,rglks).astype(np.complex64)
    C23 = mlook_arr(Kl[1]*np.conj(Kl[2]),azlks,rglks).astype(np.complex64)


    inFolder = os.path.dirname(inFile)   
    if not inFolder:
        inFolder = "."
    C3Folder = os.path.join(inFolder,os.path.basename(inFile).split('.h5')[0],'C3')
    os.makedirs(C3Folder,exist_ok=True)
    
    if not os.path.isdir(C3Folder):
        print("C3 folder does not exist. \nCreating folder {}".format(C3Folder))
        os.mkdir(C3Folder)
    
    # write_C3(np.dstack([C11,C12,C13,np.conjugate(C12),C22,C23,np.conjugate(C13),np.conjugate(C23),C33]),C3Folder)
    write_C3([np.real(C11),np.real(C12),np.imag(C12),np.real(C13),np.imag(C13),
                np.real(C22),np.real(C23),np.imag(C23),
                np.real(C33)],C3Folder)



def asar_geo_C4(inFile,azlks,rglks,cc_linear):
    # print("Considering S12 = S21")
    try:
        h5File = h5py.File(inFile,"r")
    except:
        raise('Invalid .h5 file !!')
    
    
    h5File = h5py.File(inFile,"r")
    if '/science/LSAR' in h5File:
        freq_band = 'L'            
        print("Detected L-band data ")
    elif '/science/SSAR' in h5File:
        freq_band = 'S'
        print(" Detected S-band data")
    else:
        print("Neither LSAR nor SSAR data found in the file.")
        h5File.close()
        return
    
    S11 = np.array(h5File[f'/science/{freq_band}SAR/RSLC/swaths/frequencyA/HH'])/cc_linear
    S12 = np.array(h5File[f'/science/{freq_band}SAR/RSLC/swaths/frequencyA/HV'])/cc_linear
    S21 = np.array(h5File[f'/science/{freq_band}SAR/RSLC/swaths/frequencyA/VH'])/cc_linear
    S22 = np.array(h5File[f'/science/{freq_band}SAR/RSLC/swaths/frequencyA/VV'])/cc_linear
    
    sceneCenterAlongTrackSpacing = np.array(h5File[f'/science/{freq_band}SAR/RSLC/swaths/frequencyA/sceneCenterAlongTrackSpacing' ])
    sceneCenterGroundRangeSpacing = np.array(h5File[f'/science/{freq_band}SAR/RSLC/swaths/frequencyA/sceneCenterGroundRangeSpacing'])
    slantRange = np.array(h5File[f'/science/{freq_band}SAR/RSLC/swaths/frequencyA/slantRange'])
    slantRangeSpacing = np.array(h5File[f'/science/{freq_band}SAR/RSLC/swaths/frequencyA/slantRangeSpacing'])
    coordinateX = np.array(h5File[f'/science/{freq_band}SAR/RSLC/metadata/geolocationGrid/coordinateX' ])[1,:,:] # at 0 height
    coordinateY = np.array(h5File[f'/science/{freq_band}SAR/RSLC/metadata/geolocationGrid/coordinateY'])[1,:,:] # at 0 height
    epsg = np.array(h5File[f'/science/{freq_band}SAR/RSLC/metadata/geolocationGrid/epsg'])

    multi_look_factor = slantRangeSpacing/sceneCenterAlongTrackSpacing


    h5File.close()


    mlrows,mlcols = S11.shape[0]//azlks,S11.shape[1]//rglks


    inFolder = os.path.dirname(inFile)   
    if not inFolder:
        inFolder = "."
    C4Folder = os.path.join(inFolder,os.path.basename(inFile).split('.h5')[0],'C4')
    
    
    if not os.path.isdir(C4Folder):
        print("C4 folder does not exist. \nCreating folder {}".format(C4Folder))
        os.makedirs(C4Folder,exist_ok=True)
    
    os.chdir(C4Folder)
    ######################################################
    lat_interp = intp_grid(coordinateY,(mlrows,mlcols))
    write_latlon('lat.bin', lat_interp)
    long_interp = intp_grid(coordinateX,(mlrows,mlcols))
    write_latlon('lon.bin', long_interp)

    # print('lat lon saved')
    gdal.Translate('lat.vrt', 'lat.bin')
    gdal.Translate('lon.vrt', 'lon.bin')

    minX, maxX = np.nanmin(long_interp),np.nanmax(long_interp)
    minY,maxY = np.nanmin(lat_interp),np.nanmax(lat_interp)

    x_res_m = sceneCenterAlongTrackSpacing*azlks  # in meters
    y_res_m = slantRangeSpacing*rglks
    mean_lat = np.nanmean(lat_interp) 
    x_res_deg = x_res_m / (111320 * np.cos(np.deg2rad(mean_lat)))
    y_res_deg = y_res_m / 111320.0

    bbox = [minX, minY, maxX, maxY]

    #######################################################


    # Kl- 3-D Lexicographic feature vector
    Kl = np.array([S11, S12, S21, S22])
    del S11, S12, S21, S22
    def write_element(data,outname):
        write_latlon('data.bin', data)
        gdal.Translate('data.vrt', 'data.bin')
        update_vrt('data.vrt')
        input_vrt = "data.vrt"
        geocode_grid(input_vrt, outname, 'bin', x_res_deg, y_res_deg, bbox,resampleAlg="near")
        os.remove('data.vrt')
        os.remove('data.bin')
        os.remove('data.hdr')
        
        
    # 3x3 COVARIANCE Matrix elements

    C11 = mlook_arr(np.abs(Kl[0])**2,azlks,rglks).astype(np.float32)
    write_element(C11,'C11.bin')
    print(f"Saved file {os.path.join(C4Folder,'C11.bin')}")
    del C11
    C22 = mlook_arr(np.abs(Kl[1])**2,azlks,rglks).astype(np.float32)
    write_element(C22,'C22.bin')
    print(f"Saved file {os.path.join(C4Folder,'C22.bin')}")
    del C22
    C33 = mlook_arr(np.abs(Kl[2])**2,azlks,rglks).astype(np.float32)
    write_element(C33,'C33.bin')
    print(f"Saved file {os.path.join(C4Folder,'C33.bin')}")
    del C33 
    C44 = mlook_arr(np.abs(Kl[3])**2,azlks,rglks).astype(np.float32)
    write_element(C44,'C44.bin')
    print(f"Saved file {os.path.join(C4Folder,'C44.bin')}")
    del C44

    C12 = mlook_arr(Kl[0]*np.conj(Kl[1]),azlks,rglks).astype(np.complex64)
    write_element(np.real(C12),'C12_real.bin')
    print(f"Saved file {os.path.join(C4Folder,'C12_real.bin')}")
    write_element(np.imag(C12),'C12_imag.bin')
    print(f"Saved file {os.path.join(C4Folder,'C12_imag.bin')}")
    del C12
    C13 = mlook_arr(Kl[0]*np.conj(Kl[2]),azlks,rglks).astype(np.complex64)
    write_element(np.real(C13),'C13_real.bin')
    print(f"Saved file {os.path.join(C4Folder,'C13_real.bin')}")
    write_element(np.imag(C13),'C13_imag.bin')
    print(f"Saved file {os.path.join(C4Folder,'C13_imag.bin')}")
    del C13
    C14 = mlook_arr(Kl[0]*np.conj(Kl[3]),azlks,rglks).astype(np.complex64)
    write_element(np.real(C14),'C14_real.bin')
    print(f"Saved file {os.path.join(C4Folder,'C14_real.bin')}")
    write_element(np.imag(C14),'C14_imag.bin')
    print(f"Saved file {os.path.join(C4Folder,'C14_imag.bin')}")
    del C14
    
    
    
    C23 = mlook_arr(Kl[1]*np.conj(Kl[2]),azlks,rglks).astype(np.complex64)
    write_element(np.real(C23),'C23_real.bin')
    print(f"Saved file {os.path.join(C4Folder,'C23_real.bin')}")
    write_element(np.imag(C23),'C23_imag.bin')
    print(f"Saved file {os.path.join(C4Folder,'C23_imag.bin')}")
    del C23
    
    C24 = mlook_arr(Kl[1]*np.conj(Kl[3]),azlks,rglks).astype(np.complex64)
    write_element(np.real(C24),'C24_real.bin')
    print(f"Saved file {os.path.join(C4Folder,'C24_real.bin')}")
    write_element(np.imag(C24),'C24_imag.bin')
    print(f"Saved file {os.path.join(C4Folder,'C24_imag.bin')}")
    del C24
    
    C34 = mlook_arr(Kl[2]*np.conj(Kl[3]),azlks,rglks).astype(np.complex64)
    write_element(np.real(C34),'C34_real.bin')
    print(f"Saved file {os.path.join(C4Folder,'C34_real.bin')}")
    write_element(np.imag(C34),'C34_imag.bin')
    print(f"Saved file {os.path.join(C4Folder,'C34_imag.bin')}")
    del C34,kl
    
    os.remove('lat.bin')
    os.remove('lat.vrt')
    os.remove('lat.hdr')
    os.remove('lon.bin')
    os.remove('lon.vrt')
    os.remove('lon.hdr')
    
def asar_C4(inFile,azlks,rglks,cc_linear):    
    
    # print("Considering S12 = S21")
    try:
        h5File = h5py.File(inFile,"r")
    except:
        raise('Invalid .h5 file !!')
    
    
    h5File = h5py.File(inFile,"r")
    if '/science/LSAR' in h5File:
        freq_band = 'L'            
        print("Detected L-band data ")
    elif '/science/SSAR' in h5File:
        freq_band = 'S'
        print(" Detected S-band data")
    else:
        print("Neither LSAR nor SSAR data found in the file.")
        h5File.close()
        return
    
    S11 = np.array(h5File[f'/science/{freq_band}SAR/RSLC/swaths/frequencyA/HH'])/cc_linear
    S12 = np.array(h5File[f'/science/{freq_band}SAR/RSLC/swaths/frequencyA/HV'])/cc_linear
    S21 = np.array(h5File[f'/science/{freq_band}SAR/RSLC/swaths/frequencyA/VH'])/cc_linear
    S22 = np.array(h5File[f'/science/{freq_band}SAR/RSLC/swaths/frequencyA/VV'])/cc_linear
    
    # xCoordinateSpacing = np.array(h5File[f'/science/{freq_band}SAR/RSLC/swaths/frequencyA/xCoordinateSpacing'])
    # yCoordinateSpacing = np.array(h5File[f'/science/{freq_band}SAR/RSLC/swaths/frequencyA/yCoordinateSpacing'])
    # xCoordinates = np.array(h5File[f'/science/{freq_band}SAR/RSLC/swaths/frequencyA/xCoordinates'])
    # yCoordinates = np.array(h5File[f'/science/{freq_band}SAR/RSLC/swaths/frequencyA/yCoordinates'])
    # projection = np.array(h5File[f'/science/{freq_band}SAR/RSLC/metadata/radarGrid/projection'])

    h5File.close()


    # Kl- 4-D Lexicographic feature vector
    Kl = np.array([S11, S12, S21, S22])
    del S11, S12, S21, S22

    # 3x3 COVARIANCE Matrix elements

    C11 = mlook_arr(np.abs(Kl[0])**2,azlks,rglks).astype(np.float32)
    C22 = mlook_arr(np.abs(Kl[1])**2,azlks,rglks).astype(np.float32)
    C33 = mlook_arr(np.abs(Kl[2])**2,azlks,rglks).astype(np.float32)
    C44 = mlook_arr(np.abs(Kl[3])**2,azlks,rglks).astype(np.float32)
    
    C12 = mlook_arr(Kl[0]*np.conj(Kl[1]),azlks,rglks).astype(np.complex64)
    C13 = mlook_arr(Kl[0]*np.conj(Kl[2]),azlks,rglks).astype(np.complex64)
    C14 = mlook_arr(Kl[0]*np.conj(Kl[3]),azlks,rglks).astype(np.complex64)

    C23 = mlook_arr(Kl[1]*np.conj(Kl[2]),azlks,rglks).astype(np.complex64)
    C24 = mlook_arr(Kl[1]*np.conj(Kl[3]),azlks,rglks).astype(np.complex64)

    C34 = mlook_arr(Kl[2]*np.conj(Kl[3]),azlks,rglks).astype(np.complex64)


    inFolder = os.path.dirname(inFile)   
    if not inFolder:
        inFolder = "."
    C4Folder = os.path.join(inFolder,os.path.basename(inFile).split('.h5')[0],'C4')
    os.makedirs(C4Folder,exist_ok=True)
    
    if not os.path.isdir(C4Folder):
        print("C4 folder does not exist. \nCreating folder {}".format(C4Folder))
        os.mkdir(C4Folder)
    
    write_C4([np.real(C11),np.real(C12),np.imag(C12),np.real(C13),np.imag(C13),np.real(C14),np.imag(C14),
                np.real(C22),np.real(C23),np.imag(C23),np.real(C24),np.imag(C24),
                np.real(C33),np.real(C34),np.imag(C34),
                np.real(C44)],C4Folder)

def load_asar_meta(filepath):
    data_dict = {}

    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()
            if line:  # Ensure line is not empty
                key, value = line.split("=", 1)  # Split only on first "="
                data_dict[key.strip()] = value.strip()
    return data_dict


def convert_l1_tif_dB(infolder,  backscatter=1, complex_flag=0, outType="tif", 
                   cog_flag=False, cog_overviews = [2, 4, 8, 16], 
                   write_flag=True, max_workers=None):
    
    l_band_files = list(sorted(glob.glob(os.path.join(infolder, "ASAR_L_JOINT_*LEVEL1_*.tif"))))
    s_band_files = list(sorted(glob.glob(os.path.join(infolder, "ASAR_S_JOINT_*LEVEL1_*.tif"))))
    inc_file = glob.glob(os.path.join(infolder, "ASAR_*LEVEL1_INC_MAP*.tif"))
    window_size=3
    # print(len(l_band_files))
    # print(len(s_band_files))
    # print(len(inc_file))
    
    nb_file = glob.glob(os.path.join(infolder, "ASAR_*LEVEL1_BAND_META*.txt"))
    
    if len(nb_file) >0:
        print("Found meta data file applying noise bias.")
        meta_dict = load_asar_meta(nb_file[0])
        l_hh_nb = meta_dict['Image_Noise_Bias_L_HH']
        l_hv_nb = meta_dict['Image_Noise_Bias_L_HV']
        l_vh_nb = meta_dict['Image_Noise_Bias_L_VH']
        l_vv_nb = meta_dict['Image_Noise_Bias_L_VV']

        s_hh_nb = meta_dict['Image_Noise_Bias_S_HH']
        s_hv_nb = meta_dict['Image_Noise_Bias_S_HV']
        s_vh_nb = meta_dict['Image_Noise_Bias_S_VH']
        s_vv_nb = meta_dict['Image_Noise_Bias_S_VV']
    else:
        print("No meta data file found. Applying noise bias of 0.")
        l_hh_nb = 0
        l_hv_nb = 0
        l_vh_nb = 0
        l_vv_nb = 0

        s_hh_nb = 0
        s_hv_nb = 0
        s_vh_nb = 0
        s_vv_nb = 0

    if backscatter==2:
        out_fix = 'g0'
    elif backscatter==3:
        out_fix = 'b0'
    else:
        out_fix = 's0' 
    
    def output_names_dB(band_files, prefix):
        out_paths = []
        for file in band_files:
            pol = file.split("_LEVEL1_")[1].split(".tif")[0]  # Extract polarization (e.g., HH, HV, VH, VV)
            out_paths.append(os.path.join(infolder, f"{prefix}_{pol}_{out_fix}_dB.{outType}"))
        return out_paths
    def output_names_complex(band_files, prefix):
        out_paths = []
        for file in band_files:
            pol = file.split("_LEVEL1_")[1].split(".tif")[0]  # Extract polarization (e.g., HH, HV, VH, VV)           
            real_path = os.path.join(infolder, f"{prefix}_{pol}_{out_fix}_real.{outType}")
            imag_path = os.path.join(infolder, f"{prefix}_{pol}_{out_fix}_imag.{outType}")
        
            out_paths.extend([real_path, imag_path]) 
        return out_paths
    
    output_l_filepaths=[]
    output_s_filepaths=[]
    
    if l_band_files:
        if complex_flag:
            output_l_filepaths.extend(output_names_complex(l_band_files, "L"))
        else:
            output_l_filepaths.extend(output_names_dB(l_band_files, "L"))

    if s_band_files:
        if complex_flag:
            output_s_filepaths.extend(output_names_complex(s_band_files, "S"))
        else:
            output_s_filepaths.extend(output_names_dB(s_band_files, "S"))
    

    def copy_gcps_to_outputs(input_filepaths, output_filepaths):
        """ Copies GCPs from one input file to all output files. """
        input_filepath = input_filepaths[0]
        # Open the input file
        input_dataset = gdal.Open(input_filepath, gdal.GA_ReadOnly)
        if input_dataset is None:
            raise FileNotFoundError(f"Cannot open {input_filepath}")

        # Extract GCPs
        gcps = input_dataset.GetGCPs()
        gcp_projection = input_dataset.GetGCPProjection()

        if not gcps:
            # print("No GCPs found in the input file.")
            return

        # Apply GCPs to each output file
        for output_filepath in output_filepaths:
            output_dataset = gdal.Open(output_filepath, gdal.GA_Update)
            if output_dataset is None:
                raise FileNotFoundError(f"Cannot open {output_filepath}")

            output_dataset.SetGCPs(gcps, gcp_projection)
            output_dataset.GetRasterBand(1).SetNoDataValue(np.nan)
            output_dataset.FlushCache()  
            output_dataset = None 

        # Close the input file
        input_dataset = None

    if len(l_band_files)==4:
        print("Processing L-band data...")
        input_filepaths = l_band_files + inc_file
        # print(input_filepaths)
        process_chunks_parallel(input_filepaths, output_l_filepaths, window_size, 
                                write_flag, process_chunk_gtif,
                                *[complex_flag, backscatter, l_hh_nb, l_hv_nb, l_vh_nb, l_vv_nb],
                                bands_to_read=[2,2,2,2,1] , block_size=(512, 512), 
                                max_workers=max_workers, num_outputs=len(output_l_filepaths),
                                cog_flag=cog_flag, cog_overviews=cog_overviews,
                                post_proc=copy_gcps_to_outputs
                                )

    if len(s_band_files)==4:
        print("Processing S-band data...")
        input_filepaths = s_band_files + inc_file
        # print(input_filepaths)
        process_chunks_parallel(input_filepaths, output_s_filepaths, window_size, 
                                write_flag, process_chunk_gtif,
                                *[complex_flag, backscatter, s_hh_nb, s_hv_nb, s_vh_nb, s_vv_nb],
                                bands_to_read=[2,2,2,2,1] , block_size=(512, 512), 
                                max_workers=max_workers, num_outputs=len(output_s_filepaths),
                                cog_flag=cog_flag, cog_overviews=cog_overviews,
                                post_proc=copy_gcps_to_outputs
                                )
    
    

def process_chunk_gtif(chunks, window_size, *args):
    # kernel = np.ones((window_size,window_size),np.float32)/(window_size*window_size)
    complex_flag=int(args[-6])
    backscatter=int(args[-5])
    hh_nb=float(args[-4])
    hv_nb=float(args[-3])
    vh_nb=float(args[-2])
    vv_nb=float(args[-1])
    
    cc_dB = 42.0
    cc_linear = np.sqrt(10**(cc_dB/10.0))

    if backscatter==2 and complex_flag==1:

        hh = np.array(chunks[0])+1j*np.array(chunks[1])*np.tan(np.deg2rad(np.array(chunks[8])))/cc_linear
        hv = np.array(chunks[2])+1j*np.array(chunks[3])*np.tan(np.deg2rad(np.array(chunks[8])))/cc_linear
        vh = np.array(chunks[4])+1j*np.array(chunks[5])*np.tan(np.deg2rad(np.array(chunks[8])))/cc_linear
        vv = np.array(chunks[6])+1j*np.array(chunks[7])*np.tan(np.deg2rad(np.array(chunks[8])))/cc_linear
    elif backscatter==2 and complex_flag==0:
        hh = 10*np.log10(np.abs(np.array(chunks[0])+1j*np.array(chunks[1]))**2-hh_nb)+10*np.log10(np.tan(np.deg2rad(np.array(chunks[8]))))-cc_dB
        hv = 10*np.log10(np.abs(np.array(chunks[2])+1j*np.array(chunks[3]))**2-hv_nb)+10*np.log10(np.tan(np.deg2rad(np.array(chunks[8]))))-cc_dB
        vh = 10*np.log10(np.abs(np.array(chunks[4])+1j*np.array(chunks[5]))**2-vh_nb)+10*np.log10(np.tan(np.deg2rad(np.array(chunks[8]))))-cc_dB
        vv = 10*np.log10(np.abs(np.array(chunks[6])+1j*np.array(chunks[7]))**2-vv_nb)+10*np.log10(np.tan(np.deg2rad(np.array(chunks[8]))))-cc_dB
        
    elif backscatter==3 and complex_flag==0:
        hh = 10*np.log10(np.abs(np.array(chunks[0])+1j*np.array(chunks[1]))**2-hh_nb)-cc_dB
        hv = 10*np.log10(np.abs(np.array(chunks[2])+1j*np.array(chunks[3]))**2-hv_nb)-cc_dB
        vh = 10*np.log10(np.abs(np.array(chunks[4])+1j*np.array(chunks[5]))**2-vh_nb)-cc_dB
        vv = 10*np.log10(np.abs(np.array(chunks[6])+1j*np.array(chunks[7]))**2-vv_nb)-cc_dB
    elif backscatter==3 and complex_flag==1:
        hh = np.array(chunks[0])+1j*np.array(chunks[1])/cc_linear
        hv = np.array(chunks[2])+1j*np.array(chunks[3])/cc_linear
        vh = np.array(chunks[4])+1j*np.array(chunks[5])/cc_linear
        vv = np.array(chunks[6])+1j*np.array(chunks[7])/cc_linear
    elif complex_flag==0:
        hh = 10*np.log10(np.abs(np.array(chunks[0])+1j*np.array(chunks[1]))**2-hh_nb)+10*np.log10(np.sin(np.deg2rad(np.array(chunks[8]))))-cc_dB
        hv = 10*np.log10(np.abs(np.array(chunks[2])+1j*np.array(chunks[3]))**2-hv_nb)+10*np.log10(np.sin(np.deg2rad(np.array(chunks[8]))))-cc_dB
        vh = 10*np.log10(np.abs(np.array(chunks[4])+1j*np.array(chunks[5]))**2-vh_nb)+10*np.log10(np.sin(np.deg2rad(np.array(chunks[8]))))-cc_dB
        vv = 10*np.log10(np.abs(np.array(chunks[6])+1j*np.array(chunks[7]))**2-vv_nb)+10*np.log10(np.sin(np.deg2rad(np.array(chunks[8]))))-cc_dB
    else:
        hh = np.array(chunks[0])+1j*np.array(chunks[1])*np.sin(np.deg2rad(np.array(chunks[8])))/cc_linear
        hv = np.array(chunks[2])+1j*np.array(chunks[3])*np.sin(np.deg2rad(np.array(chunks[8])))/cc_linear
        vh = np.array(chunks[4])+1j*np.array(chunks[5])*np.sin(np.deg2rad(np.array(chunks[8])))/cc_linear
        vv = np.array(chunks[6])+1j*np.array(chunks[7])*np.sin(np.deg2rad(np.array(chunks[8])))/cc_linear

    hh[hh==0]=np.nan
    hh[hh==np.inf]=np.nan
    hh[hh==-np.inf]=np.nan
    
    hv[hv==0]=np.nan
    hv[hv==np.inf]=np.nan
    hv[hv==-np.inf]=np.nan
    
    vh[vh==0]=np.nan
    vh[vh==np.inf]=np.nan
    vh[vh==-np.inf]=np.nan
    
    vv[vv==0]=np.nan
    vv[vv==np.inf]=np.nan
    vv[vv==-np.inf]=np.nan
    
    if complex_flag:
        return np.real(hh),np.imag(hh),np.real(hv),np.imag(hv),np.real(vh),np.imag(vh),np.real(vv),np.imag(vv)
    else:
        return hh,hv,vh,vv
    
@time_it
def isro_asar_old(inFile,matrix='T3',azlks=8,rglks=6,geocode_flag=False,calibration_constant = 42):
    """
    Extracts the S2/C3/T3  matrix elements from a ASAR RSLC HDF5 file 
    and saves them into respective binary files.
    
    Example:
    --------
    >>> nisar_rslc("path_to_file.h5", azlks=30, rglks=15)
    This will extract T3 matrix elements from the specified ASAR RSLC file 
    and save them in the respective folders.
    
    Parameters:
    -----------
    inFile : str
        The path to the ASAR RSLC HDF5 file containing the radar data.

    azlks : int, optional (default=8)
        The number of azimuth looks for multi-looking. 

    rglks : int, optional (default=6)
        The number of range looks for multi-looking. 
        
    matrixType : str, optional (default = T3)
        Type of matrix to extract. Valid options are 'S2', 'C3', and 'T3'.
         

    Returns:
    --------
    None
        The function does not return any value. Instead, it creates a folder 
        named `C2/S2/C3/T3` (if not already present) and saves the following binary files:


    Raises:
    -------
    Exception
        If the RSLC HDF5 file is invalid or cannot be read.


    """
    
    
    
    cc_linear = np.sqrt(10**(calibration_constant/10))

    if matrix == 'S2':


        print("Considering S12 = S21")
            
        try:
            h5File = h5py.File(inFile,"r")
        except:
            raise('Invalid .h5 file !!')
        
        h5File = h5py.File(inFile,"r")
        if '/science/LSAR' in h5File:
            freq_band = 'L'            
            print("Detected L-band data ")
        elif '/science/SSAR' in h5File:
            freq_band = 'S'
            print(" Detected S-band data")
        else:
            print("Neither LSAR nor SSAR data found in the file.")
            h5File.close()
            return
        
        S11 = np.array(h5File[f'/science/{freq_band}SAR/RSLC/swaths/frequencyA/HH'])/cc_linear
        S12 = np.array(h5File[f'/science/{freq_band}SAR/RSLC/swaths/frequencyA/HV'])/cc_linear
        S21 = np.array(h5File[f'/science/{freq_band}SAR/RSLC/swaths/frequencyA/VH'])/cc_linear
        S22 = np.array(h5File[f'/science/{freq_band}SAR/RSLC/swaths/frequencyA/VV'])/cc_linear

        h5File.close()
        
        
        rows,cols = S11.shape
        inFolder = os.path.dirname(inFile)   
        if not inFolder:
            inFolder = "."
        out_dir = os.path.join(inFolder,os.path.basename(inFile).split('.h5')[0],'S2')
        os.makedirs(out_dir,exist_ok=True)
       
        
        out_file = os.path.join(out_dir,'s11.bin')
        write_s2_bin(out_file,S11)
        print("Saved file "+out_file)
        
        out_file = os.path.join(out_dir,'s12.bin')
        write_s2_bin(out_file,(S12+S21)/2)
        print("Saved file "+out_file)
        
        out_file = os.path.join(out_dir,'s21.bin')
        write_s2_bin(out_file,(S12+S21)/2)
        print("Saved file "+out_file)
        
        out_file = os.path.join(out_dir,'s22.bin')
        write_s2_bin(out_file,S22)
        print("Saved file "+out_file)
        
        file = open(out_dir +'/config.txt',"w+")
        file.write('Nrow\n%d\n---------\nNcol\n%d\n---------\nPolarCase\nmonostatic\n---------\nPolarType\nfull'%(rows,cols))
        file.close() 
        
    elif matrix == 'T3':
        
        if geocode_flag:
            asar_geo_T3(inFile,azlks,rglks,cc_linear)
        else:
            asar_T3(inFile,azlks,rglks,cc_linear)

    elif matrix == 'C3':
        if geocode_flag:
            asar_geo_C3(inFile,azlks,rglks,cc_linear)
        else:
            asar_C3(inFile,azlks,rglks,cc_linear)
            
    elif matrix == 'C4':
        if geocode_flag:
            asar_geo_C4(inFile,azlks,rglks,cc_linear)
        else:
            asar_C4(inFile,azlks,rglks,cc_linear)
        

    else:
        raise ValueError('Invalid matrix type. Valid types are "S2", "C4", "T3" and "C3"')