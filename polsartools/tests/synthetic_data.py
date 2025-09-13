#%%
from osgeo import gdal
import numpy as np
import os

def write_complex_tiff(path, data, shape):
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(path, shape[1], shape[0], 2, gdal.GDT_CFloat32)
    ds.GetRasterBand(1).WriteArray(data)
    ds.FlushCache()
    ds = None
def save_float_tiff(path, data):
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(path, data.shape[1], data.shape[0], 2, gdal.GDT_Float32)
    ds.GetRasterBand(1).WriteArray(data)
    ds.FlushCache()
    ds = None

def gen_S2(folder, shape=(100, 100)):
    os.makedirs(folder, exist_ok=True)

    s11 = np.random.randn(*shape) + 1j * np.random.randn(*shape)
    s12 = np.random.randn(*shape) + 1j * np.random.randn(*shape)
    s22 = np.random.randn(*shape) + 1j * np.random.randn(*shape)

    write_complex_tiff(os.path.join(folder, 's11.tif'), s11, shape)
    write_complex_tiff(os.path.join(folder, 's12.tif'), s12, shape)
    write_complex_tiff(os.path.join(folder, 's21.tif'), s12, shape)
    write_complex_tiff(os.path.join(folder, 's22.tif'), s22, shape)

    return s11, s12, s22

def gen_C2(folder, s11, s12, s22):
    os.makedirs(folder, exist_ok=True) 
    C11 = np.abs(s11)**2
    C12 = s11 * np.conj(s12)
    C22 = np.abs(s22)**2
    save_float_tiff(os.path.join(folder, 'C11.tif'), C11.astype(np.float32))
    save_float_tiff(os.path.join(folder, 'C12_real.tif'), C12.real.astype(np.float32))
    save_float_tiff(os.path.join(folder, 'C12_imag.tif'), C12.imag.astype(np.float32))
    save_float_tiff(os.path.join(folder, 'C22.tif'), C22.astype(np.float32))

def gen_T3(folder, s11, s12, s22):
    os.makedirs(folder, exist_ok=True)
    k1 = (1/np.sqrt(2))*(s11+s22)
    k2 = (1/np.sqrt(2))*(s11 - s22)
    k3 = 2*s12

    # Compute outer products to form T3 elements
    T11 = np.real(k1 * np.conj(k1))
    T22 = np.real(k2 * np.conj(k2))
    T33 = np.real(k3 * np.conj(k3))

    T12 = k1 * np.conj(k2)
    T13 = k1 * np.conj(k3)
    T23 = k2 * np.conj(k3)

    # Save all components
    save_float_tiff(os.path.join(folder, 'T11.tif'), T11)
    save_float_tiff(os.path.join(folder, 'T22.tif'), T22)
    save_float_tiff(os.path.join(folder, 'T33.tif'), T33)

    save_float_tiff(os.path.join(folder, 'T12_real.tif'), T12.real)
    save_float_tiff(os.path.join(folder, 'T12_imag.tif'), T12.imag)

    save_float_tiff(os.path.join(folder, 'T13_real.tif'), T13.real)
    save_float_tiff(os.path.join(folder, 'T13_imag.tif'), T13.imag)

    save_float_tiff(os.path.join(folder, 'T23_real.tif'), T23.real)
    save_float_tiff(os.path.join(folder, 'T23_imag.tif'), T23.imag)
    
# # Generate S2 and C2
# s2_folder = './test_data/S2'
# c2_folder = './test_data/C2'
# s11, s12, s22 = gen_S2(s2_folder)
# gen_C2(c2_folder, s11, s12, s22)
# # Generate T3
# t3_folder = './test_data/T3'
# # generate_synthetic_T3(t3_folder)
# gen_T3(t3_folder, s11, s12, s22)
