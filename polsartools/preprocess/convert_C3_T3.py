import os
import numpy as np
from osgeo import gdal
gdal.UseExceptions()
from polsartools.utils.proc_utils import process_chunks_parallel
from polsartools.utils.utils import conv2d,time_it

from polsartools.utils.convert_matrices import C3_T3_mat

@time_it
def convert_C3_T3(infolder, outType="tif", window_size=1, 
                  write_flag=True, max_workers=None,block_size=(512, 512), 
                  cog_flag=False,cog_overviews = [2, 4, 8, 16], 
                  progress_callback=None  
                  
                  ):

    if os.path.isfile(os.path.join(infolder,"C11.bin")):
        ds = gdal.Open(os.path.join(infolder,"C11.bin"))
        rows,cols = ds.RasterYSize, ds.RasterXSize 
        ds = None
        
        input_filepaths = [
        os.path.join(infolder,"C11.bin"),
        os.path.join(infolder,'C12_real.bin'), os.path.join(infolder,'C12_imag.bin'),  
        os.path.join(infolder,'C13_real.bin'), os.path.join(infolder,'C13_imag.bin'),
        os.path.join(infolder,"C22.bin"),
        os.path.join(infolder,'C23_real.bin'), os.path.join(infolder,'C23_imag.bin'),  
        os.path.join(infolder,"C33.bin"),
        ]
        os.makedirs(os.path.join(os.path.dirname(infolder), "T3"), exist_ok=True)
        
    elif os.path.isfile(os.path.join(infolder,"C11.tif")):
        ds = gdal.Open(os.path.join(infolder,"C11.tif"))
        rows,cols = ds.RasterYSize, ds.RasterXSize 
        ds = None
        
        input_filepaths = [
        os.path.join(infolder,"C11.tif"),
        os.path.join(infolder,'C12_real.tif'), os.path.join(infolder,'C12_imag.tif'),          
        os.path.join(infolder,'C13_real.tif'), os.path.join(infolder,'C13_imag.tif'),
        os.path.join(infolder,"C22.tif"),
        os.path.join(infolder,'C23_real.tif'), os.path.join(infolder,'C23_imag.tif'),          
        os.path.join(infolder,"C33.tif"),
        ]
        os.makedirs(os.path.join(os.path.dirname(infolder), "T3"), exist_ok=True)
    else:
        raise Exception(f"Invalid C3 folder!!")

    output_filepaths = []
    
    if outType == "bin":
        output_filepaths.append(os.path.join(os.path.dirname(infolder), "T3", "T11.bin"))
        output_filepaths.append(os.path.join(os.path.dirname(infolder), "T3", "T12_real.bin"))
        output_filepaths.append(os.path.join(os.path.dirname(infolder), "T3", "T12_imag.bin"))
        output_filepaths.append(os.path.join(os.path.dirname(infolder), "T3", "T13_real.bin"))
        output_filepaths.append(os.path.join(os.path.dirname(infolder), "T3", "T13_imag.bin"))
        output_filepaths.append(os.path.join(os.path.dirname(infolder), "T3", "T22.bin"))
        output_filepaths.append(os.path.join(os.path.dirname(infolder), "T3", "T23_real.bin"))
        output_filepaths.append(os.path.join(os.path.dirname(infolder), "T3", "T23_imag.bin"))
        output_filepaths.append(os.path.join(os.path.dirname(infolder), "T3", "T33.bin"))
        
    else:
        output_filepaths.append(os.path.join(os.path.dirname(infolder), "T3", "T11.tif"))
        output_filepaths.append(os.path.join(os.path.dirname(infolder), "T3", "T12_real.tif"))
        output_filepaths.append(os.path.join(os.path.dirname(infolder), "T3", "T12_imag.tif"))
        output_filepaths.append(os.path.join(os.path.dirname(infolder), "T3", "T13_real.tif"))
        output_filepaths.append(os.path.join(os.path.dirname(infolder), "T3", "T13_imag.tif"))
        output_filepaths.append(os.path.join(os.path.dirname(infolder), "T3", "T22.tif"))
        output_filepaths.append(os.path.join(os.path.dirname(infolder), "T3", "T23_real.tif"))
        output_filepaths.append(os.path.join(os.path.dirname(infolder), "T3", "T23_imag.tif"))
        output_filepaths.append(os.path.join(os.path.dirname(infolder), "T3", "T33.tif"))
    
    process_chunks_parallel(input_filepaths, list(output_filepaths), 
                            window_size=window_size, write_flag=write_flag,
            processing_func=process_chunk_C3T3,
            block_size=block_size, max_workers=max_workers, 
            num_outputs=len(output_filepaths), cog_flag=cog_flag, 
            cog_overviews=cog_overviews, 
            progress_callback=progress_callback
            )
    
    file = open(os.path.join(os.path.dirname(infolder), "T3",'config.txt') ,"w+")
    file.write('Nrow\n%d\n---------\nNcol\n%d\n---------\nPolarCase\nmonostatic\n---------\nPolarType\nfull'%(rows,cols))
    file.close()  

def process_chunk_C3T3(chunks, window_size, input_filepaths, *args):


    if 'C11' in input_filepaths[0] and 'C22' in input_filepaths[5] and 'C33' in input_filepaths[8]:
        C11 = np.array(chunks[0])
        C12 = np.array(chunks[1])+1j*np.array(chunks[2])
        C13 = np.array(chunks[3])+1j*np.array(chunks[4])
        C21 = np.conj(C12)
        C22 = np.array(chunks[5])
        C23 = np.array(chunks[6])+1j*np.array(chunks[7])
        C31 = np.conj(C13)
        C32 = np.conj(C23)
        C33 = np.array(chunks[8])
        T_T1 = np.array([[C11, C12, C13], 
                         [C21, C22, C23], 
                         [C31, C32, C33]])
        T_T1 = C3_T3_mat(T_T1)
    else:
        raise Exception(f"Invalid C3 folder!!")

    if window_size>1:
        kernel = np.ones((window_size,window_size),np.float32)/(window_size*window_size)

        t11f = conv2d(T_T1[0,0,:,:],kernel)
        t12f = conv2d(np.real(T_T1[0,1,:,:]),kernel)+1j*conv2d(np.imag(T_T1[0,1,:,:]),kernel)
        t13f = conv2d(np.real(T_T1[0,2,:,:]),kernel)+1j*conv2d(np.imag(T_T1[0,2,:,:]),kernel)
        
        t21f = np.conj(t12f) 
        t22f = conv2d(T_T1[1,1,:,:],kernel)
        t23f = conv2d(np.real(T_T1[1,2,:,:]),kernel)+1j*conv2d(np.imag(T_T1[1,2,:,:]),kernel)

        t31f = np.conj(t13f) 
        t32f = np.conj(t23f) 
        t33f = conv2d(T_T1[2,2,:,:],kernel)

        T_T1 = np.array([[t11f, t12f, t13f], [t21f, t22f, t23f], [t31f, t32f, t33f]])


    # return T_T1[0,0,:,:].astype(np.float32), T_T1[0,1,:,:].real.astype(np.float32), T_T1[0,1,:,:].imag, T_T1[0,2,:,:].real, T_T1[0,2,:,:].imag, T_T1[1,1,:,:], T_T1[1,2,:,:].real, T_T1[1,2,:,:].imag, T_T1[2,2,:,:]

    return T_T1[0,0,:,:].astype(np.float32), T_T1[0,1,:,:].real.astype(np.float32), T_T1[0,1,:,:].imag.astype(np.float32), T_T1[0,2,:,:].real.astype(np.float32), T_T1[0,2,:,:].imag.astype(np.float32), T_T1[1,1,:,:].astype(np.float32), T_T1[1,2,:,:].real.astype(np.float32), T_T1[1,2,:,:].imag.astype(np.float32), T_T1[2,2,:,:].astype(np.float32)