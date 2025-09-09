import numpy as np
import glob
from osgeo import gdal,osr
gdal.UseExceptions()
import os
import sys
# import simplekml
import json
from polsartools.utils.utils import time_it
from polsartools.preprocess.convert_C3_T3 import convert_C3_T3

def write_bin_uav(file, wdata, lat, lon, dx, dy, sensor_type="UAVSAR"):
    [rows, cols] = wdata.shape
    driver = gdal.GetDriverByName("ENVI")
    outdata = driver.Create(file, cols, rows, 1, gdal.GDT_Float32)
    outdata.SetGeoTransform([lon, dx, 0, lat, 0, dy])
    outdata.SetProjection("EPSG:4326")
    band = outdata.GetRasterBand(1)
    band.WriteArray(wdata)
    band.SetNoDataValue(0)
    outdata.FlushCache()
    outdata = None
    
def update_hdr(hdrFile):
    with open(hdrFile, 'r') as file:
        content = file.read()

    # Replace "Arbitrary" with "Geographic Lat/Lon" and "North" with "WGS-84"
    content = content.replace('{Arbitrary', '{Geographic Lat/Lon')
    content = content.replace('North}', 'WGS-84}')
    content = content.replace('South}', 'WGS-84}')

    # Write the modified content back to the file
    with open(hdrFile, 'w') as file:
        file.write(content)


def write_geotiff(file, wdata, lat, lon, dx, dy, cog=False, ovr=[2, 4, 8, 16], comp=False):
    rows, cols = wdata.shape
    driver = gdal.GetDriverByName("GTiff")
    
    options = ['BIGTIFF=IF_SAFER']
    if comp:
        # options += ['COMPRESS=DEFLATE', 'PREDICTOR=2', 'ZLEVEL=9']
        options += ['COMPRESS=LZW']
    if cog:
        options += ['TILED=YES', 'BLOCKXSIZE=512', 'BLOCKYSIZE=512']

    outdata = driver.Create(file, cols, rows, 1, gdal.GDT_Float32, options)
    outdata.SetGeoTransform([lon, dx, 0, lat, 0, -abs(dy)])  # north-up orientation

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)  # WGS84
    outdata.SetProjection(srs.ExportToWkt())

    band = outdata.GetRasterBand(1)
    band.SetNoDataValue(0)
    band.WriteArray(wdata)
    if cog:
        outdata.BuildOverviews("NEAREST", ovr)

    outdata.FlushCache()
    outdata = None
    
# def create_kml_polygon(corner_coords, output_filename):
    
#     kml = simplekml.Kml()
#     pol = kml.newpolygon(name="Polygon")
#     polygon_coords = corner_coords + [corner_coords[0]]
#     pol.outerboundaryis.coords = polygon_coords
#     pol.style.polystyle.color = simplekml.Color.changealphaint(150, simplekml.Color.red)  
#     kml.save(output_filename)

def create_extent(annFile):
    inFolder = os.path.dirname(annFile)
    annFile = open(annFile, 'r')
    for line in annFile:
        if "Approximate Upper Left Latitude" in line:
            uly = float(line.split('=')[1].split(';')[0])
        if "Approximate Upper Left Longitude" in line:
            ulx = float(line.split('=')[1].split(';')[0])
        if "Approximate Lower Right Latitude" in line:
            lry = float(line.split('=')[1].split(';')[0])
        if "Approximate Lower Right Longitude" in line:
            lrx = float(line.split('=')[1].split(';')[0])
        if "Approximate Lower Left Latitude" in line:
            lly = float(line.split('=')[1].split(';')[0])
        if "Approximate Lower Left Longitude" in line:
            llx = float(line.split('=')[1].split(';')[0]) 
        if "Approximate Upper Right Longitude" in line:
            urx = float(line.split('=')[1].split(';')[0])
        if "Approximate Upper Right Latitude" in line:
            ury = float(line.split('=')[1].split(';')[0]) 

    # Define polygon coordinates in (longitude, latitude) order
    corner_coordinates = [
        (ulx, uly),
        (urx, ury),
        (lrx, lry),
        (llx, lly),
        (ulx, uly)  # Closing the polygon
    ]

    geojson_polygon = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {
                "name": "EPSG:4326"
            }
        },
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [corner_coordinates]
                }
            }
        ]
    }
    output_file = os.path.join(inFolder, "scene_extent.geojson")
    with open(output_file, 'w') as out_file:
        json.dump(geojson_polygon, out_file, indent=4)


def grdList(annFile):
    grdkeys = {
    'grdHHHH': None,
    'grdHVHV': None,
    'grdVVVV': None,
    'grdHHHV': None,
    'grdHHVV': None,
    'grdHVVV': None
    }

    with open(annFile, 'r') as file:
        for line in file:
            for pattern in grdkeys:
                if pattern in line:
                    parts = line.split()
                    # Find the .grd file associated with the pattern
                    grdkeys[pattern] = next(part for part in parts if '.grd' in part)
    
    return grdkeys                

def mlcList(annFile):
    mlckeys = {
    'mlcHHHH': None,
    'mlcHVHV': None,
    'mlcVVVV': None,
    'mlcHHHV': None,
    'mlcHHVV': None,
    'mlcHVVV': None
    }

    with open(annFile, 'r') as file:
        for line in file:
            for pattern in mlckeys:
                if pattern in line:
                    parts = line.split()
                    # Find the .grd file associated with the pattern
                    mlckeys[pattern] = next(part for part in parts if '.grd' in part)
    
    return mlckeys     


@time_it    
def uavsar_grd(ann,mat='C3',fmt='tif',
            cog=False,ovr = [2, 4, 8, 16],comp=False,
            out_dir = None):
    """
    Extracts specified matrix elements (C3 or T3) from a UAVSAR GRD .ann file and saves them 
    as georeferenced raster files in the specified format.

    Example:
    --------
    >>> uavsar_grd("path_to_file.ann", mat='C3')
    Extracts C3 matrix elements and saves them as GeoTIFFs in the 'C3' directory.

    >>> uavsar_grd("path_to_file.ann", mat='T3', fmt='tif', cog=True, comp=True)
    Extracts T3 matrix elements and saves them as Cloud Optimized GeoTIFFs with compression.

    Parameters:
    -----------
    ann : str
        Path to the UAVSAR annotation file (.ann) containing metadata for the radar data.

    mat : str, optional (default='C3')
        Type of matrix to extract. Must be either 'C3' or 'T3'.

    fmt : str, optional (default='tif')
        Output file format. Currently supports 'tif' (GeoTIFF) and 'bin' (ENVI/PolSARpro/binary).

    cog : bool, optional (default=False)
        If True, output files will be saved as Cloud Optimized GeoTIFFs (COGs) with tiling and overviews.

    ovr : list of int, optional (default=[2, 4, 8, 16])
        Overview levels to generate for COGs. Ignored if cog is False.

    comp : bool, optional (default=False)
        If True, applies LZW compression to reduce file size.

    out_dir : str or None, optional (default=None)
        Directory to save output files. If None, a subdirectory named after the matrix type ('C3' or 'T3') will be created.

    """
    
    inFolder = os.path.dirname(ann)
    grdfiles = grdList(ann)
    create_extent(ann)
    ann_ = open(ann, 'r')
    for line in ann_:
        if "grd_mag.set_rows" in line:
            rows = int(line.split('=')[1].split(';')[0])
        if "grd_mag.set_cols" in line:
            cols = int(line.split('=')[1].split(';')[0])
        if "grd_mag.row_addr" in line:
            lat = float(line.split('=')[1].split(';')[0])
        if "grd_mag.col_addr" in line:
            lon = float(line.split('=')[1].split(';')[0])
        if "grd_mag.row_mult" in line and "(deg/pixel)" in line:
            dy = float(line.split('=')[1].split(';')[0])
        if "grd_mag.col_mult" in line and "(deg/pixel)" in line:
            dx = float(line.split('=')[1].split(';')[0])

    if mat not in ['C3','T3']:
        print("Invalid matrix type. Defaulting to C3")
    if mat=='T3':
        print("Note: First extracting C3 matrix elements and then converting to T3...")
    
    
    if out_dir is None:
        outFolder = inFolder+'/C3'
    else:
        outFolder = out_dir+'/C3'
        
    if not os.path.isdir(outFolder):
        print("C3 folder does not exist. \nCreating folder {}".format(outFolder))
        os.mkdir(outFolder)
    else:
        print("C3 folder exists. \nReplacing C3 elements in folder {}".format(outFolder))

    hhhh = np.fromfile(os.path.join(inFolder,grdfiles['grdHHHH']), dtype='<f',).reshape(rows,cols)
    if fmt=='bin':
        write_bin_uav(outFolder+'/C11.bin',hhhh,lat,lon,dx,dy)
        print(f"Saved file {outFolder}/C11.bin")
        update_hdr(outFolder+'/C11.hdr') 
    else:
        write_geotiff(outFolder+'/C11.tif',hhhh,lat,lon,dx,dy,cog,ovr,comp)
        print(f"Saved file {outFolder}/C11.tif")
    
    del hhhh
    
    vvvv = np.fromfile(os.path.join(inFolder,grdfiles['grdVVVV']), dtype='<f',).reshape(rows,cols)
    if fmt=='bin':
        write_bin_uav(outFolder+'/C33.bin',vvvv,lat,lon,dx,dy)
        print(f"Saved file {outFolder}/C33.bin")
        update_hdr(outFolder+'/C33.hdr')
    else:
        write_geotiff(outFolder+'/C33.tif',vvvv,lat,lon,dx,dy,cog,ovr,comp)
        print(f"Saved file {outFolder}/C33.tif")
        
    del vvvv
    
    
    hvhv = np.fromfile(os.path.join(inFolder,grdfiles['grdHVHV']), dtype='<f',).reshape(rows,cols)
    
    if fmt=='bin':
        write_bin_uav(outFolder+'/C22.bin',2*hvhv,lat,lon,dx,dy)
        print(f"Saved file {outFolder}/C22.bin")
        update_hdr(outFolder+'/C22.hdr')
    else:
        write_geotiff(outFolder+'/C22.tif',2*hvhv,lat,lon,dx,dy,cog,ovr,comp)
        print(f"Saved file {outFolder}/C22.tif")
    del hvhv
    
    
    hhhv = np.fromfile(os.path.join(inFolder,grdfiles['grdHHHV']), dtype='<F',).reshape(rows,cols)
    if fmt=='bin':
        write_bin_uav(outFolder+'/C12_real.bin',np.real(np.sqrt(2)*hhhv),lat,lon,dx,dy)
        print(f"Saved file {outFolder}/C12_real.bin")
        update_hdr(outFolder+'/C12_real.hdr')
        write_bin_uav(outFolder+'/C12_imag.bin',np.imag(np.sqrt(2)*hhhv),lat,lon,dx,dy)
        print(f"Saved file {outFolder}/C12_imag.bin")
        update_hdr(outFolder+'/C12_imag.hdr')
    else:
        write_geotiff(outFolder+'/C12_real.tif',np.real(np.sqrt(2)*hhhv),lat,lon,dx,dy,cog,ovr,comp)
        print(f"Saved file {outFolder}/C12_real.tif")
        write_geotiff(outFolder+'/C12_imag.tif',np.imag(np.sqrt(2)*hhhv),lat,lon,dx,dy,cog,ovr,comp)
        print(f"Saved file {outFolder}/C12_imag.tif")    
    
    del hhhv
    hhvv = np.fromfile(os.path.join(inFolder,grdfiles['grdHHVV']), dtype='<F',).reshape(rows,cols)
    
    if fmt=='bin':
        write_bin_uav(outFolder+'/C13_real.bin',np.real(hhvv),lat,lon,dx,dy)
        print(f"Saved file {outFolder}/C13_real.bin")
        update_hdr(outFolder+'/C13_real.hdr')
        write_bin_uav(outFolder+'/C13_imag.bin',np.imag(hhvv),lat,lon,dx,dy)
        print(f"Saved file {outFolder}/C13_imag.bin")
        update_hdr(outFolder+'/C13_imag.hdr')
    else:
        write_geotiff(outFolder+'/C13_real.tif',np.real(hhvv),lat,lon,dx,dy,cog,ovr,comp)
        print(f"Saved file {outFolder}/C13_real.tif")
        write_geotiff(outFolder+'/C13_imag.tif',np.imag(hhvv),lat,lon,dx,dy,cog,ovr,comp)
        print(f"Saved file {outFolder}/C13_imag.tif")
    del hhvv
    
    hvvv = np.fromfile(os.path.join(inFolder,grdfiles['grdHVVV']), dtype='<F',).reshape(rows,cols)
    
    if fmt=='bin':
        write_bin_uav(outFolder+'/C23_real.bin',np.real(np.sqrt(2)*hvvv),lat,lon,dx,dy)
        print(f"Saved file {outFolder}/C23_real.bin")
        update_hdr(outFolder+'/C23_real.hdr')
        write_bin_uav(outFolder+'/C23_imag.bin',np.imag(np.sqrt(2)*hvvv),lat,lon,dx,dy)
        print(f"Saved file {outFolder}/C23_imag.bin")
        update_hdr(outFolder+'/C23_imag.hdr')
    else:
        write_geotiff(outFolder+'/C23_real.tif',np.real(np.sqrt(2)*hvvv),lat,lon,dx,dy,cog,ovr,comp)
        print(f"Saved file {outFolder}/C23_real.tif")
        write_geotiff(outFolder+'/C23_imag.tif',np.imag(np.sqrt(2)*hvvv),lat,lon,dx,dy,cog,ovr,comp)
        print(f"Saved file {outFolder}/C23_imag.tif")
    del hvvv

    file = open(outFolder +'/config.txt',"w+")
    file.write('Nrow\n%d\n---------\nNcol\n%d\n---------\nPolarCase\nmonostatic\n---------\nPolarType\nfull'%(rows,cols))
    file.close()  
    print("Extracted C3 files to %s"%outFolder)
    
    if mat=='T3':
        print("Converting C3 to T3")
        convert_C3_T3(outFolder)
    

    
@time_it  
def uavsar_mlc(ann,mat='C3',fmt='tif',
            cog=False,ovr = [2, 4, 8, 16],comp=False,
            out_dir = None):
    """
    Extracts specified matrix elements (C3 or T3) from a UAVSAR GRD .ann file and saves them 
    as raster files in the specified format.

    Example:
    --------
    >>> uavsar_mlc("path_to_file.ann", mat='C3')
    Extracts C3 matrix elements and saves them as GeoTIFFs in the 'C3' directory.

    >>> uavsar_mlc("path_to_file.ann", mat='T3', fmt='tif', cog=True, comp=True)
    Extracts T3 matrix elements and saves them as Cloud Optimized GeoTIFFs with compression.

    Parameters:
    -----------
    ann : str
        Path to the UAVSAR annotation file (.ann) containing metadata for the radar data.

    mat : str, optional (default='C3')
        Type of matrix to extract. Must be either 'C3' or 'T3'.

    fmt : str, optional (default='tif')
        Output file format. Currently supports 'tif' (GeoTIFF) and 'bin' (ENVI/PolSARpro/binary).

    cog : bool, optional (default=False)
        If True, output files will be saved as Cloud Optimized GeoTIFFs (COGs) with tiling and overviews.

    ovr : list of int, optional (default=[2, 4, 8, 16])
        Overview levels to generate for COGs. Ignored if cog is False.

    comp : bool, optional (default=False)
        If True, applies LZW compression to reduce file size.

    out_dir : str or None, optional (default=None)
        Directory to save output files. If None, a subdirectory named after the matrix type ('C3' or 'T3') will be created.

    """
    
    create_extent(ann)
    mlcfiles = mlcList(ann)
    inFolder = os.path.dirname(ann)
    create_extent(ann)
    ann_ = open(ann, 'r')
    for line in ann_:
        if "mlc_mag.set_rows" in line:
            rows = int(line.split('=')[1].split(';')[0])
        if "mlc_mag.set_cols" in line:
            cols = int(line.split('=')[1].split(';')[0])
        if "grd_mag.row_addr" in line:
            lat = float(line.split('=')[1].split(';')[0])
        if "grd_mag.col_addr" in line:
            lon = float(line.split('=')[1].split(';')[0])
        if "grd_mag.row_mult" in line and "(deg/pixel)" in line:
            dy = float(line.split('=')[1].split(';')[0])
        if "grd_mag.col_mult" in line and "(deg/pixel)" in line:
            dx = float(line.split('=')[1].split(';')[0])
        # if "set_plon" in line  and "(deg)" in line:
        #     lon = float(line.split('=')[1].split(';')[0])
        # if "set_plat" in line  and "(deg)" in line:
        #     lat = float(line.split('=')[1].split(';')[0])
        # if "mlc_mag.row_mult" in line and "(m/pixel) " in line:
        #     dy = float(line.split('=')[1].split(';')[0])*0.00001
        # if "mlc_mag.col_mult" in line and "(m/pixel) " in line:
        #     dx = float(line.split('=')[1].split(';')[0])*0.00001

    if mat not in ['C3','T3']:
        print("Invalid matrix type. Defaulting to C3")
    if mat=='T3':
        print("Note: First extracting C3 matrix elements and then converting to T3...")
    
    if out_dir is None:
        outFolder = inFolder+'/C3'
    else:
        outFolder = out_dir+'/C3'
        
    if not os.path.isdir(outFolder):
        print("C3 folder does not exist. \nCreating folder {}".format(outFolder))
        os.mkdir(outFolder)
    else:
        print("C3 folder exists. \nReplacing C3 elements in folder {}".format(outFolder))

        
    hhhh = np.fromfile(os.path.join(inFolder,mlcfiles['mlcHHHH']), dtype='<f',).reshape(rows,cols)
    if fmt=='bin':
        write_bin_uav(outFolder+'/C11.bin',hhhh,lat,lon,dx,dy)
        print(f"Saved file {outFolder}/C11.bin")
        update_hdr(outFolder+'/C11.hdr') 
    else:
        write_geotiff(outFolder+'/C11.tif',hhhh,lat,lon,dx,dy,cog,ovr,comp)
        print(f"Saved file {outFolder}/C11.tif")
    del hhhh
    vvvv = np.fromfile(os.path.join(inFolder,mlcfiles['mlcVVVV']), dtype='<f',).reshape(rows,cols)
    if fmt=='bin':
        write_bin_uav(outFolder+'/C33.bin',vvvv,lat,lon,dx,dy)
        print(f"Saved file {outFolder}/C33.bin")
        update_hdr(outFolder+'/C33.hdr')
    else:
        write_geotiff(outFolder+'/C33.tif',vvvv,lat,lon,dx,dy,cog,ovr,comp)
        print(f"Saved file {outFolder}/C33.tif")
    del vvvv
    hvhv = np.fromfile(os.path.join(inFolder,mlcfiles['mlcHVHV']), dtype='<f',).reshape(rows,cols)
    if fmt=='bin':
        write_bin_uav(outFolder+'/C22.bin',2*hvhv,lat,lon,dx,dy)
        print(f"Saved file {outFolder}/C22.bin")
        update_hdr(outFolder+'/C22.hdr')
    else:
        write_geotiff(outFolder+'/C22.tif',2*hvhv,lat,lon,dx,dy,cog,ovr,comp)
        print(f"Saved file {outFolder}/C22.tif")
    del hvhv
    hhhv = np.fromfile(os.path.join(inFolder,mlcfiles['mlcHHHV']), dtype='<F',).reshape(rows,cols)
    if fmt=='bin':
        write_bin_uav(outFolder+'/C12_real.bin',np.real(np.sqrt(2)*hhhv),lat,lon,dx,dy)
        print(f"Saved file {outFolder}/C12_real.bin")
        update_hdr(outFolder+'/C12_real.hdr')
        write_bin_uav(outFolder+'/C12_imag.bin',np.imag(np.sqrt(2)*hhhv),lat,lon,dx,dy)
        print(f"Saved file {outFolder}/C12_imag.bin")
        update_hdr(outFolder+'/C12_imag.hdr')
    else:
        write_geotiff(outFolder+'/C12_real.tif',np.real(np.sqrt(2)*hhhv),lat,lon,dx,dy,cog,ovr,comp)
        print(f"Saved file {outFolder}/C12_real.tif")
        write_geotiff(outFolder+'/C12_imag.tif',np.imag(np.sqrt(2)*hhhv),lat,lon,dx,dy,cog,ovr,comp)
        print(f"Saved file {outFolder}/C12_imag.tif")    
    del hhhv
    hhvv = np.fromfile(os.path.join(inFolder,mlcfiles['mlcHHVV']), dtype='<F',).reshape(rows,cols)
    if fmt=='bin':
        write_bin_uav(outFolder+'/C13_real.bin',np.real(hhvv),lat,lon,dx,dy)
        print(f"Saved file {outFolder}/C13_real.bin")
        update_hdr(outFolder+'/C13_real.hdr')
        write_bin_uav(outFolder+'/C13_imag.bin',np.imag(hhvv),lat,lon,dx,dy)
        print(f"Saved file {outFolder}/C13_imag.bin")
        update_hdr(outFolder+'/C13_imag.hdr')
    else:
        write_geotiff(outFolder+'/C13_real.tif',np.real(hhvv),lat,lon,dx,dy,cog,ovr,comp)
        print(f"Saved file {outFolder}/C13_real.tif")
        write_geotiff(outFolder+'/C13_imag.tif',np.imag(hhvv),lat,lon,dx,dy,cog,ovr,comp)
        print(f"Saved file {outFolder}/C13_imag.tif")
    del hhvv
    hvvv = np.fromfile(os.path.join(inFolder,mlcfiles['mlcHVVV']), dtype='<F',).reshape(rows,cols)
    if fmt=='bin':
        write_bin_uav(outFolder+'/C23_real.bin',np.real(np.sqrt(2)*hvvv),lat,lon,dx,dy)
        print(f"Saved file {outFolder}/C23_real.bin")
        update_hdr(outFolder+'/C23_real.hdr')
        write_bin_uav(outFolder+'/C23_imag.bin',np.imag(np.sqrt(2)*hvvv),lat,lon,dx,dy)
        print(f"Saved file {outFolder}/C23_imag.bin")
        update_hdr(outFolder+'/C23_imag.hdr')
    else:
        write_geotiff(outFolder+'/C23_real.tif',np.real(np.sqrt(2)*hvvv),lat,lon,dx,dy,cog,ovr,comp)
        print(f"Saved file {outFolder}/C23_real.tif")
        write_geotiff(outFolder+'/C23_imag.tif',np.imag(np.sqrt(2)*hvvv),lat,lon,dx,dy,cog,ovr,comp)
        print(f"Saved file {outFolder}/C23_imag.tif")
    del hvvv

    file = open(outFolder +'/config.txt',"w+")
    file.write('Nrow\n%d\n---------\nNcol\n%d\n---------\nPolarCase\nmonostatic\n---------\nPolarType\nfull'%(rows,cols))
    file.close()  
    print("Extracted C3 files to %s"%outFolder)

    if mat=='T3':
        print("Converting C3 to T3")
        convert_C3_T3(outFolder)
