from osgeo import gdal, osr
gdal.UseExceptions()
import os
import numpy as np
import glob
from tqdm import tqdm 
from xml.etree import cElementTree as ElementTree
from scipy.interpolate import LinearNDInterpolator as interpnd
import warnings
warnings.filterwarnings('ignore')


def read_bin(file):
    ds = gdal.Open(file,gdal.GA_ReadOnly)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    return arr

def write_tif(filepath, array, reference_path=None, 
              fmt='tif', cog=False, ovr=[2, 4, 8, 16], comp=False
              ):
    
    out_dtype = gdal.GDT_Float32
    
    if fmt=='bin':
        driver = gdal.GetDriverByName("ENVI")
        height, width = array.shape

        ds = driver.Create(filepath, width, height, 1, gdal.GDT_Float32)
    else:
        driver = gdal.GetDriverByName("GTiff")
        options = ['BIGTIFF=IF_SAFER']
        if comp:
            options += ['COMPRESS=LZW']
        if cog:
            options += ['TILED=YES', 'BLOCKXSIZE=512', 'BLOCKYSIZE=512']
        
        ds = driver.Create(
            filepath,
            array.shape[1],      
            array.shape[0],      
            1,                   
            out_dtype,
            options    
        )
    

    if reference_path:
        ref = gdal.Open(reference_path)
        ds.SetGeoTransform(ref.GetGeoTransform())
        ds.SetProjection(ref.GetProjection())

    ds.GetRasterBand(1).WriteArray(array)
    ds.FlushCache()
    
    if cog:
        ds.BuildOverviews("NEAREST", ovr)
    ds = None


def warp_with_options(out_file_geo, out_file, xRes, yRes,
                      fmt='tif', cog=False, ovr=[2, 4, 8, 16], comp=False):

    # Build creation options list
    creation_options = ['BIGTIFF=IF_SAFER']
    if comp:
        creation_options.append('COMPRESS=LZW')
    if cog:
        creation_options.extend([
            'TILED=YES',
            'BLOCKXSIZE=512',
            'BLOCKYSIZE=512'
        ])

    # Create WarpOptions with cleaned creationOptions
    warp_options = gdal.WarpOptions(
        format='GTiff' if fmt == 'tif' else 'ENVI',
        xRes=xRes,
        yRes=yRes,
        resampleAlg='average',
        dstSRS='EPSG:4326',
        creationOptions=creation_options
    )

    # Perform warp
    gdal.Warp(out_file_geo, out_file, options=warp_options)

    # Build overviews if COG is requested
    if cog and fmt=='tif':
        ds = gdal.Open(out_file_geo, gdal.GA_Update)
        ds.BuildOverviews("NEAREST", ovr)
        ds = None


class XmlListConfig(list):
    def __init__(self, aList):
        for element in aList:
            if element:
                # treat like dict
                if len(element) == 1 or element[0].tag != element[1].tag:
                    self.append(XmlDictConfig(element))
                # treat like list
                elif element[0].tag == element[1].tag:
                    self.append(XmlListConfig(element))
            elif element.text:
                text = element.text.strip()
                if text:
                    self.append(text)


class XmlDictConfig(dict):
    def __init__(self, parent_element):
        if parent_element.items():
            self.update(dict(parent_element.items()))
        for element in parent_element:
            if element:
                if len(element) == 1 or element[0].tag != element[1].tag:
                    aDict = XmlDictConfig(element)
                else:
                    aDict = {element[0].tag: XmlListConfig(element)}
                if element.items():
                    aDict.update(dict(element.items()))
                self.update({element.tag: aDict})
            elif element.items():
                self.update({element.tag: dict(element.items())})

            else:
                self.update({element.tag: element.text})
                
def radio_cal(incalXml,rows,cols,parm='sigmaNought'):
    """ SIGMA0 CALIBRATION LUT"""
    tree = ElementTree.parse(incalXml)
    root = tree.getroot()
    # xmldict = XmlDictConfig(root)

    caliRoot = root.find('calibrationVectorList')
    # nVectors  = int(caliRoot.items()[0][1])
    xx = [];
    yy = [];
    zz = [];
#     for child in caliRoot.getchildren():
    for child in list(caliRoot):
        line  = int(child.find('line').text)
        pixel = list(map(int, child.find('pixel').text.split()))
        nPixel = int(child.find('pixel').items()[0][1])
        sigmaNought = list(map(float, child.find(parm).text.split()))
        xx = xx + pixel
        yy = yy + [line]*nPixel
        zz = zz + sigmaNought
        
    npt = len(zz)
    coord = np.hstack((np.array(xx).reshape(npt,1),np.array(yy).reshape(npt,1)))
    sigma  = np.array(zz).reshape(npt,1)
    interpfn1  = interpnd(coord,sigma)
    fullX, fullY = np.meshgrid(list(range(cols)), list(range(rows)))
    return interpfn1(fullX, fullY).astype(np.float32)



def s1_grd(in_dir, bsc="sigma0", out_dir=None,pols=None, dB=False, 
           geocode=True,
           xRes = 0.001, yRes=0.001, 
           fmt='tif', cog=False, ovr=[2, 4, 8, 16], comp=False,
           ):
    """
    Convert Sentinel-1 GRD measurement TIFF(s) into radiometrically calibrated 
    GeoTIFF(s) with optional geocoding and formatting.

    This function processes Sentinel-1 Ground Range Detected (GRD) products by:
    - Applying radiometric calibration (sigma0, beta0, or gamma0).
    - Optionally converting values to decibels.
    - Attaching Ground Control Points (GCPs) from annotation XML files.
    - Optionally geocoding to a regular grid with specified resolution using GCPs (Note: may not be accurate for areas with rough terrain).
    - Exporting results in GeoTIFF format with optional Cloud Optimized GeoTIFF (COG) structure.

    Parameters
    ----------
    in_dir : str
        Path to the input directory containing 'measurement' and 'annotation' subfolders.
    bsc : {'sigma0', 'beta0', 'gamma0'}, default='sigma0'
        Backscatter coefficient type to compute.
    out_dir : str, optional
        Output directory for processed files. Defaults to `<in_dir>/processed`.
    pols : list of str, optional
        Polarizations to process (e.g., ['vh', 'vv']). If None, available polarizations 
        are automatically detected.
    dB : bool, default=False
        If True, convert calibrated values to decibels.
    geocode : bool, default=True
        If True, geocode the output to a regular grid using the specified resolution.
    xRes : float, default=0.001
        Output pixel size in longitude units (degrees).
    yRes : float, default=0.001
        Output pixel size in latitude units (degrees).
    fmt : str, default='tif'
        Output raster format (e.g., 'tif', 'bin').
    cog : bool, default=False
        If True, produce Cloud Optimized GeoTIFFs.
    ovr : list of int, default=[2, 4, 8, 16]
        Pyramid overview levels to generate for faster visualization.
    comp : bool, default=False
        If True, apply compression to the output files.
    """
    if out_dir is None:
        out_dir = os.path.join(in_dir, 'processed')
    os.makedirs(out_dir, exist_ok=True)

    # determine polarizations automatically if not provided
    if pols is None:
        measurement_dir = os.path.join(in_dir, 'measurement')
        candidates = glob.glob(os.path.join(measurement_dir, "*.tiff"))
        pols_found = set()
        for c in candidates:
            lower = os.path.basename(c).lower()
            if 'vh' in lower:
                pols_found.add('vh')
            if 'vv' in lower:
                pols_found.add('vv')
            if 'hh' in lower:
                pols_found.add('hh')
            if 'hv' in lower:
                pols_found.add('hv')
        if not pols_found:
            raise FileNotFoundError(f"No measurement TIFFs found in {measurement_dir}")
        pols = sorted(pols_found)

    for pol in pols:
        if geocode:
            steps = ["Calibration", "Writing raster", "Rewriting GCPs", "Geocode using GCPs"]
        else:
            steps = ["Calibration", "Writing raster", "Rewriting GCPs"]
        
        with tqdm(total=len(steps), desc=f"{pol} processing", unit="step") as pbar:
            pbar.set_description(f"{pol}: {steps[0]}")
            # find files
            try:
                inRaster = glob.glob(os.path.join(in_dir, "measurement", f"*{pol}*.tiff"))[0]
            except IndexError:
                print(f"Skipping {pol}: no measurement TIFF found.")
                continue

            try:
                incalXml = glob.glob(os.path.join(in_dir, "annotation", "calibration", f"calibration-*grd-{pol}-*.xml"))[0]
            except IndexError:
                print(f"Skipping {pol}: no calibration XML found for {pol}.")
                continue

            try:
                gcpXml = glob.glob(os.path.join(in_dir, "annotation", f"*grd-{pol}-*.xml"))[0]
            except IndexError:
                print(f"Skipping {pol}: no geolocation XML found for {pol}.")
                continue
            if dB:
                if fmt=='bin':
                    out_file = os.path.join(out_dir, f"{pol}_{bsc}_dB.bin")
                else:
                    out_file = os.path.join(out_dir, f"{pol}_{bsc}_dB.tif")
            else:
                if fmt=='bin':
                    out_file = os.path.join(out_dir, f"{pol}_{bsc}_lin.bin")
                else:
                    out_file = os.path.join(out_dir, f"{pol}_{bsc}_lin.tif")


            ds = gdal.Open(inRaster, gdal.GA_ReadOnly)
            if ds is None:
                print(f"Failed to open raster {inRaster}; skipping {pol}.")
                continue

            cols = ds.RasterXSize
            rows = ds.RasterYSize

            # get calibration/interpolation array from helper radio_cal
            if bsc == "sigma0":
                sigmaIntrp = radio_cal(incalXml, rows, cols, 'sigmaNought')
            elif bsc == "beta0":
                sigmaIntrp = radio_cal(incalXml, rows, cols, 'betaNought')
            elif bsc == "gamma0":
                sigmaIntrp = radio_cal(incalXml, rows, cols, 'gammaNought')
            else:
                ds = None
                raise ValueError("Invalid radiometry type. Valid options: sigma0, gamma0, beta0")

            # read data (first band)
            band_arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
            ds = None  # close original file

            # ensure sigmaIntrp shape compatibility; assume sigmaIntrp[:,:,0] as in original
            try:
                sigma_map = sigmaIntrp[:, :, 0].astype(np.float32)
                del sigmaIntrp
            except Exception:
                raise RuntimeError("sigmaIntrp has unexpected shape; expected (rows, cols, channels)")

            # avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                cal = (band_arr ** 2) / (sigma_map ** 2)
                del band_arr, sigma_map
            if dB:
                # convert to dB, handle non-positive by setting to nan
                with np.errstate(invalid='ignore'):
                    cal = 10.0 * np.log10(cal)
            # replace -inf with nan and any non-finite with nan
            cal[~np.isfinite(cal)] = np.nan
            pbar.update(1)
            pbar.set_description(f"{pol}: {steps[1]}")
            write_tif(out_file, cal, inRaster, fmt=fmt, cog=cog, ovr=ovr, comp=comp)
            del cal
            pbar.update(1)

            pbar.set_description(f"{pol}: {steps[2]}")
            # parse GCPs from geolocation xml and set them on the written file
            tree = ElementTree.parse(gcpXml)
            root = tree.getroot()
            xmldict = XmlDictConfig(root)
            gcpGrid = xmldict.get("geolocationGrid")
            if gcpGrid is None:
                print(f"No geolocationGrid in {gcpXml}; skipping GCP assignment for {pol}.")
                continue

            # xml structure: geolocationGrid -> something -> geolocationGridPoint (list)
            values_container = list(gcpGrid.values())[0]
            values = values_container.get('geolocationGridPoint', [])
            if not values:
                print(f"No geolocationGridPoint entries in {gcpXml}; skipping GCP assignment for {pol}.")
                continue

            gcpList = []
            for v in values:
                try:
                    pixel = int(v.get('pixel'))
                    line = int(v.get('line'))
                    lon = float(v.get('longitude'))
                    lat = float(v.get('latitude'))
                    h = float(v.get('height'))
                    gcpList.append((pixel, line, lon, lat, h))
                except Exception:
                    continue

            if not gcpList:
                print(f"No valid GCPs parsed from {gcpXml}; skipping GCP assignment for {pol}.")
                continue

            # open output for update and set GCPs
            dsOut = gdal.Open(out_file, gdal.GA_Update)
            if dsOut is None:
                print(f"Failed to open output {out_file} for update; GCPs not set.")
                continue

            # create GDAL GCP objects: gdal.GCP(x, y, z, pixel, line)
            gcp_objects = [gdal.GCP(lon, lat, h, pixel, line) for (pixel, line, lon, lat, h) in gcpList]

            srs = osr.SpatialReference()
            srs.ImportFromEPSG(4326)
            srs_wkt = srs.ExportToWkt()

            dsOut.SetGCPs(gcp_objects, srs_wkt)
            dsOut = None  # close dataset to flush changes
            pbar.update(1)

            if geocode:
                pbar.set_description(f"{pol}: {steps[3]}")
                if dB:
                    if fmt=='bin':
                        out_file_geo = os.path.join(out_dir, f"{pol}_{bsc}_dB_geo.bin")
                    else:
                        out_file_geo = os.path.join(out_dir, f"{pol}_{bsc}_dB_geo.tif")
                else:
                    if fmt=='bin':
                        out_file_geo = os.path.join(out_dir, f"{pol}_{bsc}_lin_geo.bin")
                    else:
                        out_file_geo = os.path.join(out_dir, f"{pol}_{bsc}_lin_geo.tif")
                        
                
                warp_with_options(out_file_geo, out_file, xRes, yRes, fmt, cog, ovr, comp)            
                # gdal.Warp(out_file_geo, out_file,     
                # xRes=xRes,
                # yRes=yRes,
                # resampleAlg='average',
                # dstSRS='EPSG:4326'
                # )
                os.remove(out_file)
                pbar.update(1)

    