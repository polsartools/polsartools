import os
from osgeo import gdal, ogr, osr
import tempfile
from polsartools.utils.utils import time_it


@time_it
def clip(folder_path, output_folder=None, 
         start_x=None, start_y=None, dx=None, dy=None,
         north=None, south=None, east=None, west=None,
         vector_path=None, in_crs=None, outType='tif', auto_reference=True):

    """
    Subsets PolSAR matrix rasters (e.g., C3, C2, T3) from a given folder using spatial or pixel-based criteria.

    This function processes all raster files in the input folder and clips them based on one of three methods:
    - Pixel-based window (`start_x`, `start_y`, `dx`, `dy`)
    - Geographic bounding box (`north`, `south`, `east`, `west`)
    - Vector file mask (`vector_path`)

    The clipped rasters are saved to the specified output folder in either GeoTIFF (.tif) or ENVI binary (.bin) format.

    Examples
    --------
    >>> clip("input_path", "out_path", start_x=100, start_y=200, dx=512, dy=512)
    >>> clip("input_path", "out_path", north=45.0, south=44.5, east=-66.0, west=-67.0)
    >>> clip("input_path", "out_path", vector_path="clip_mask.shp", outType='bin')

    Parameters
    ----------
    folder_path : str
        Path to the input folder containing PolSAR matrix rasters (e.g., C3, C2, T3).
    output_folder : str
        Path to the folder where clipped rasters will be saved. Defaults to a subfolder named "clip" in the input folder.
    start_x : int, optional
        Starting pixel index along the x-axis (column) for pixel-based clipping.
    start_y : int, optional
        Starting pixel index along the y-axis (row) for pixel-based clipping.
    dx : int, optional
        Width of the pixel window to clip.
    dy : int, optional
        Height of the pixel window to clip.
    north : float, optional
        Northern latitude boundary for geographic clipping.
    south : float, optional
        Southern latitude boundary for geographic clipping.
    east : float, optional
        Eastern longitude boundary for geographic clipping.
    west : float, optional
        Western longitude boundary for geographic clipping.
    vector_path : str, optional
        Path to a vector file (e.g., shapefile or GeoJSON) used for spatial masking.
    outType : str, optional
        Output format: 'tif' for GeoTIFF or 'bin' for ENVI binary. Defaults to 'tif'.
    auto_reference : bool, optional
        If True, automatically infers georeferencing from input rasters. Defaults to True.

    Returns
    -------
    None
        The function saves clipped raster files to the specified output folder.

    Notes
    -----
    - Only one clipping method should be used per call: pixel window, geographic bbox, or vector mask.
    - All rasters in the input folder are processed and clipped uniformly.
    - ENVI binary output includes accompanying `.hdr` header files.

    """

    if output_folder is None:
        output_folder = os.path.join(os.path.dirname(folder_path), "clip") 
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Validate clipping mode
    pixel_params = all(v is not None for v in [start_x, start_y, dx, dy])
    geo_params = all(v is not None for v in [north, south, east, west])
    vector_param = vector_path is not None

    mode_count = sum([pixel_params, geo_params, vector_param])
    if mode_count != 1:
        print("Error: Please provide only one clipping method: pixel-based, geo-based (NSEW), or vector file.")
        return

    raster_exts = ['.tif', '.bin']
    shape_counter = {}
    shape_map = {}

    #collect shapes
    for filename in os.listdir(folder_path):
        if not any(filename.lower().endswith(ext) for ext in raster_exts):
            continue
        input_path = os.path.join(folder_path, filename)
        dataset = gdal.Open(input_path)
        if not dataset:
            continue
        shape = (dataset.RasterXSize, dataset.RasterYSize)
        shape_map[filename] = shape
        shape_counter[shape] = shape_counter.get(shape, 0) + 1

    if not shape_counter:
        print("No valid raster files found.")
        return

    ref_shape = max(shape_counter.items(), key=lambda x: x[1])[0]

    # process matching rasters
    for filename in os.listdir(folder_path):
        if not any(filename.lower().endswith(ext) for ext in raster_exts):
            continue
        if shape_map.get(filename) != ref_shape:
            print(f"Skipping {filename}: shape mismatch.")
            continue

        input_path = os.path.join(folder_path, filename)
        dataset = gdal.Open(input_path)
        if not dataset:
            print(f"Failed to open {filename}")
            continue

        geotransform = dataset.GetGeoTransform()
        projection = dataset.GetProjection()
        output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.{outType}")

        if vector_path:
            # Vector-based clipping
            vector_ds = ogr.Open(vector_path)
            layer = vector_ds.GetLayer()
            vector_srs = layer.GetSpatialRef()

            raster_srs = osr.SpatialReference()
            raster_srs.ImportFromWkt(projection)

            if not raster_srs.IsSame(vector_srs):
                print(f"Reprojecting vector to match raster CRS for {filename}...")

                # Create temporary GeoJSON file
                temp_vector_path = os.path.join(
                    tempfile.gettempdir(),
                    f"reprojected_{os.path.splitext(os.path.basename(vector_path))[0]}.geojson"
                )
                driver = ogr.GetDriverByName("GeoJSON")
                if os.path.exists(temp_vector_path):
                    os.remove(temp_vector_path)
                reproj_ds = driver.CreateDataSource(temp_vector_path)
                reproj_layer = reproj_ds.CreateLayer("reprojected", srs=raster_srs, geom_type=layer.GetGeomType())

                coord_transform = osr.CoordinateTransformation(vector_srs, raster_srs)

                layer_defn = layer.GetLayerDefn()
                for i in range(layer_defn.GetFieldCount()):
                    field_defn = layer_defn.GetFieldDefn(i)
                    reproj_layer.CreateField(field_defn)

                for feature in layer:
                    geom = feature.GetGeometryRef()
                    geom.Transform(coord_transform)
                    new_feature = ogr.Feature(reproj_layer.GetLayerDefn())
                    new_feature.SetGeometry(geom)
                    for i in range(layer_defn.GetFieldCount()):
                        new_feature.SetField(layer_defn.GetFieldDefn(i).GetNameRef(), feature.GetField(i))
                    reproj_layer.CreateFeature(new_feature)
                    new_feature = None

                vector_ds = None
                layer = None
                vector_path = temp_vector_path  # Use reprojected GeoJSON

            warp_options = gdal.WarpOptions(
                format='GTiff' if outType == 'tif' else 'ENVI',
                cutlineDSName=vector_path,
                cropToCutline=True,
                dstNodata=0,
                creationOptions=['COMPRESS=DEFLATE', 'PREDICTOR=2', 'ZLEVEL=9', 'TILED=YES'] if outType == 'tif' else None
            )
            gdal.Warp(output_path, dataset, options=warp_options)
            print(f"Saved file {output_path}")
            continue

        # Pixel-based or geo-based clipping
        if pixel_params:
            xoff, yoff = start_x, start_y
            xsize, ysize = dx, dy

        # elif geo_params:
        #     origin_x, pixel_width, _, origin_y, _, pixel_height = geotransform
        #     xoff = int((west - origin_x) / pixel_width)
        #     yoff = int((north - origin_y) / pixel_height)
        #     xsize = int((east - west) / pixel_width)
        #     ysize = int((south - north) / pixel_height)
        elif geo_params:
            raster_srs = osr.SpatialReference()
            raster_srs.ImportFromWkt(projection)

            # If in_crs is None, assume input bounds are already in raster CRS
            if in_crs is None:
                input_srs = raster_srs
            elif isinstance(in_crs, int):
                input_srs = osr.SpatialReference()
                input_srs.ImportFromEPSG(in_crs)
            elif isinstance(in_crs, str):
                input_srs = osr.SpatialReference()
                if in_crs.lower().startswith("epsg:"):
                    input_srs.ImportFromEPSG(int(in_crs.split(":")[1]))
                else:
                    input_srs.ImportFromWkt(in_crs)
            else:
                raise ValueError("in_crs must be None, an EPSG integer, or a WKT string")


            # Build transformation from input CRS to raster CRS
            coord_transform = osr.CoordinateTransformation(input_srs, raster_srs)

            # Create polygon from NSWE bounds in input CRS
            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint(west, south)
            ring.AddPoint(east, south)
            ring.AddPoint(east, north)
            ring.AddPoint(west, north)
            ring.AddPoint(west, south)
            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(ring)

            # Transform polygon to raster CRS (no-op if input_srs == raster_srs)
            poly.Transform(coord_transform)

            # Get bounding box in raster CRS
            min_x, max_x, min_y, max_y = poly.GetEnvelope()

            origin_x, pixel_width, _, origin_y, _, pixel_height = geotransform
            xoff = int((min_x - origin_x) / pixel_width)
            yoff = int((max_y - origin_y) / pixel_height)
            xsize = int((max_x - min_x) / pixel_width)
            ysize = int((max_y - min_y) / abs(pixel_height))



        else:
            print(f"Skipping {filename}: insufficient clipping parameters.")
            continue

        clipped = dataset.ReadAsArray(xoff, yoff, xsize, ysize)
        driver = gdal.GetDriverByName('ENVI' if outType == 'bin' else 'GTiff')
        out_ds = driver.Create(output_path, xsize, ysize, dataset.RasterCount,
                               dataset.GetRasterBand(1).DataType,
                               options=['COMPRESS=DEFLATE', 'PREDICTOR=2', 'ZLEVEL=9', 'TILED=YES'] if outType == 'tif' else None)

        for i in range(dataset.RasterCount):
            out_ds.GetRasterBand(i + 1).WriteArray(clipped[i] if dataset.RasterCount > 1 else clipped)

        new_gt = (
            geotransform[0] + xoff * geotransform[1],
            geotransform[1],
            geotransform[2],
            geotransform[3] + yoff * geotransform[5],
            geotransform[4],
            geotransform[5]
        )
        out_ds.SetGeoTransform(new_gt)
        out_ds.SetProjection(projection)
        out_ds.FlushCache()
        print(f"Saved file {output_path}")