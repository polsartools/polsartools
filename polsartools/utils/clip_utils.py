from osgeo import ogr, osr
import os
import numpy as np
from osgeo import gdal


def get_clip_geometry(clip, clip_crs=None):

    # ---------------------------------------
    # CASE 1: Vector file
    # ---------------------------------------
    if isinstance(clip, (str, os.PathLike)):

        vector_path = os.fspath(clip)

        if not os.path.isfile(vector_path):
            raise FileNotFoundError(
                f"Vector file not found: {vector_path}"
            )

        vector_ds = ogr.Open(vector_path, 0)

        if vector_ds is None:
            raise ValueError(
                f"Could not open vector file: {vector_path}"
            )

        layer = vector_ds.GetLayer(0)

        if layer is None:
            raise ValueError("No vector layer found.")

        source_srs = layer.GetSpatialRef()

        if source_srs is None:
            raise ValueError(
                "Vector file has no CRS. Define its CRS before clipping."
            )

        source_srs = source_srs.Clone()

        source_srs.SetAxisMappingStrategy(
            osr.OAMS_TRADITIONAL_GIS_ORDER
        )

        # Combine all features into one geometry
        geometry = None

        layer.ResetReading()

        for feature in layer:

            feature_geom = feature.GetGeometryRef()

            if feature_geom is None:
                continue

            feature_geom = feature_geom.Clone()

            if geometry is None:
                geometry = feature_geom
            else:
                geometry = geometry.Union(feature_geom)

                if geometry is None:
                    raise RuntimeError(
                        "Failed to combine vector geometries."
                    )

        vector_ds = None

        if geometry is None:
            raise ValueError(
                "Vector file contains no valid geometries."
            )

        return geometry, source_srs

    # ---------------------------------------
    # CASE 2: Bounding box
    # ---------------------------------------
    if isinstance(clip, (tuple, list)) and len(clip) == 4:

        if clip_crs is None:
            raise ValueError(
                "clip_crs must be specified for bounding-box input."
            )

        xmin, ymin, xmax, ymax = map(float, clip)

        if xmin >= xmax or ymin >= ymax:
            raise ValueError("Invalid bounding box coordinates.")

        ring = ogr.Geometry(ogr.wkbLinearRing)

        ring.AddPoint(xmin, ymin)
        ring.AddPoint(xmax, ymin)
        ring.AddPoint(xmax, ymax)
        ring.AddPoint(xmin, ymax)
        ring.AddPoint(xmin, ymin)

        geometry = ogr.Geometry(ogr.wkbPolygon)
        geometry.AddGeometry(ring)

        source_srs = osr.SpatialReference()
        source_srs.ImportFromEPSG(int(clip_crs))

        source_srs.SetAxisMappingStrategy(
            osr.OAMS_TRADITIONAL_GIS_ORDER
        )

        return geometry, source_srs

    raise ValueError(
        "clip must be either a bounding-box tuple "
        "or a vector file path."
    )
def apply_vector_mask(raster_path, vector_path, nodata=0):

    src = gdal.Open(raster_path)

    if src is None:
        raise RuntimeError(f"Cannot open raster: {raster_path}")

    # ---------------------------------------
    # Raster CRS
    # ---------------------------------------
    raster_wkt = src.GetProjection()

    if not raster_wkt:
        raise ValueError("Raster has no CRS.")

    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(raster_wkt)

    raster_srs.SetAxisMappingStrategy(
        osr.OAMS_TRADITIONAL_GIS_ORDER
    )

    # ---------------------------------------
    # Vector CRS
    # ---------------------------------------
    vector_ds = ogr.Open(vector_path, 0)

    if vector_ds is None:
        raise RuntimeError(
            f"Cannot open vector file: {vector_path}"
        )

    layer = vector_ds.GetLayer(0)

    if layer is None:
        raise ValueError("No vector layer found.")

    vector_srs = layer.GetSpatialRef()

    if vector_srs is None:
        raise ValueError(
            "Vector file has no CRS. "
            "Define its CRS before clipping."
        )

    vector_srs = vector_srs.Clone()

    vector_srs.SetAxisMappingStrategy(
        osr.OAMS_TRADITIONAL_GIS_ORDER
    )

    # ---------------------------------------
    # CRS comparison
    # ---------------------------------------
    same_crs = bool(
        raster_srs.IsSame(vector_srs)
    )

    # if same_crs:
    #     print("Vector and raster CRS are identical.")
    # else:
    #     print("Vector and raster CRS differ.")
    #     print("Automatic reprojection will be applied.")

    # Close vector dataset
    vector_ds = None

    # ---------------------------------------
    # Preserve raster extent and dimensions
    # ---------------------------------------
    gt = src.GetGeoTransform()

    width = src.RasterXSize
    height = src.RasterYSize

    bounds = [
        gt[0],
        gt[3] + height * gt[5],
        gt[0] + width * gt[1],
        gt[3]
    ]

    # ---------------------------------------
    # Apply polygon mask
    # ---------------------------------------
    temp_path = raster_path + ".mask.tif"

    options = gdal.WarpOptions(
        format="GTiff",
        cutlineDSName=vector_path,

        cropToCutline=False,

        outputBounds=bounds,
        width=width,
        height=height,

        dstSRS=raster_wkt,

        dstNodata=nodata,

        resampleAlg="near",

        cutlineBlend=0
    )

    dst = gdal.Warp(
        temp_path,
        src,
        options=options
    )

    if dst is None:
        src = None
        raise RuntimeError("Vector masking failed.")

    dst = None
    src = None

    os.replace(temp_path, raster_path)

    # print("Vector masking completed successfully.")

# def apply_vector_mask(
#     raster_path,
#     vector_path,
#     nodata=0
# ):
#     """
#     Apply vector cutline while preserving the existing
#     raster dimensions, resolution and geotransform.
#     """

#     src = gdal.Open(raster_path)

#     if src is None:
#         raise RuntimeError(f"Cannot open {raster_path}")

#     gt = src.GetGeoTransform()

#     width = src.RasterXSize
#     height = src.RasterYSize

#     bounds = [
#         gt[0],
#         gt[3] + height * gt[5],
#         gt[0] + width * gt[1],
#         gt[3]
#     ]

#     temp_path = raster_path + ".mask.tif"

#     options = gdal.WarpOptions(
#         format='GTiff',
#         cutlineDSName=vector_path,
#         cropToCutline=False,
#         outputBounds=bounds,
#         width=width,
#         height=height,
#         dstSRS=src.GetProjection(),
#         dstNodata=nodata,
#         resampleAlg='near'
#     )

#     dst = gdal.Warp(
#         temp_path,
#         src,
#         options=options
#     )

#     if dst is None:
#         src = None
#         raise RuntimeError("Vector masking failed.")

#     dst = None
#     src = None

#     os.replace(temp_path, raster_path)

def prepare_clip_window(
    clip,
    clip_crs,
    projection,
    xcoords,
    ycoords,
    azlks,
    rglks
):

    if clip is None:
        return None

    geometry, source_srs = get_clip_geometry(
        clip,
        clip_crs
    )

    geometry = transform_clip_geometry(
        geometry,
        source_srs,
        projection
    )

    minx, maxx, miny, maxy = geometry.GetEnvelope()

    # Raster origin
    start_x = float(xcoords[0])
    start_y = float(ycoords[0])

    # Pixel spacing
    dx = float(np.mean(np.diff(xcoords)))
    dy = float(np.mean(np.diff(ycoords)))

    # Calculate pixel coordinates
    px1 = (minx - start_x) / dx
    px2 = (maxx - start_x) / dx

    py1 = (miny - start_y) / dy
    py2 = (maxy - start_y) / dy

    # Pixel window
    x0 = int(np.floor(min(px1, px2)))
    x1 = int(np.ceil(max(px1, px2)))

    y0 = int(np.floor(min(py1, py2)))
    y1 = int(np.ceil(max(py1, py2)))

    # Clamp
    x0 = max(0, x0)
    y0 = max(0, y0)

    x1 = min(len(xcoords), x1)
    y1 = min(len(ycoords), y1)

    if x0 >= x1 or y0 >= y1:
        raise ValueError("AOI does not intersect raster.")

    # Align to multilook blocks
    x0 = (x0 // rglks) * rglks
    y0 = (y0 // azlks) * azlks

    x1 = min(
        len(xcoords),
        int(np.ceil(x1 / rglks)) * rglks
    )

    y1 = min(
        len(ycoords),
        int(np.ceil(y1 / azlks)) * azlks
    )

    clip_window = (x0, x1, y0, y1)

    # print("\n========== CLIP DEBUG ==========")
    # print("AOI:", minx, maxx, miny, maxy)
    # print("X origin:", start_x)
    # print("Y origin:", start_y)
    # print("dx:", dx)
    # print("dy:", dy)
    # print("Pixel window:", clip_window)
    # print("================================")

    return clip_window


def transform_clip_geometry(geometry, source_srs, target_epsg):

    target_srs = osr.SpatialReference()
    target_srs.ImportFromEPSG(int(target_epsg))

    source_srs = source_srs.Clone()

    source_srs.SetAxisMappingStrategy(
        osr.OAMS_TRADITIONAL_GIS_ORDER
    )

    target_srs.SetAxisMappingStrategy(
        osr.OAMS_TRADITIONAL_GIS_ORDER
    )

    transform = osr.CoordinateTransformation(
        source_srs,
        target_srs
    )

    geometry = geometry.Clone()

    result = geometry.Transform(transform)

    if result != 0:
        raise RuntimeError("Coordinate transformation failed.")

    return geometry
