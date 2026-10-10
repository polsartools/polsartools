
import geopandas as gpd
import numpy as np
import os, tempfile, warnings
from osgeo import gdal, ogr, osr    
from tqdm import tqdm
import polsartools.utils as utils
warnings.simplefilter("ignore", category=RuntimeWarning)
gdal.UseExceptions()



def verify_and_reproject_vector(raster_path, vector_path):
    # 1. Get Spatial Reference from Raster
    ds_raster = gdal.Open(raster_path)
    if not ds_raster:
        raise ValueError(f"Could not open raster file: {raster_path}")
    
    raster_wkt = ds_raster.GetProjection()
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(raster_wkt)
    ds_raster = None  # Close dataset

    # 2. Get Spatial Reference from Vector
    ds_vector = ogr.Open(vector_path)
    if not ds_vector:
        raise ValueError(f"Could not open vector file: {vector_path}")
    
    layer = ds_vector.GetLayer()
    vector_srs = layer.GetSpatialRef()
    num_points = layer.GetFeatureCount()

    if not vector_srs:
        ds_vector = None
        raise ValueError(f"Vector file has no defined spatial reference: {vector_path}")
    
    ds_vector = None  # Close dataset

    # 3. Extract EPSG codes (tries AutoIdentify if not explicitly embedded)
    raster_epsg = raster_srs.GetAuthorityCode(None)
    if not raster_epsg:
        raster_srs.AutoIdentifyEPSG()
        raster_epsg = raster_srs.GetAuthorityCode(None)

    vector_epsg = vector_srs.GetAuthorityCode(None)
    if not vector_epsg:
        vector_srs.AutoIdentifyEPSG()
        vector_epsg = vector_srs.GetAuthorityCode(None)

    print(f"Raster EPSG: {raster_epsg or 'Unknown (Custom/Local CRS)'}")
    print(f"Vector EPSG: {vector_epsg or 'Unknown (Custom/Local CRS)'}")
    print(f"Vector Feature Count: {num_points} points")

    # 4. Compare Projections Robustly via EPSG Codes or IsSame
    match_found = False
    if raster_epsg and vector_epsg and raster_epsg == vector_epsg:
        match_found = True
    elif raster_srs.IsSame(vector_srs) == 1:
        match_found = True

    if match_found:
        return [vector_path, 0]
    else:
        print("Projections do not match (or definitions differ). Reprojecting vector to match raster...")
        
        temp_dir = tempfile.gettempdir()
        base_name = os.path.basename(vector_path)
        base_name_no_ext = os.path.splitext(base_name)[0]
        temp_vector_path = os.path.join(temp_dir, f"reprojected_{base_name_no_ext}.gpkg")
        
        if os.path.exists(temp_vector_path):
            driver = ogr.GetDriverByName("GPKG")
            if driver:
                driver.DeleteDataSource(temp_vector_path)

        options = gdal.VectorTranslateOptions(
            dstSRS=raster_srs.ExportToWkt(),
            format="GPKG"
        )
        
        reprojected_ds = gdal.VectorTranslate(temp_vector_path, vector_path, options=options)
        if not reprojected_ds:
            raise RuntimeError(f"Failed to reproject vector file: {vector_path}")
        
        reprojected_ds = None  # Close and flush dataset
        return [temp_vector_path, 1]
   
def rst_sample(vector_file, raster_files, output_file, custom_column_names, window=1):

    if isinstance(raster_files, str):
        raster_files = [raster_files]
    elif not isinstance(raster_files, list) or len(raster_files) == 0:
        raise ValueError("raster_files must be a valid file path string or a non-empty list of file paths.")

    first_raster = raster_files[0]

    points = gpd.read_file(vector_file)

    src_ds = gdal.Open(first_raster)
    if src_ds is None:
        raise FileNotFoundError(f"Could not open {first_raster}")
    
    gt = src_ds.GetGeoTransform()
    # Invert the geotransform to convert map coordinates (x, y) to pixel/line (col, row)
    inv_gt = gdal.InvGeoTransform(gt)
    if inv_gt is None:
        raise RuntimeError("Failed to invert geotransform for the reference raster.")
    
    # Calculate row and column indices for each point once
    index_data = []
    for idx, row in points.iterrows():
        x, y = row.geometry.x, row.geometry.y
        col_idx = inv_gt[0] + inv_gt[1] * x + inv_gt[2] * y
        row_idx = inv_gt[3] + inv_gt[4] * x + inv_gt[5] * y
        index_data.append((int(np.floor(row_idx)), int(np.floor(col_idx))))
    
    # Close the reference dataset
    src_ds = None

    # Calculate half-window size for neighborhood extraction
    half_w = window // 2
    if len(raster_files) > 1:
        raster_iterator = tqdm(raster_files, desc="Sampling Rasters")
    else:
        raster_iterator = raster_files

    for raster_file in raster_iterator:

        ds = gdal.Open(raster_file)
        if ds is None:
            print(f"Could not open {raster_file}, skipping.")
            continue
            
        raster_name = custom_column_names.get(os.path.basename(raster_file).split('.tif')[0], raster_file)

        # Read the first band as a NumPy array
        band = ds.GetRasterBand(1)
        raster_data = band.ReadAsArray()
        
        # If the raster uses a specific NoData value, convert them to NaNs
        nodata_val = band.GetNoDataValue()
        if nodata_val is not None:
            raster_data = np.where(raster_data == nodata_val, np.nan, raster_data)

        rows_max, cols_max = raster_data.shape
        raster_values = []
        
        # Loop through each precomputed index to get the raster value or windowed average
        for (row_idx, col_idx) in index_data:
            # Check if the center index is completely out of bounds
            if not (0 <= row_idx < rows_max and 0 <= col_idx < cols_max):
                raster_values.append(np.nan)
                continue
            
            if window <= 1:
                val = raster_data[row_idx, col_idx]
                raster_values.append(np.float32(val) if not np.isnan(val) else np.nan)
            else:
                # Define window bounds with edge clipping
                r_min = max(0, row_idx - half_w)
                r_max = min(rows_max, row_idx + half_w + 1)
                c_min = max(0, col_idx - half_w)
                c_max = min(cols_max, col_idx + half_w + 1)
                
                # Extract the neighborhood window slice
                window_slice = raster_data[r_min:r_max, c_min:c_max]
                
                if window_slice.size > 0:
                    # Compute windowed mean, ignoring NaNs
                    val = np.nanmean(window_slice)
                    raster_values.append(np.float32(val) if not np.isnan(val) else np.nan)
                else:
                    raster_values.append(np.nan)

        # Add raster values to the GeoDataFrame
        points[raster_name] = raster_values
        
        # Close dataset
        ds = None

    # Save the sampled points with the new columns
    points.to_file(output_file)


def create_column_names(raster_files):

    if isinstance(raster_files, str):
        raster_files = [raster_files]
    elif not isinstance(raster_files, list):
        raise ValueError("raster_files must be a valid file path string or a list of file paths.")

    custom_column_names = {}
    for raster_file in raster_files:
        # Strip extension safely using os.path.splitext
        file_no_ext = os.path.splitext(raster_file)[0]
        base_name = os.path.basename(file_no_ext)
        custom_column_names[base_name] = base_name
        
    return custom_column_names

def sample_rasters(raster_files,vector_file,output_file,window=1,custom_column_names=None):

    """
    Samples pixel values from one or more raster files at point locations specified 
    in a vector file, with automatic spatial reference verification and reprojection.

    This function automatically checks if the vector file's coordinate reference system (CRS) 
    matches the reference raster. If they mismatch, it reprojects the vector file on-the-fly 
    to a temporary GeoPackage, extracts the raster values (or windowed averages if `window > 1`), 
    appends them as new attributes, saves the resulting GeoDataFrame, and cleans up any temporary files.

    Parameters:
    -----------
    raster_files : str or list of str
        A single file path string or a list of file paths pointing to the target raster dataset(s).
    vector_file : str
        File path to the input vector dataset (e.g., Shapefile, GeoJSON, GeoPackage) containing point geometries.
    output_file : str
        File path where the final sampled vector dataset will be saved.
    window : int, optional
        The size of the square neighborhood window (in pixels) to sample around each point. 
        - If window=1 (default), it extracts the exact pixel value at the point location.
        - If window > 1, it computes the local spatial mean (ignoring NoData values) within the window.
    custom_column_names : dict, optional
        A dictionary mapping raster base names to custom column names for the output attributes. 
        If None, column names are automatically generated from the raster file names using `create_column_names()`. 
        Example: { "NISAR_HH_band_20260119": "backscatter_hh", "NISAR_HV_band_20260119": "backscatter_hv"}
    
    Returns:
    --------
    None
        The sampled dataset is written directly to `output_file`.
    
    Examples:
    ---------
    >>> # Example 1: Sampling multiple rasters with default settings
    >>> raster_list = ['/ratser1.tif', '/raster2.tif', '/raster3.tif']
    >>> sample_rasters(
    ...     raster_files=raster_list,
    ...     vector_file="test_points.geojson",
    ...     output_file="test_points_sampled.geojson",
    ...     window=1
    ... )

    >>> # Example 2: Using a 3x3 window mean for neighborhood extraction
    >>> sample_rasters(
    ...     raster_files=raster_list,
    ...     vector_file="test_points.geojson",
    ...     output_file="test_points_window_sampled.geojson",
    ...     window=3
    ... )

    >>> # Example 3: Passing a single raster file path string
    >>> sample_rasters(
    ...     raster_files="raster1.tif",
    ...     vector_file="test_points.geojson",
    ...     output_file="single_raster_sampled.geojson",
    ...     window=1
    ... )
    """

    if custom_column_names is None:
        custom_column_names = create_column_names(raster_files)
    if isinstance(raster_files, str):
        first_raster = raster_files
    else:
        first_raster = raster_files[0]

    in_vector_path, reprojected = verify_and_reproject_vector(first_raster, vector_file)

    rst_sample(in_vector_path, raster_files, output_file, custom_column_names,window)

    if reprojected == 1:
        # Clean up the temporary reprojected vector file
        if os.path.exists(in_vector_path):
            os.remove(in_vector_path)