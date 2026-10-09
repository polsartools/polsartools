

import tables, re, os
import numpy as np
from datetime import datetime
import warnings
from tables.exceptions import DataTypeWarning
warnings.filterwarnings('ignore', category=DataTypeWarning)
import polsartools

def identify_nisar_product(file_path):
    with tables.open_file(file_path, mode="r") as h5file:

        band = "Unknown"
        if "/science/LSAR" in h5file:
            band = "LSAR"
        elif "/science/SSAR" in h5file:
            band = "SSAR"

        product_type = None
        supported_products = ["RSLC", "GSLC", "GCOV"]
        
        if band != "Unknown":
            for prod in supported_products:
                path = f"/science/{band}/{prod}"
                if path in h5file:
                    product_type = prod
                    break
        
        if not product_type:
            product_type = "Unknown"

        return band, product_type

def print_rslc_meta(file_path, band, product_type):

    """Prints a summary of metadata for a given NISAR HDF5 product file.

    This function checks the given NASA-ISRO NISAR product file, 
    identifies its radar band (L-band or S-band) and product type (RSLC, GSLC, 
    or GCOV), and outputs a detailed text summary. 

    The printed report includes:
        - Product identification (Band, Product Type, Polarizations)
        - Acquisition bandwidth and orbit pass direction
        - Acquisition start/end timestamps and calculated duration
        - Spatial grid and pixel spacing parameters (along-track, slant/ground range, 
          or UTM coordinate spacings)
        - Bounding polygon geographic corners (NW, NE, SE, SW)
        - Near, mid, and far-range incidence angle

    Args:
        `product_file` (str or pathlib.Path): Path to the NISAR HDF5 data product file.

    Raises:
        FileNotFoundError: If the specified `product_file` does not exist.
        KeyError: If expected HDF5 internal paths or datasets are missing.
        ValueError: If the product type or band cannot be correctly identified 
            or is unsupported.

    Example:
        >>> nisar_product_info("path/to/NISAR_PRODUCT.h5")
    """
    if band == "LSAR":
        band_prefix = "L"
    elif band == "SSAR":
        band_prefix = "S"
    else:
        print("Unknown band. Cannot proceed.")
        return

    with tables.open_file(file_path, mode="r") as h5file:
        # 1. Read spacing values
        along_track = h5file.get_node(f"/science/{band_prefix}SAR/RSLC/swaths/frequencyA/sceneCenterAlongTrackSpacing").read()
        ground_range = h5file.get_node(f"/science/{band_prefix}SAR/RSLC/swaths/frequencyA/sceneCenterGroundRangeSpacing").read()
        incidence_angle = h5file.get_node(f"/science/{band_prefix}SAR/RSLC/metadata/geolocationGrid/incidenceAngle").read()
        orbit_pass_direction = h5file.get_node(f"/science/{band_prefix}SAR/identification/orbitPassDirection").read()
        slant_range = h5file.get_node(f"/science/{band_prefix}SAR/RSLC/swaths/frequencyA/slantRangeSpacing").read()
        zero_doppler_end_time = h5file.get_node(f"/science/{band_prefix}SAR/identification/zeroDopplerEndTime").read()
        zero_doppler_start_time = h5file.get_node(f"/science/{band_prefix}SAR/identification/zeroDopplerStartTime").read()
        listOfPolarizations = h5file.get_node(f"/science/{band_prefix}SAR/{product_type}/swaths/frequencyA/listOfPolarizations").read()
        listOfPolarizations = listOfPolarizations.astype(str).tolist()
        acquiredRangeBandwidth = h5file.get_node(f"/science/{band_prefix}SAR/{product_type}/swaths/frequencyA/acquiredRangeBandwidth").read()

        num_cols = incidence_angle.shape[1]

        # Calculate the mean of the first, middle, and last columns
        min_angle = np.mean(incidence_angle[:, 0])
        center_angle = np.mean(incidence_angle[:, num_cols // 2])
        max_angle = np.mean(incidence_angle[:, -1])



        # 2. Read and decode the bounding polygon
        polygon_node = h5file.get_node(f"/science/{band_prefix}SAR/identification/boundingPolygon")
        polygon_val = polygon_node.read()
        polygon_str = polygon_val.decode('utf-8') if isinstance(polygon_val, bytes) else str(polygon_val)
        
        # Extract coordinate triplets (Longitude, Latitude, Height)
        coords = re.findall(r"([-+]?\d*\.\d+|\d+)\s+([-+]?\d*\.\d+|\d+)\s+([-+]?\d*\.\d+|\d+)", polygon_str)
        
        lons = [float(c[0]) for c in coords]
        lats = [float(c[1]) for c in coords]
        
        # 3. Derive the 4 bounding rectangle corners (Longitude, Latitude)
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)
        
        corners = {
            "Top-Left (NW)": (min_lon, max_lat),
            "Top-Right (NE)": (max_lon, max_lat),
            "Bottom-Right (SE)": (max_lon, min_lat),
            "Bottom-Left (SW)": (min_lon, min_lat)
        }


        val = orbit_pass_direction.item() if hasattr(orbit_pass_direction, 'item') else orbit_pass_direction
        if isinstance(val, bytes):
            val = val.decode('utf-8')
        time_bytes = zero_doppler_end_time.item()
        time_str = time_bytes.decode('utf-8')
        dt_obj = datetime.fromisoformat(time_str[:26])
        start_bytes = zero_doppler_start_time.item()
        end_bytes = zero_doppler_end_time.item()

        start_str = start_bytes.decode('utf-8')
        end_str = end_bytes.decode('utf-8')

        # 2. Parse into datetime objects (slicing to [:26] avoids nanosecond overflow errors)
        start_dt = datetime.fromisoformat(start_str[:26])
        end_dt = datetime.fromisoformat(end_str[:26])

        # 3. Calculate the time difference (duration)
        duration = end_dt - start_dt
    print(f"{band.split('SAR')[0]}-band {product_type} with polarizations: {', '.join(listOfPolarizations)}")
    print(f"Acquired Range Bandwidth: {acquiredRangeBandwidth/1e6:.2f} MHz")
    print(f"Orbit Pass Direction: {val}")
    print("Acquisition Date/Time (UTC):", dt_obj.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"Acquisition Duration: {duration.total_seconds()} seconds")
    print("-" * 40)
    print(f"Along-Track Spacing:  {along_track:3.3f} m")
    print(f"Ground Range Spacing: {ground_range:3.3f} m")
    print(f"Slant Range Spacing: {slant_range:3.3f} m")
    print("-" * 40)
    print("Coverage Coordinates (Longitude, Latitude):")
    for name, (lon, lat) in corners.items():
        print(f"  {name:<17}: {lon:.6f}, {lat:.6f}")
    print("-" * 40)
    print(f"Near-range Incidence Angle:  {min_angle:.6f}°")
    print(f"mid-range Incidence Angle:  {center_angle:.6f}°")
    print(f"far-range Incidence Angle: {max_angle:.6f}°")

def print_gslc_meta(file_path, band, product_type):

    if band == "LSAR":
        band_prefix = "L"
    elif band == "SSAR":
        band_prefix = "S"
    else:
        print("Unknown band. Cannot proceed.")
        return

    with tables.open_file(file_path, mode="r") as h5file:
        # 1. Read spacing values
        
        along_track = h5file.get_node(f"/science/{band_prefix}SAR/{product_type}/metadata/sourceData/swaths/frequencyA/sceneCenterAlongTrackSpacing").read()
        if product_type == "GSLC":
            slant_range = h5file.get_node(f"/science/{band_prefix}SAR/{product_type}/grids/frequencyA/slantRangeSpacing").read()
        elif product_type == "GCOV":
            slant_range = h5file.get_node(f"/science/{band_prefix}SAR/{product_type}/metadata/sourceData/swaths/frequencyA/slantRangeSpacing").read()
        # ground_range = h5file.get_node(f"/science/{band_prefix}SAR/{product_type}/metadata/sourceData/swaths/frequencyA/sceneCenterGroundRangeSpacing").read()
        x_coordinate_spacing = h5file.get_node(f"/science/{band_prefix}SAR/{product_type}/grids/frequencyA/xCoordinateSpacing").read()
        y_coordinate_spacing = h5file.get_node(f"/science/{band_prefix}SAR/{product_type}/grids/frequencyA/yCoordinateSpacing").read()
        listOfPolarizations = h5file.get_node(f"/science/{band_prefix}SAR/{product_type}/grids/frequencyA/listOfPolarizations").read()
        listOfPolarizations = listOfPolarizations.astype(str).tolist()
        incidence_angle = h5file.get_node(f"/science/{band_prefix}SAR/{product_type}/metadata/radarGrid/incidenceAngle").read()
        orbit_pass_direction = h5file.get_node(f"/science/{band_prefix}SAR/identification/orbitPassDirection").read()

        acquiredRangeBandwidth = h5file.get_node(f"/science/{band_prefix}SAR/{product_type}/metadata/sourceData/swaths/frequencyA/acquiredRangeBandwidth").read()
        zero_doppler_end_time = h5file.get_node(f"/science/{band_prefix}SAR/identification/zeroDopplerEndTime").read()
        zero_doppler_start_time = h5file.get_node(f"/science/{band_prefix}SAR/identification/zeroDopplerStartTime").read()

        num_cols = incidence_angle.shape[1]

        # Calculate the mean of the first, middle, and last columns
        min_angle = np.mean(incidence_angle[:, 0])
        center_angle = np.mean(incidence_angle[:, num_cols // 2])
        max_angle = np.mean(incidence_angle[:, -1])

        # 2. Read and decode the bounding polygon
        polygon_node = h5file.get_node(f"/science/{band_prefix}SAR/identification/boundingPolygon")
        polygon_val = polygon_node.read()
        polygon_str = polygon_val.decode('utf-8') if isinstance(polygon_val, bytes) else str(polygon_val)
        
        # Extract coordinate triplets (Longitude, Latitude, Height)
        coords = re.findall(r"([-+]?\d*\.\d+|\d+)\s+([-+]?\d*\.\d+|\d+)\s+([-+]?\d*\.\d+|\d+)", polygon_str)
        
        lons = [float(c[0]) for c in coords]
        lats = [float(c[1]) for c in coords]
        
        # 3. Derive the 4 bounding rectangle corners (Longitude, Latitude)
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)
        
        corners = {
            "Top-Left (NW)": (min_lon, max_lat),
            "Top-Right (NE)": (max_lon, max_lat),
            "Bottom-Right (SE)": (max_lon, min_lat),
            "Bottom-Left (SW)": (min_lon, min_lat)
        }


        val = orbit_pass_direction.item() if hasattr(orbit_pass_direction, 'item') else orbit_pass_direction
        if isinstance(val, bytes):
            val = val.decode('utf-8')
        time_bytes = zero_doppler_end_time.item()
        time_str = time_bytes.decode('utf-8')
        dt_obj = datetime.fromisoformat(time_str[:26])
        start_bytes = zero_doppler_start_time.item()
        end_bytes = zero_doppler_end_time.item()

        start_str = start_bytes.decode('utf-8')
        end_str = end_bytes.decode('utf-8')

        # 2. Parse into datetime objects (slicing to [:26] avoids nanosecond overflow errors)
        start_dt = datetime.fromisoformat(start_str[:26])
        end_dt = datetime.fromisoformat(end_str[:26])

        # 3. Calculate the time difference (duration)
        duration = end_dt - start_dt
    print(f"{band.split('SAR')[0]}-band {product_type} with polarizations: {', '.join(listOfPolarizations)}")
    print(f"Acquired Range Bandwidth: {acquiredRangeBandwidth/1e6:.2f} MHz")
    print(f"Orbit Pass Direction: {val}")
    print("Acquisition Date/Time (UTC):", dt_obj.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"Acquisition Duration: {duration.total_seconds()} seconds")
    print("-" * 40)
    print(f"Along-Track Spacing:  {along_track:3.3f} m")
    print(f"Slant Range Spacing: {slant_range:3.3f} m")

    print(f"X Coordinate Spacing (ground): {x_coordinate_spacing:3.1f} m")
    print(f"Y Coordinate Spacing (ground): {abs(y_coordinate_spacing):3.1f} m")
    print("-" * 40)
    print("Coverage Coordinates (Longitude, Latitude):")
    for name, (lon, lat) in corners.items():
        print(f"  {name:<17}: {lon:.6f}, {lat:.6f}")
    print("-" * 40)
    print(f"Near-range Incidence Angle:  {min_angle:.6f}°")
    print(f"mid-range Incidence Angle:  {center_angle:.6f}°")
    print(f"far-range Incidence Angle: {max_angle:.6f}°")

def nisar_product_info(product_file):

    band, product_type = identify_nisar_product(product_file)
    print(os.path.basename(product_file).split('.h5')[0])
    print('FrequencyA metadata')
    print("-" * 40)
    if product_type == "RSLC":
        print_rslc_meta(product_file, band, product_type)
    elif product_type == "GSLC" or product_type == "GCOV":
        print_gslc_meta(product_file, band, product_type)
    else:
        print(f"Unsupported product type: {product_type}")
    print("-" * 40)












