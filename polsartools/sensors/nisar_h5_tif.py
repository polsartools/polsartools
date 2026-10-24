import os
import numpy as np
import tables
from osgeo import gdal, osr

tables.parameters.PYTABLES_SYS_ATTRS = False

def nisar_h5_tif(h5_path, layer_to_extract=None, out_dir=None, 
                 fmt='tif', cog=False, ovr=[2, 4, 8, 16], comp=False):
 
"""
Extract data layers from NISAR HDF5 files and export them as GeoTIFF,
Cloud Optimized GeoTIFF (COG), or ENVI binary files.

Supported products
------------------
- SME2: Soil Moisture
- GUNW: Geocoded Unwrapped Interferogram
- GOFF: Geocoded Offset

Examples
--------
>>> # Extract a single layer
>>> nisar_h5_tif("file.h5", layer_to_extract="SM")

>>> # Extract multiple layers
>>> nisar_h5_tif(
...     h5_path="file.h5",
...     layer_to_extract=["UWP", "WI", "COH"]
... )

>>> # Extract all supported layers
>>> nisar_h5_tif("file.h5", layer_to_extract="ALL")

Parameters
----------
h5_path : str
    Path to the input NISAR HDF5 file.

layer_to_extract : str, list[str], tuple[str], or None, default=None
    Layer or layers to extract. Accepts a single layer key, a list or tuple
    of layer keys, or "ALL" to extract all supported layers.

output_directory : str or None, default=None
    Parent directory where a subfolder named after the input file is created
    to store the output files. If None, defaults to the input file's directory.

fmt : {'tif', 'bin'}, default='tif'
    Output file format:
    - 'tif': GeoTIFF format
    - 'bin': ENVI binary format

cog : bool, default=False
    If True, generates Cloud Optimized GeoTIFFs with internal tiling
    and overviews.

ovr : list[int], default=[2, 4, 8, 16]
    Overview levels for COG generation. Ignored if cog=False.

comp : bool, default=False
    If True, applies LZW compression to GeoTIFF outputs.

Raises
------
ValueError
    If the input format is unsupported or an invalid layer key is provided.

FileNotFoundError
    If the specified input HDF5 file does not exist.
"""

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"HDF5 file not found at '{h5_path}'.")

    # Inspect internal HDF5 nodes to determine product type
    with tables.open_file(h5_path, mode='r') as h5file:
        is_sme2 = "/science/LSAR/SME2" in h5file
        is_gunw = any(f"{p}/GUNW" in h5file for p in ['/science/LSAR', '/science/SSAR'])
        is_goff = any(f"{p}/GOFF" in h5file for p in ['/science/LSAR', '/science/SSAR'])

    if is_sme2:
        print("Detected Product Type: SME2 (Soil Moisture)")
        _process_sme2(h5_path, layer_to_extract, out_dir, fmt, cog, ovr, comp)
    elif is_gunw:
        print("Detected Product Type: GUNW (Geocoded Unwrapped Interferogram)")
        _process_gunw(h5_path, layer_to_extract, out_dir, fmt, cog, ovr, comp)
    elif is_goff:
        print("Detected Product Type: GOFF (Geocoded Pixel Offsets)")
        _process_goff(h5_path, layer_to_extract, out_dir, fmt, cog, ovr, comp)
    else:
        raise ValueError(
            f"\n[ERROR] Unsupported NISAR product format in file: {os.path.basename(h5_path)}\n"
            f"-> Supported Product Types: SME2, GUNW, GOFF\n"
            f"-> Please ensure the HDF5 structure contains valid paths under '/science/LSAR' or '/science/SSAR'."
        )


def _write_raster(xcoords, ycoords, x_spacing, y_spacing, projection, data_array, tif_path, 
                  fmt='tif', cog=False, ovr=[2, 4, 8, 16], comp=False):
    """Internal helper to write arrays into georeferenced raster files using GDAL."""
    if data_array.ndim > 2:
        data_array = np.squeeze(data_array)
        
    height, width = data_array.shape
    start_x = np.min(xcoords) - (x_spacing / 2.0)
    start_y = np.max(ycoords) + (abs(y_spacing) / 2.0)
    geotransform = (start_x, abs(x_spacing), 0.0, start_y, 0.0, -abs(y_spacing))
    
    gdal_datatype_map = {
        np.dtype('float32'): gdal.GDT_Float32,
        np.dtype('float64'): gdal.GDT_Float64,
        np.dtype('complex64'): gdal.GDT_CFloat32,
        np.dtype('complex128'): gdal.GDT_CFloat64,
        np.dtype('int16'): gdal.GDT_Int16,
        np.dtype('uint16'): gdal.GDT_UInt16,
        np.dtype('int32'): gdal.GDT_Int32,
        np.dtype('uint32'): gdal.GDT_UInt32,
        np.dtype('uint8'): gdal.GDT_Byte
    }
    
    if data_array.dtype not in gdal_datatype_map:
        print(f"[WARNING] Unsupported NumPy data type {data_array.dtype}. Skipping.")
        return
        
    gdal_dtype = gdal_datatype_map[data_array.dtype]
    
    # Adjust path extension if bin format is chosen
    if fmt == 'bin':
        tif_path = os.path.splitext(tif_path)[0] + '.bin'
        driver = gdal.GetDriverByName('ENVI')
    else:
        driver = gdal.GetDriverByName('GTiff')

    options = ['BIGTIFF=IF_SAFER']
    if comp:
        options += ['COMPRESS=LZW']
    if cog:
        options += ['TILED=YES', 'BLOCKXSIZE=512', 'BLOCKYSIZE=512']
    options = options if fmt == 'tif' else []

    out_ds = driver.Create(tif_path, width, height, 1, gdal_dtype, options)
    
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(int(projection))
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    out_ds.SetProjection(srs.ExportToWkt())
    out_ds.SetGeoTransform(geotransform)
    
    print(f"Writing output file: {tif_path}")
    band = out_ds.GetRasterBand(1)
    if np.issubdtype(data_array.dtype, np.floating):
        band.SetNoDataValue(-9999)
        
    band.WriteArray(data_array)
    band.FlushCache()

    # Build overviews if COG is requested
    if cog and fmt == 'tif':
        out_ds.BuildOverviews("NEAREST", ovr)

    out_ds = None
    band = None


def _get_output_directory(h5_path, output_directory):
    """Helper to construct output path with a subfolder named after the HDF5 file."""
    base_name = os.path.basename(h5_path).split('.h5')[0]
    if not output_directory:
        parent_dir = os.path.dirname(h5_path) or "."
        out_dir = os.path.join(parent_dir, base_name)
    else:
        out_dir = os.path.join(output_directory, base_name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _process_sme2(h5_path, layer_to_extract, output_directory, fmt, cog, ovr, comp):
    layer_mapping = {
        "SM": {"name": "Soil Moisture", "path": "/science/LSAR/SME2/grids/soilMoisture"},
        "SMU": {"name": "Soil Moisture Uncertainty", "path": "/science/LSAR/SME2/grids/soilMoistureUncertainty"}
    }
    
    if not layer_to_extract:
        print(f"\n[INFO] No layers specified for SME2 product.")
        print(f"       -> Supported layers for SME2: 'SM', 'SMU', or 'ALL'")
        print(f"       -> Example usage: nisar_h5_tif(h5_path, layer_to_extract='ALL')\n")
        return
    elif isinstance(layer_to_extract, (list, tuple)):
        keys_to_process = [k.upper() for k in layer_to_extract]
        for k in keys_to_process:
            if k not in layer_mapping:
                raise ValueError(f"Invalid SME2 layer '{k}' in list.")
    else:
        l_key = str(layer_to_extract).upper()
        if l_key == "ALL":
            keys_to_process = list(layer_mapping.keys())
        elif l_key in layer_mapping:
            keys_to_process = [l_key]
        else:
            raise ValueError(f"Invalid SME2 layer '{layer_to_extract}'. Choose from: 'SM', 'SMU'")

    out_dir = _get_output_directory(h5_path, output_directory)
    ext = 'bin' if fmt == 'bin' else 'tif'

    with tables.open_file(h5_path, mode='r') as h5file:
        grid_base = "/science/LSAR/SME2/grids"

        for l_key in keys_to_process:
            target_info = layer_mapping[l_key]
            tif_path = os.path.join(out_dir, f"{l_key.lower()}.{ext}")

            print(f"\nProcessing {target_info['name']} ({l_key})...")
            if f"{grid_base}/xCoordinates" not in h5file or target_info["path"] not in h5file:
                print(f"[WARNING] Dataset for {l_key} not found in file. Skipping.")
                continue

            xcoords = h5file.get_node(f"{grid_base}/xCoordinates").read()
            ycoords = h5file.get_node(f"{grid_base}/yCoordinates").read()
            x_spacing = float(h5file.get_node(f"{grid_base}/xCoordinateSpacing").read())
            y_spacing = float(h5file.get_node(f"{grid_base}/yCoordinateSpacing").read())
            projection = int(h5file.get_node(f"{grid_base}/projection").read())
            data_array = h5file.get_node(target_info["path"]).read()

            _write_raster(xcoords, ycoords, x_spacing, y_spacing, projection, data_array, tif_path, fmt, cog, ovr, comp)
            print(f"Successfully exported {l_key}!")


def _process_gunw(h5_path, layer_to_extract, output_directory, fmt, cog, ovr, comp):
    layer_mapping = {
        "UWP": {"name": "Unwrapped Phase (HH)", "subgrid": "unwrappedInterferogram", "path": "unwrappedInterferogram/HH/unwrappedPhase"},
        "WI": {"name": "Wrapped Interferogram (HH)", "subgrid": "wrappedInterferogram", "path": "wrappedInterferogram/HH/wrappedInterferogram"},
        "COH": {"name": "Coherence Magnitude (HH)", "subgrid": "unwrappedInterferogram", "path": "unwrappedInterferogram/HH/coherenceMagnitude"},
        "ION": {"name": "Ionosphere Phase Screen (HH)", "subgrid": "unwrappedInterferogram", "path": "unwrappedInterferogram/HH/ionospherePhaseScreen"},
        "ATO": {"name": "Along Track Pixel Offset (HH)", "subgrid": "pixelOffsets", "path": "pixelOffsets/HH/alongTrackOffset"},
        "SRO": {"name": "Slant Range Pixel Offset (HH)", "subgrid": "pixelOffsets", "path": "pixelOffsets/HH/slantRangeOffset"}
    }
    
    if not layer_to_extract:
        print(f"\n[INFO] No layers specified for GUNW product.")
        print(f"       -> Supported layers for GUNW: 'UWP', 'WI', 'COH', 'ION', 'ATO', 'SRO', or 'ALL'")
        print(f"       -> Example usage: nisar_h5_tif(h5_path, layer_to_extract=['UWP', 'COH'])\n")
        return
    elif isinstance(layer_to_extract, (list, tuple)):
        keys_to_process = [k.upper() for k in layer_to_extract]
        for k in keys_to_process:
            if k not in layer_mapping:
                raise ValueError(f"Invalid GUNW layer '{k}' in list.")
    else:
        l_key = str(layer_to_extract).upper()
        if l_key == "ALL":
            keys_to_process = list(layer_mapping.keys())
        elif l_key in layer_mapping:
            keys_to_process = [l_key]
        else:
            raise ValueError(f"Invalid GUNW layer '{layer_to_extract}'.")

    out_dir = _get_output_directory(h5_path, output_directory)
    ext = 'bin' if fmt == 'bin' else 'tif'

    with tables.open_file(h5_path, mode='r') as h5file:
        freq_path = '/science/LSAR' if '/science/LSAR' in h5file else '/science/SSAR'
        grid_base = f"{freq_path}/GUNW/grids/frequencyA"

        for l_key in keys_to_process:
            target_info = layer_mapping[l_key]
            tif_path = os.path.join(out_dir, f"{l_key.lower()}.{ext}")

            subgrid_base = f"{grid_base}/{target_info['subgrid']}"
            target_node_path = f"{grid_base}/{target_info['path']}"

            print(f"\nProcessing {target_info['name']} ({l_key})...")
            if subgrid_base not in h5file or target_node_path not in h5file:
                print(f"[WARNING] Subgrid or dataset for {l_key} not found in file. Skipping.")
                continue

            xcoords = h5file.get_node(f"{subgrid_base}/xCoordinates").read()
            ycoords = h5file.get_node(f"{subgrid_base}/yCoordinates").read()
            x_spacing = float(h5file.get_node(f"{subgrid_base}/xCoordinateSpacing").read())
            y_spacing = float(h5file.get_node(f"{subgrid_base}/yCoordinateSpacing").read())
            projection = int(h5file.get_node(f"{subgrid_base}/projection").read())
            data_array = h5file.get_node(target_node_path).read()

            _write_raster(xcoords, ycoords, x_spacing, y_spacing, projection, data_array, tif_path, fmt, cog, ovr, comp)
            print(f"Successfully exported {l_key}!")


def _process_goff(h5_path, layer_to_extract, output_directory, fmt, cog, ovr, comp):
    layer_mapping = {}
    for i in [1, 2, 3]:
        layer_mapping.update({
            f"ATO{i}": {"name": f"Layer {i} Along Track Offset (HH)", "subgrid": "pixelOffsets", "path": f"pixelOffsets/HH/layer{i}/alongTrackOffset"},
            f"SRO{i}": {"name": f"Layer {i} Slant Range Offset (HH)", "subgrid": "pixelOffsets", "path": f"pixelOffsets/HH/layer{i}/slantRangeOffset"},
            f"ATOV{i}": {"name": f"Layer {i} Along Track Offset Variance (HH)", "subgrid": "pixelOffsets", "path": f"pixelOffsets/HH/layer{i}/alongTrackOffsetVariance"},
            f"SROV{i}": {"name": f"Layer {i} Slant Range Offset Variance (HH)", "subgrid": "pixelOffsets", "path": f"pixelOffsets/HH/layer{i}/slantRangeOffsetVariance"},
            f"CSP{i}": {"name": f"Layer {i} Correlation Surface Peak (HH)", "subgrid": "pixelOffsets", "path": f"pixelOffsets/HH/layer{i}/correlationSurfacePeak"},
            f"COV{i}": {"name": f"Layer {i} Cross Offset Variance (HH)", "subgrid": "pixelOffsets", "path": f"pixelOffsets/HH/layer{i}/crossOffsetVariance"},
            f"SNR{i}": {"name": f"Layer {i} Signal To Noise Ratio (HH)", "subgrid": "pixelOffsets", "path": f"pixelOffsets/HH/layer{i}/snr"}
        })

    if not layer_to_extract:
        print(f"\n[INFO] No layers specified for GOFF product.")
        print(f"       -> Supported layers for GOFF: Layers 1-3 for ATO, SRO, ATOV, SROV, CSP, COV, SNR (e.g., 'ATO1'), or 'ALL'")
        print(f"       -> Example usage: nisar_h5_tif(h5_path, layer_to_extract='ALL')\n")
        return
    elif isinstance(layer_to_extract, (list, tuple)):
        keys_to_process = [k.upper() for k in layer_to_extract]
        for k in keys_to_process:
            if k not in layer_mapping:
                raise ValueError(f"Invalid GOFF layer '{k}' in list.")
    else:
        l_key = str(layer_to_extract).upper()
        if l_key == "ALL":
            keys_to_process = list(layer_mapping.keys())
        elif l_key in layer_mapping:
            keys_to_process = [l_key]
        else:
            raise ValueError(f"Invalid GOFF layer '{layer_to_extract}'.")

    out_dir = _get_output_directory(h5_path, output_directory)
    ext = 'bin' if fmt == 'bin' else 'tif'

    with tables.open_file(h5_path, mode='r') as h5file:
        freq_path = '/science/LSAR' if '/science/LSAR' in h5file else '/science/SSAR'
        grid_base = f"{freq_path}/GOFF/grids/frequencyA"

        for l_key in keys_to_process:
            target_info = layer_mapping[l_key]
            tif_path = os.path.join(out_dir, f"{l_key.lower()}.{ext}")

            subgrid_base = f"{grid_base}/{target_info['subgrid']}"
            target_node_path = f"{grid_base}/{target_info['path']}"

            print(f"\nProcessing {target_info['name']} ({l_key})...")
            if subgrid_base not in h5file or target_node_path not in h5file:
                print(f"[WARNING] Subgrid or dataset for {l_key} not found in file. Skipping.")
                continue

            xcoords = h5file.get_node(f"{subgrid_base}/xCoordinates").read()
            ycoords = h5file.get_node(f"{subgrid_base}/yCoordinates").read()
            x_spacing = float(h5file.get_node(f"{subgrid_base}/xCoordinateSpacing").read())
            y_spacing = float(h5file.get_node(f"{subgrid_base}/yCoordinateSpacing").read())
            projection = int(h5file.get_node(f"{subgrid_base}/projection").read())
            data_array = h5file.get_node(target_node_path).read()

            _write_raster(xcoords, ycoords, x_spacing, y_spacing, projection, data_array, tif_path, fmt, cog, ovr, comp)
            print(f"Successfully exported {l_key}!")
