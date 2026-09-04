
#%%
import os
import requests
import gzip
import shutil
import zipfile
import numpy as np
from osgeo import gdal
from tqdm import tqdm
gdal.UseExceptions()

from polsartools.utils.utils import time_it


# === DEM Metadata Dictionary ===
DEM_SOURCES = {
    "SRTM1": {
        "url_template": "https://s3.amazonaws.com/elevation-tiles-prod/skadi/{prefix}/{name}.hgt.gz",
        "ext": ".hgt.gz",
        "extract": "gzip"
    },
    "COP30": {
        "api_template": "https://portal.opentopography.org/API/globaldem?demtype=COP30&south={south}&north={north}&west={west}&east={east}&outputFormat=GTiff&API_Key={api_key}",
        "ext": ".tif",
        "extract": "direct"
    },
    "COP90": {
        "api_template": "https://portal.opentopography.org/API/globaldem?demtype=COP90&south={south}&north={north}&west={west}&east={east}&outputFormat=GTiff&API_Key={api_key}",
        "ext": ".tif",
        "extract": "direct"
    },
    "SRTM3": {
        "url_template": "https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF/{name}.zip",
        "ext": ".zip",
        "extract": "zip"
    }
}

def tile_name(lat, lon):
    ns = 'N' if lat >= 0 else 'S'
    ew = 'E' if lon >= 0 else 'W'
    return f"{ns}{abs(int(lat)):02d}{ew}{abs(int(lon)):03d}"

def cgiar_tile_name(lat, lon):
    
    lat = lat+5 #right bottom indexing 
    # print(lat,lon)
    
    lat_base = (lat // 5) * 5
    lon_base = (lon // 5) * 5
    lat_index = int((60 - lat_base) // 5) + 1  # Top-down indexing
    lon_index = int((lon_base + 180) // 5) + 1  # Left-right indexing
    return f"srtm_{lon_index:02d}_{lat_index:02d}"


def download_tile(name, hgt_dir, dem_type="SRTM1"):
    meta = DEM_SOURCES.get(dem_type)
    if not meta:
        raise ValueError(f"Unsupported DEM type: {dem_type}")

    hgt_path = os.path.join(hgt_dir, f"{name}.hgt")
    tif_path = os.path.join(hgt_dir, f"{name}.tif")

    # === Copernicus DEMs via OpenTopography API ===
    if dem_type.startswith("COP"):
        api_key = os.getenv("OPENTOPO_API_KEY")
        if not api_key:
            raise EnvironmentError("Missing OpenTopography API key. Set OPENTOPO_API_KEY.")

        lat = int(name[1:3]) * (1 if name[0] == 'N' else -1)
        lon = int(name[4:7]) * (1 if name[3] == 'E' else -1)
        south, north = lat, lat + 1
        west, east = lon, lon + 1

        url = meta["api_template"].format(south=south, north=north, west=west, east=east, api_key=api_key)

        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200:
                with open(tif_path, "wb") as f:
                    f.write(r.content)
                # print(f"Downloaded: {name} ({dem_type})")
                return tif_path
            else:
                print(f"Copernicus tile not found: {name}")
                return None
        except Exception as e:
            print(f"Error downloading Copernicus tile {name}: {e}")
            return None

    # === CGIAR SRTM90 ===
    if dem_type == "SRTM3":
        url = meta["url_template"].format(name=name)
        archive_path = os.path.join(hgt_dir, f"{name}.zip")

        # Skip download if any matching .tif already exists
        for file in os.listdir(hgt_dir):
            if file.startswith(name) and file.endswith(".tif"):
                # print(f"Already downloaded: {file}")
                return os.path.join(hgt_dir, file)

        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200:
                with open(archive_path, "wb") as f:
                    f.write(r.content)
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(hgt_dir)
                os.remove(archive_path)
                # print(f"Downloaded: {name} (SRTM3)")
                # Return first matching .tif file
                for file in os.listdir(hgt_dir):
                    if file.startswith(name) and file.endswith(".tif"):
                        return os.path.join(hgt_dir, file)
                print(f"No .tif file found after extracting {name}")
                return None
            else:
                print(f"CGIAR tile not found: {name}")
                return None
        except Exception as e:
            print(f"Error downloading CGIAR tile {name}: {e}")
            return None

    # === SRTM1 ===
    prefix = name[:3]
    url = meta["url_template"].format(prefix=prefix, name=name)
    archive_path = os.path.join(hgt_dir, f"{name}{meta['ext']}")

    if os.path.exists(hgt_path):
        # print(f"Already downloaded: {name}")
        return hgt_path

    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            with open(archive_path, "wb") as f:
                f.write(r.content)
            if meta["extract"] == "gzip":
                with gzip.open(archive_path, 'rb') as f_in, open(hgt_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(archive_path)
            # print(f"Downloaded: {name} ({dem_type})")
            return hgt_path
        else:
            print(f"Tile not found: {name}")
            return None
    except Exception as e:
        print(f"Error downloading {name}: {e}")
        return None

def create_empty_hgt(path):
    empty_data = np.full((3601, 3601), -32768, dtype=np.int16)
    with open(path, 'wb') as f:
        f.write(empty_data.tobytes())
    print(f"Created empty HGT tile: {os.path.basename(path)}")
@time_it
def prepare_dem(bbox,out_path="dem.tif", 
                dem_type="SRTM1",
                hgt_dir="hgt_tiles",  
                apply_correction=True,                 
                clip=True,
                clean_tiles=True
                ):
    
    """
    Prepares a Digital Elevation Model (DEM) from downloaded elevation tiles.

    This function downloads elevation tiles based on a bounding box, mosaics them into a single raster,
    optionally applies ellipsoid height correction, clips to the bounding box, and saves the result as a GeoTIFF.
    Temporary tile files can be cleaned up after processing.
    
    Examples
    --------
    >>> bbox = (-70.2, 43.5, -66.2, 45.8)  # xmin, ymin, xmax, ymax
    >>> prepare_dem(bbox, out_path="SRTM1_mosaic_clip.tif")
    >>> prepare_dem(bbox, out_path="SRTM3_mosaic_clip.tif", dem_type="SRTM3")

    Parameters
    ----------
    bbox : tuple of float
        Bounding box in the format (xmin, ymin, xmax, ymax), in geographic coordinates (longitude, latitude).
    out_path : str, optional
        Path to save the output DEM GeoTIFF file. Defaults to "dem.tif".    
    dem_type : str, optional
        Type of DEM source. Options include "SRTM1", "SRTM3", "COP30", "COP90". Defaults to "SRTM1".    
    hgt_dir : str, optional
        Directory to store downloaded tiles. Defaults to "hgt_tiles".
    apply_correction : bool, optional
        Whether to apply WGS84ellipsoid height correction to SRTM tiles. Defaults to True.
    clip : bool, optional
        Whether to clip the output DEM to the bounding box. Defaults to False.
    clean_tiles : bool, optional
        Whether to delete the tile directory after processing. Defaults to True.

    Returns
    -------
    None
        The function saves the DEM to the specified `out_path`.

    Notes
    -----
    - Uses GDAL for mosaicking and warping.
    - Supports ellipsoid correction for SRTM1 and SRTM3.
    - Automatically creates empty tiles for missing SRTM1 data.
    """
    tile_limit=9
    
    os.makedirs(hgt_dir, exist_ok=True)
    xmin, ymin, xmax, ymax = bbox

    if dem_type == "SRTM3":
        lon_range = range(int(np.floor(xmin)), int(np.ceil(xmax)))
        lat_range = range(int(np.floor(ymin)), int(np.ceil(ymax)))
        tiles = set()
        for lon in lon_range:
            for lat in lat_range:
                tiles.add(cgiar_tile_name(lat, lon))
    else:
        lon_range = range(int(np.floor(xmin)), int(np.ceil(xmax)))
        lat_range = range(int(np.floor(ymin)), int(np.ceil(ymax)))
        tiles = [tile_name(lat, lon) for lon in lon_range for lat in lat_range]

    def chunk(lst, size):
        for i in range(0, len(lst), size):
            yield lst[i:i + size]

    hgt_paths = []

    # Create a single progress bar over all tiles
    with tqdm(total=len(tiles), desc="Downloading tiles", unit="tile") as pbar:
        for batch in chunk(list(tiles), tile_limit):
            for name in batch:
                ext = ".tif" if dem_type in ["COP30", "COP90", "SRTM3"] else ".hgt"
                hgt_path = os.path.join(hgt_dir, f"{name}{ext}")

                if os.path.exists(hgt_path):
                    hgt_paths.append(hgt_path)
                    pbar.update(1)
                    continue

                downloaded = download_tile(name, hgt_dir=hgt_dir, dem_type=dem_type)
                if downloaded or (not downloaded and dem_type == "SRTM1"):
                    if not downloaded:
                        create_empty_hgt(hgt_path)
                    hgt_paths.append(hgt_path)

                pbar.update(1)

    print(f"\nProcessing {len(hgt_paths)} tiles...")

    # === Mosaic and optional correction ===
    vrt_path = os.path.splitext(out_path)[0] + ".vrt"  
    gdal.BuildVRT(vrt_path, hgt_paths)

    warp_args = {
        "format": "GTiff",
        "creationOptions": ["BIGTIFF=IF_SAFER"]
    }

    if apply_correction and dem_type in ["SRTM1", "SRTM3"]:
        warp_args.update({
            "srcSRS": "EPSG:4326+5773",
            "dstSRS": "EPSG:4326+4979",
            "warpOptions": ["SOURCE_EXTRA=1000"]
        })

    if clip:
        warp_args["outputBounds"] = (xmin, ymin, xmax, ymax)

    gdal.Warp(out_path, vrt_path, **warp_args)
    label = "with ellipsoid correction" if apply_correction else "without ellipsoid correction"
    clip_label = "clipped to bbox" if clip else "full extent"
    print(f"\nDEM saved to: {out_path} ({label}, {clip_label})")

    # === Clean up ===
    if clean_tiles:
        if os.path.exists(hgt_dir):
            shutil.rmtree(hgt_dir)

# bbox = (-70.2, 43.5, -66.2, 45.8)  # xmin, ymin, xmax, ymax
# prepare_dem(bbox, out_path="srtm1_mosaic_clip.tif", dem_type="SRTM1", clip=True)
# prepare_dem(bbox, out_path="SRTM3_mosaic_clip.tif", dem_type="SRTM3", clip=True)

# # prepare_dem(bbox, out_path="cop30_mosaic.tif", dem_type="COP30")
# # prepare_dem(bbox, out_path="cop90_mosaic.tif", dem_type="COP90")