import os
import numpy as np
from osgeo import gdal
gdal.UseExceptions()
def find_file(infolder, base_name):
    for ext in [".bin", ".tif"]:
        path = os.path.join(infolder, f"{base_name}{ext}")
        if os.path.isfile(path):
            return path
    return None

def dxpc2files(infolder):
    """
    Returns file paths for C2 matrix elements, supporting .bin and .tif formats.
    """
    required_keys = ["C11", "C12_real", "C12_imag", "C22"]
    input_filepaths = [find_file(infolder, key) for key in required_keys]

    if all(input_filepaths):
        return input_filepaths
    else:
        print("Invalid C2 folder: missing required files or unsupported formats.")
        return None

def S_norm(S_array):
    S_5 = np.nanpercentile(S_array, 2)
    S_95 = np.nanpercentile(S_array, 98)
    S_cln = np.where(S_array > S_95, S_95, S_array)
    S_cln = np.where(S_cln < S_5, S_5, S_cln)
    S_cln_max = np.nanmax(S_cln)
    S_norm_array = np.divide(S_cln,S_cln_max) 
    
    return S_norm_array


def grd_global_stats(cpFile, xpFile, bins=100000):

    ds1 = gdal.Open(cpFile, gdal.GA_ReadOnly)
    ds2 = gdal.Open(xpFile, gdal.GA_ReadOnly)

    b1 = ds1.GetRasterBand(1)
    b2 = ds2.GetRasterBand(1)

    xsize = b1.XSize
    ysize = b1.YSize
    bx, by = b1.GetBlockSize()

    # ---- PASS 1: global min/max ----
    min_c11, max_c11 = np.inf, -np.inf
    min_c22, max_c22 = np.inf, -np.inf
    min_s1,  max_s1  = np.inf, -np.inf

    for y in range(0, ysize, by):
        rows = min(by, ysize - y)
        for x in range(0, xsize, bx):
            cols = min(bx, xsize - x)

            c11 = b1.ReadAsArray(x, y, cols, rows)
            c22 = b2.ReadAsArray(x, y, cols, rows)

            mask = np.isfinite(c11) & np.isfinite(c22)
            if not np.any(mask):
                continue

            c11 = c11[mask]
            c22 = c22[mask]
            s1  = np.abs(c11 - c22)

            min_c11 = min(min_c11, c11.min())
            max_c11 = max(max_c11, c11.max())

            min_c22 = min(min_c22, c22.min())
            max_c22 = max(max_c22, c22.max())

            min_s1 = min(min_s1, s1.min())
            max_s1 = max(max_s1, s1.max())

    # ---- histograms ----
    bins_c11 = np.linspace(min_c11, max_c11, bins+1)
    bins_c22 = np.linspace(min_c22, max_c22, bins+1)
    bins_s1  = np.linspace(min_s1,  max_s1,  bins+1)

    hist_c11 = np.zeros(bins)
    hist_c22 = np.zeros(bins)
    hist_s1  = np.zeros(bins)

    # ---- PASS 2: histograms ----
    for y in range(0, ysize, by):
        rows = min(by, ysize - y)
        for x in range(0, xsize, bx):
            cols = min(bx, xsize - x)

            c11 = b1.ReadAsArray(x, y, cols, rows)
            c22 = b2.ReadAsArray(x, y, cols, rows)

            mask = np.isfinite(c11) & np.isfinite(c22)
            if not np.any(mask):
                continue

            c11 = c11[mask]
            c22 = c22[mask]
            s1  = np.abs(c11 - c22)

            h, _ = np.histogram(c11, bins=bins_c11)
            hist_c11 += h

            h, _ = np.histogram(c22, bins=bins_c22)
            hist_c22 += h

            h, _ = np.histogram(s1, bins=bins_s1)
            hist_s1 += h

    def get_percentiles(hist, bin_edges):
        cdf = np.cumsum(hist)
        total = cdf[-1]
        p2  = bin_edges[np.searchsorted(cdf, 0.02 * total)]
        p98 = bin_edges[np.searchsorted(cdf, 0.98 * total)]
        return p2, p98

    S5_c11, S95_c11 = get_percentiles(hist_c11, bins_c11)
    S5_c22, S95_c22 = get_percentiles(hist_c22, bins_c22)
    S5_s1,  S95_s1  = get_percentiles(hist_s1,  bins_s1)

    # ---- PASS 3: clipped max ----
    max_c11_clip = -np.inf
    max_c22_clip = -np.inf
    max_s1_clip  = -np.inf

    for y in range(0, ysize, by):
        rows = min(by, ysize - y)
        for x in range(0, xsize, bx):
            cols = min(bx, xsize - x)

            c11 = b1.ReadAsArray(x, y, cols, rows)
            c22 = b2.ReadAsArray(x, y, cols, rows)

            mask = np.isfinite(c11) & np.isfinite(c22)
            if not np.any(mask):
                continue

            c11 = c11[mask]
            c22 = c22[mask]
            s1  = np.abs(c11 - c22)

            max_c11_clip = max(max_c11_clip, np.clip(c11, S5_c11, S95_c11).max())
            max_c22_clip = max(max_c22_clip, np.clip(c22, S5_c22, S95_c22).max())
            max_s1_clip  = max(max_s1_clip,  np.clip(s1,  S5_s1,  S95_s1).max())

    return S5_c11, S95_c11, max_c11_clip, S5_c22, S95_c22, max_c22_clip, S5_s1, S95_s1, max_s1_clip


def stokes_global_stats(c11File, c22File, c12realFile, c12imagFile, bins=500000):

    ds1 = gdal.Open(c11File, gdal.GA_ReadOnly)
    ds2 = gdal.Open(c22File, gdal.GA_ReadOnly)
    ds3 = gdal.Open(c12realFile, gdal.GA_ReadOnly)
    ds4 = gdal.Open(c12imagFile, gdal.GA_ReadOnly)

    b1 = ds1.GetRasterBand(1)
    b2 = ds2.GetRasterBand(1)
    br = ds3.GetRasterBand(1)
    bi = ds4.GetRasterBand(1)

    xsize = b1.XSize
    ysize = b1.YSize
    bx, by = b1.GetBlockSize()

    # -------------------------
    # PASS 1: min/max
    # -------------------------
    min_s = [np.inf]*4
    max_s = [-np.inf]*4

    for y in range(0, ysize, by):
        rows = min(by, ysize - y)
        for x in range(0, xsize, bx):
            cols = min(bx, xsize - x)

            c11 = b1.ReadAsArray(x, y, cols, rows)
            c22 = b2.ReadAsArray(x, y, cols, rows)
            cr  = br.ReadAsArray(x, y, cols, rows)
            ci  = bi.ReadAsArray(x, y, cols, rows)

            mask = np.isfinite(c11) & np.isfinite(c22) & np.isfinite(cr) & np.isfinite(ci)
            if not np.any(mask):
                continue

            c11 = c11[mask]
            c22 = c22[mask]
            c12 = cr[mask] + 1j * ci[mask]

            s0 = np.abs(c11 + c22)
            s1 = np.abs(c11 - c22)
            s2 = np.abs(2 * np.real(c12))
            s3 = np.abs(2 * np.imag(c12))

            for i, s in enumerate([s0, s1, s2, s3]):
                min_s[i] = min(min_s[i], s.min())
                max_s[i] = max(max_s[i], s.max())

    # -------------------------
    # histograms
    # -------------------------
    bins_s0 = np.linspace(min_s[0], max_s[0], bins + 1)
    bins_s1 = np.linspace(min_s[1], max_s[1], bins + 1)
    bins_s2 = np.linspace(min_s[2], max_s[2], bins + 1)
    bins_s3 = np.linspace(min_s[3], max_s[3], bins + 1)

    hist = [np.zeros(bins) for _ in range(4)]

    # -------------------------
    # PASS 2: histogram fill
    # -------------------------
    for y in range(0, ysize, by):
        rows = min(by, ysize - y)
        for x in range(0, xsize, bx):
            cols = min(bx, xsize - x)

            c11 = b1.ReadAsArray(x, y, cols, rows)
            c22 = b2.ReadAsArray(x, y, cols, rows)
            cr  = br.ReadAsArray(x, y, cols, rows)
            ci  = bi.ReadAsArray(x, y, cols, rows)

            mask = np.isfinite(c11) & np.isfinite(c22) & np.isfinite(cr) & np.isfinite(ci)
            if not np.any(mask):
                continue

            c11 = c11[mask]
            c22 = c22[mask]
            c12 = cr[mask] + 1j * ci[mask]

            s_vals = [
                c11 + c22,
                c11 - c22,
                2 * np.real(c12),
                2 * np.imag(c12)
            ]

            for i, s in enumerate(s_vals):
                h, _ = np.histogram(
                    s,
                    bins=[bins_s0, bins_s1, bins_s2, bins_s3][i]
                )
                hist[i] += h

    # -------------------------
    # percentile helper (2%, 98%)
    # -------------------------
    def get_p2_p98(hist, edges):
        cdf = np.cumsum(hist)
        total = cdf[-1]

        p2 = edges[np.searchsorted(cdf, 0.02 * total)]
        p98 = edges[np.searchsorted(cdf, 0.98 * total)]
        return p2, p98

    S0_2, S0_98 = get_p2_p98(hist[0], bins_s0)
    S1_2, S1_98 = get_p2_p98(hist[1], bins_s1)
    S2_2, S2_98 = get_p2_p98(hist[2], bins_s2)
    S3_2, S3_98 = get_p2_p98(hist[3], bins_s3)

    # -------------------------
    # PASS 3: clipped max
    # -------------------------
    max_clip = [-np.inf]*4

    for y in range(0, ysize, by):
        rows = min(by, ysize - y)
        for x in range(0, xsize, bx):
            cols = min(bx, xsize - x)

            c11 = b1.ReadAsArray(x, y, cols, rows)
            c22 = b2.ReadAsArray(x, y, cols, rows)
            cr  = br.ReadAsArray(x, y, cols, rows)
            ci  = bi.ReadAsArray(x, y, cols, rows)

            mask = np.isfinite(c11) & np.isfinite(c22) & np.isfinite(cr) & np.isfinite(ci)
            if not np.any(mask):
                continue

            c11 = c11[mask]
            c22 = c22[mask]
            c12 = cr[mask] + 1j * ci[mask]

            s_vals = [
                c11 + c22,
                c11 - c22,
                2 * np.real(c12),
                2 * np.imag(c12)
            ]

            ranges = [
                (S0_2, S0_98),
                (S1_2, S1_98),
                (S2_2, S2_98),
                (S3_2, S3_98),
            ]

            for i, s in enumerate(s_vals):
                lo, hi = ranges[i]
                max_clip[i] = max(max_clip[i], np.clip(s, lo, hi).max())

    return S0_2, S0_98, max_clip[0], S1_2, S1_98, max_clip[1], S2_2, S2_98, max_clip[2], S3_2, S3_98, max_clip[3]