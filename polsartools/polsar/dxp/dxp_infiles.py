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

def grd_global_stats(cpFile, xpFile, max_sample_size=1000000):
    ds1 = gdal.Open(cpFile, gdal.GA_ReadOnly)
    ds2 = gdal.Open(xpFile, gdal.GA_ReadOnly)

    b1 = ds1.GetRasterBand(1)
    b2 = ds2.GetRasterBand(1)

    nodata1 = b1.GetNoDataValue()
    nodata2 = b2.GetNoDataValue()

    xsize = b1.XSize
    ysize = b1.YSize
    bx, by = b1.GetBlockSize()

    def load_block(x, y, cols, rows):
        c11 = b1.ReadAsArray(x, y, cols, rows).astype(np.float32)
        c22 = b2.ReadAsArray(x, y, cols, rows).astype(np.float32)

        # Handle explicit NoData, NaNs, and non-positive values
        invalid = np.isnan(c11) | np.isnan(c22) | (c11 <= 0) | (c22 <= 0)
        if nodata1 is not None:
            invalid |= (c11 == nodata1)
        if nodata2 is not None:
            invalid |= (c22 == nodata2)

        valid_mask = ~invalid
        if not np.any(valid_mask):
            return None, None, None

        c11_v = c11[valid_mask]
        c22_v = c22[valid_mask]
        s1_v  = np.abs(c11_v - c22_v)
        return c11_v, c22_v, s1_v

    # ---- PASS: Collect a representative sample for robust percentile calculation ----
    c11_samples = []
    c22_samples = []
    s1_samples  = []
    
    total_valid_pixels = 0

    for y in range(0, ysize, by):
        rows = min(by, ysize - y)
        for x in range(0, xsize, bx):
            cols = min(bx, xsize - x)
            c11, c22, s1 = load_block(x, y, cols, rows)
            if c11 is None:
                continue

            n = c11.size
            total_valid_pixels += n

            # Reservoir sampling or proportional collection to keep memory safe
            # If we haven't hit our sample cap, keep them all; otherwise subsample proportionally
            if sum(len(s) for s in c11_samples) < max_sample_size:
                c11_samples.append(c11)
                c22_samples.append(c22)
                s1_samples.append(s1)
            else:
                # Randomly subsample block to maintain representative distribution without OOM
                idx = np.random.choice(n, size=min(n, 100), replace=False)
                c11_samples.append(c11[idx])
                c22_samples.append(c22[idx])
                s1_samples.append(s1[idx])

    # Concatenate samples
    c11_all = np.concatenate(c11_samples)
    c22_all = np.concatenate(c22_samples)
    s1_all  = np.concatenate(s1_samples)

    # Compute true 2nd and 98th percentiles directly on the distribution
    S2_c11, S98_c11 = np.percentile(c11_all, [2, 98])
    S2_c22, S98_c22 = np.percentile(c22_all, [2, 98])
    S2_s1,  S98_s1  = np.percentile(s1_all,  [2, 98])

    # ---- PASS 2: Find max clipping values excluding outliers above 98th percentile ----
    max_c11_clip = -np.inf
    max_c22_clip = -np.inf
    max_s1_clip  = -np.inf

    for y in range(0, ysize, by):
        rows = min(by, ysize - y)
        for x in range(0, xsize, bx):
            cols = min(bx, xsize - x)
            c11, c22, s1 = load_block(x, y, cols, rows)
            if c11 is None:
                continue

            sub_c11 = c11[c11 <= S98_c11]
            sub_c22 = c22[c22 <= S98_c22]
            sub_s1  = s1[s1 <= S98_s1]

            if sub_c11.size > 0: max_c11_clip = max(max_c11_clip, sub_c11.max())
            if sub_c22.size > 0: max_c22_clip = max(max_c22_clip, sub_c22.max())
            if sub_s1.size > 0:  max_s1_clip  = max(max_s1_clip,  sub_s1.max())

    return S2_c11, S98_c11, max_c11_clip, S2_c22, S98_c22, max_c22_clip, S2_s1, S98_s1, max_s1_clip

def stokes_global_stats(c11File, c22File, c12realFile, c12imagFile, max_sample_size=1000000):
    ds1 = gdal.Open(c11File, gdal.GA_ReadOnly)
    ds2 = gdal.Open(c22File, gdal.GA_ReadOnly)
    ds3 = gdal.Open(c12realFile, gdal.GA_ReadOnly)
    ds4 = gdal.Open(c12imagFile, gdal.GA_ReadOnly)

    b1 = ds1.GetRasterBand(1)
    b2 = ds2.GetRasterBand(1)
    br = ds3.GetRasterBand(1)
    bi = ds4.GetRasterBand(1)

    nodata1 = b1.GetNoDataValue()
    nodata2 = b2.GetNoDataValue()
    nodatar = br.GetNoDataValue()
    nodatai = bi.GetNoDataValue()

    xsize = b1.XSize
    ysize = b1.YSize
    bx, by = b1.GetBlockSize()

    def load_block(x, y, cols, rows):
        c11 = b1.ReadAsArray(x, y, cols, rows).astype(np.float32)
        c22 = b2.ReadAsArray(x, y, cols, rows).astype(np.float32)
        cr  = br.ReadAsArray(x, y, cols, rows).astype(np.float32)
        ci  = bi.ReadAsArray(x, y, cols, rows).astype(np.float32)

        # Handle explicit NoData and NaNs
        invalid = np.isnan(c11) | np.isnan(c22) | np.isnan(cr) | np.isnan(ci)
        if nodata1 is not None: invalid |= (c11 == nodata1)
        if nodata2 is not None: invalid |= (c22 == nodata2)
        if nodatar is not None: invalid |= (cr == nodatar)
        if nodatai is not None: invalid |= (ci == nodatai)

        valid_mask = ~invalid
        if not np.any(valid_mask):
            return None

        c11_v = c11[valid_mask]
        c22_v = c22[valid_mask]
        c12_v = cr[valid_mask] + 1j * ci[valid_mask]

        s0 = c11_v + c22_v
        s1 = c11_v - c22_v
        s2 = 2 * np.real(c12_v)
        s3 = 2 * np.imag(c12_v)

        return np.column_stack((s0, s1, s2, s3))

    # ---------------------------------------------------------
    # PASS 1: Collect a representative sample for robust percentiles
    # ---------------------------------------------------------
    samples = [[] for _ in range(4)]
    block_count = 0

    for y in range(0, ysize, by):
        rows = min(by, ysize - y)
        for x in range(0, xsize, bx):
            cols = min(bx, xsize - x)
            block_s = load_block(x, y, cols, rows)
            if block_s is None:
                continue

            block_count += 1
            n = block_s.shape[0]
            total_collected = sum(len(s) for s in samples[0])
            
            if total_collected < max_sample_size:
                for i in range(4):
                    samples[i].append(block_s[:, i])
            else:
                idx = np.random.choice(n, size=min(n, 100), replace=False)
                for i in range(4):
                    samples[i].append(block_s[idx, i])

    s_all = [np.concatenate(samples[i]) for i in range(4)]

    # print(f"[DEBUG Stokes Pass 1] Processed {block_count} valid blocks. Total sampled pixels: {s_all[0].size}")
    # for i, name in enumerate(['S0', 'S1', 'S2', 'S3']):
    #     print(f"[DEBUG Stokes Pass 1] {name} raw sample min/max: {s_all[i].min()} / {s_all[i].max()}")

    percentiles = []
    for i in range(4):
        p2, p98 = np.percentile(s_all[i], [2, 98])
        percentiles.append((p2, p98))
        # print(f"[DEBUG Stokes Percentiles] S{i} -> 2th: {p2}, 98th: {p98}")

    # ---------------------------------------------------------
    # PASS 2: Find max clipping values excluding extreme outliers
    # ---------------------------------------------------------
    max_clip = [-np.inf] * 4

    for y in range(0, ysize, by):
        rows = min(by, ysize - y)
        for x in range(0, xsize, bx):
            cols = min(bx, xsize - x)
            block_s = load_block(x, y, cols, rows)
            if block_s is None:
                continue

            for i in range(4):
                _, hi = percentiles[i]
                s_col = block_s[:, i]
                sub_s = s_col[s_col <= hi]
                if sub_s.size > 0:
                    max_clip[i] = max(max_clip[i], sub_s.max())

    # for i, name in enumerate(['S0', 'S1', 'S2', 'S3']):
    #     print(f"[DEBUG Stokes Clipped Max] {name} max_clip: {max_clip[i]}")

    return percentiles[0][0], percentiles[0][1], max_clip[0], percentiles[1][0], percentiles[1][1], max_clip[1],percentiles[2][0], percentiles[2][1], max_clip[2],percentiles[3][0], percentiles[3][1], max_clip[3]


# def grd_global_stats(cpFile, xpFile, bins=100000):

#     ds1 = gdal.Open(cpFile, gdal.GA_ReadOnly)
#     ds2 = gdal.Open(xpFile, gdal.GA_ReadOnly)

#     b1 = ds1.GetRasterBand(1)
#     b2 = ds2.GetRasterBand(1)

#     xsize = b1.XSize
#     ysize = b1.YSize
#     bx, by = b1.GetBlockSize()

#     # ---- PASS 1: global min/max ----
#     min_c11, max_c11 = np.inf, -np.inf
#     min_c22, max_c22 = np.inf, -np.inf
#     min_s1,  max_s1  = np.inf, -np.inf

#     for y in range(0, ysize, by):
#         rows = min(by, ysize - y)
#         for x in range(0, xsize, bx):
#             cols = min(bx, xsize - x)

#             c11 = b1.ReadAsArray(x, y, cols, rows)
#             c22 = b2.ReadAsArray(x, y, cols, rows)

#             mask = (c11 == 0) & (c22 == 0)
#             c11[mask], c22[mask] = np.nan, np.nan

#             mask = np.isfinite(c11) & np.isfinite(c22)
#             if not np.any(mask):
#                 continue

#             c11 = c11[mask]
#             c22 = c22[mask]
#             s1  = np.abs(c11 - c22)

#             min_c11 = min(min_c11, c11.min())
#             max_c11 = max(max_c11, c11.max())

#             min_c22 = min(min_c22, c22.min())
#             max_c22 = max(max_c22, c22.max())

#             min_s1 = min(min_s1, s1.min())
#             max_s1 = max(max_s1, s1.max())

#     # ---- histograms ----
#     bins_c11 = np.linspace(min_c11, max_c11, bins+1)
#     bins_c22 = np.linspace(min_c22, max_c22, bins+1)
#     bins_s1  = np.linspace(min_s1,  max_s1,  bins+1)

#     hist_c11 = np.zeros(bins)
#     hist_c22 = np.zeros(bins)
#     hist_s1  = np.zeros(bins)

#     # ---- PASS 2: histograms ----
#     for y in range(0, ysize, by):
#         rows = min(by, ysize - y)
#         for x in range(0, xsize, bx):
#             cols = min(bx, xsize - x)

#             c11 = b1.ReadAsArray(x, y, cols, rows)
#             c22 = b2.ReadAsArray(x, y, cols, rows)

#             mask = np.isfinite(c11) & np.isfinite(c22)
#             if not np.any(mask):
#                 continue

#             c11 = c11[mask]
#             c22 = c22[mask]
#             s1  = np.abs(c11 - c22)

#             h, _ = np.histogram(c11, bins=bins_c11)
#             hist_c11 += h

#             h, _ = np.histogram(c22, bins=bins_c22)
#             hist_c22 += h

#             h, _ = np.histogram(s1, bins=bins_s1)
#             hist_s1 += h

#     def get_percentiles(hist, bin_edges):
#         cdf = np.cumsum(hist)
#         total = cdf[-1]
#         p2  = bin_edges[np.searchsorted(cdf, 0.02 * total)]
#         p98 = bin_edges[np.searchsorted(cdf, 0.98 * total)]
#         return p2, p98

#     S5_c11, S95_c11 = get_percentiles(hist_c11, bins_c11)
#     S5_c22, S95_c22 = get_percentiles(hist_c22, bins_c22)
#     S5_s1,  S95_s1  = get_percentiles(hist_s1,  bins_s1)

#     # ---- PASS 3: clipped max ----
#     max_c11_clip = -np.inf
#     max_c22_clip = -np.inf
#     max_s1_clip  = -np.inf

#     for y in range(0, ysize, by):
#         rows = min(by, ysize - y)
#         for x in range(0, xsize, bx):
#             cols = min(bx, xsize - x)

#             c11 = b1.ReadAsArray(x, y, cols, rows)
#             c22 = b2.ReadAsArray(x, y, cols, rows)

#             mask = np.isfinite(c11) & np.isfinite(c22)
#             if not np.any(mask):
#                 continue

#             c11 = c11[mask]
#             c22 = c22[mask]
#             s1  = np.abs(c11 - c22)

#             max_c11_clip = max(max_c11_clip, np.clip(c11, S5_c11, S95_c11).max())
#             max_c22_clip = max(max_c22_clip, np.clip(c22, S5_c22, S95_c22).max())
#             max_s1_clip  = max(max_s1_clip,  np.clip(s1,  S5_s1,  S95_s1).max())

#     return S5_c11, S95_c11, max_c11_clip, S5_c22, S95_c22, max_c22_clip, S5_s1, S95_s1, max_s1_clip


# def stokes_global_stats(c11File, c22File, c12realFile, c12imagFile, bins=500000):

#     ds1 = gdal.Open(c11File, gdal.GA_ReadOnly)
#     ds2 = gdal.Open(c22File, gdal.GA_ReadOnly)
#     ds3 = gdal.Open(c12realFile, gdal.GA_ReadOnly)
#     ds4 = gdal.Open(c12imagFile, gdal.GA_ReadOnly)

#     b1 = ds1.GetRasterBand(1)
#     b2 = ds2.GetRasterBand(1)
#     br = ds3.GetRasterBand(1)
#     bi = ds4.GetRasterBand(1)

#     xsize = b1.XSize
#     ysize = b1.YSize
#     bx, by = b1.GetBlockSize()

#     # -------------------------
#     # PASS 1: min/max
#     # -------------------------
#     min_s = [np.inf]*4
#     max_s = [-np.inf]*4

#     for y in range(0, ysize, by):
#         rows = min(by, ysize - y)
#         for x in range(0, xsize, bx):
#             cols = min(bx, xsize - x)

#             c11 = b1.ReadAsArray(x, y, cols, rows)
#             c22 = b2.ReadAsArray(x, y, cols, rows)
#             cr  = br.ReadAsArray(x, y, cols, rows)
#             ci  = bi.ReadAsArray(x, y, cols, rows)

#             mask = np.isfinite(c11) & np.isfinite(c22) & np.isfinite(cr) & np.isfinite(ci)
#             if not np.any(mask):
#                 continue

#             c11 = c11[mask]
#             c22 = c22[mask]
#             c12 = cr[mask] + 1j * ci[mask]

#             s0 = np.abs(c11 + c22)
#             s1 = np.abs(c11 - c22)
#             s2 = np.abs(2 * np.real(c12))
#             s3 = np.abs(2 * np.imag(c12))

#             for i, s in enumerate([s0, s1, s2, s3]):
#                 min_s[i] = min(min_s[i], s.min())
#                 max_s[i] = max(max_s[i], s.max())

#     # -------------------------
#     # histograms
#     # -------------------------
#     bins_s0 = np.linspace(min_s[0], max_s[0], bins + 1)
#     bins_s1 = np.linspace(min_s[1], max_s[1], bins + 1)
#     bins_s2 = np.linspace(min_s[2], max_s[2], bins + 1)
#     bins_s3 = np.linspace(min_s[3], max_s[3], bins + 1)

#     hist = [np.zeros(bins) for _ in range(4)]

#     # -------------------------
#     # PASS 2: histogram fill
#     # -------------------------
#     for y in range(0, ysize, by):
#         rows = min(by, ysize - y)
#         for x in range(0, xsize, bx):
#             cols = min(bx, xsize - x)

#             c11 = b1.ReadAsArray(x, y, cols, rows)
#             c22 = b2.ReadAsArray(x, y, cols, rows)
#             cr  = br.ReadAsArray(x, y, cols, rows)
#             ci  = bi.ReadAsArray(x, y, cols, rows)

#             mask = np.isfinite(c11) & np.isfinite(c22) & np.isfinite(cr) & np.isfinite(ci)
#             if not np.any(mask):
#                 continue

#             c11 = c11[mask]
#             c22 = c22[mask]
#             c12 = cr[mask] + 1j * ci[mask]

#             s_vals = [
#                 c11 + c22,
#                 c11 - c22,
#                 2 * np.real(c12),
#                 2 * np.imag(c12)
#             ]

#             for i, s in enumerate(s_vals):
#                 h, _ = np.histogram(
#                     s,
#                     bins=[bins_s0, bins_s1, bins_s2, bins_s3][i]
#                 )
#                 hist[i] += h

#     # -------------------------
#     # percentile helper (2%, 98%)
#     # -------------------------
#     def get_p2_p98(hist, edges):
#         cdf = np.cumsum(hist)
#         total = cdf[-1]

#         p2 = edges[np.searchsorted(cdf, 0.02 * total)]
#         p98 = edges[np.searchsorted(cdf, 0.98 * total)]
#         return p2, p98

#     S0_2, S0_98 = get_p2_p98(hist[0], bins_s0)
#     S1_2, S1_98 = get_p2_p98(hist[1], bins_s1)
#     S2_2, S2_98 = get_p2_p98(hist[2], bins_s2)
#     S3_2, S3_98 = get_p2_p98(hist[3], bins_s3)

#     # -------------------------
#     # PASS 3: clipped max
#     # -------------------------
#     max_clip = [-np.inf]*4

#     for y in range(0, ysize, by):
#         rows = min(by, ysize - y)
#         for x in range(0, xsize, bx):
#             cols = min(bx, xsize - x)

#             c11 = b1.ReadAsArray(x, y, cols, rows)
#             c22 = b2.ReadAsArray(x, y, cols, rows)
#             cr  = br.ReadAsArray(x, y, cols, rows)
#             ci  = bi.ReadAsArray(x, y, cols, rows)

#             mask = np.isfinite(c11) & np.isfinite(c22) & np.isfinite(cr) & np.isfinite(ci)
#             if not np.any(mask):
#                 continue

#             c11 = c11[mask]
#             c22 = c22[mask]
#             c12 = cr[mask] + 1j * ci[mask]

#             s_vals = [
#                 c11 + c22,
#                 c11 - c22,
#                 2 * np.real(c12),
#                 2 * np.imag(c12)
#             ]

#             ranges = [
#                 (S0_2, S0_98),
#                 (S1_2, S1_98),
#                 (S2_2, S2_98),
#                 (S3_2, S3_98),
#             ]

#             for i, s in enumerate(s_vals):
#                 lo, hi = ranges[i]
#                 max_clip[i] = max(max_clip[i], np.clip(s, lo, hi).max())

#     return S0_2, S0_98, max_clip[0], S1_2, S1_98, max_clip[1], S2_2, S2_98, max_clip[2], S3_2, S3_98, max_clip[3]

