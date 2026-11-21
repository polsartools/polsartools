import os
import math
import numpy as np

from osgeo import gdal

from polsartools.utils.utils import conv2d, time_it
from polsartools.utils.io_utils import mlook_arr
from polsartools.utils.proc_utils import process_chunks_parallel
from polsartools.utils.convert_matrices import C3_T3_mat

gdal.UseExceptions()

from polsartools.preprocess.convert_C import (
    C4_C3_mat,
    C4_C2_mat,
    C3_C2CP_mat,
    process_C3_matrix,
    process_C4_matrix,
    matrix_to_components,
    get_filter_kernel,
    find_file,
    get_output_filepaths
)


T3_KEYS = [
    "T11",
    "T12_real", "T12_imag",
    "T13_real", "T13_imag",
    "T22",
    "T23_real", "T23_imag",
    "T33"
]

T4_KEYS = [
    "T11",
    "T12_real", "T12_imag",
    "T13_real", "T13_imag",
    "T14_real", "T14_imag",
    "T22",
    "T23_real", "T23_imag",
    "T24_real", "T24_imag",
    "T33",
    "T34_real", "T34_imag",
    "T44"
]

SUPPORTED_T_CONVERSIONS = {
    "T3": {
        "T3", "C3", "C2",
        "C2HX", "C2VX", "C2HV", "C2CP"
    },

    "T4": {
        "T4", "C4", "C3",
        "C2", "C2HX", "C2VX",
        "C2HV", "C2CP"
    }
}
def get_t_input_filepaths(in_dir):

    components = {}

    for key in T4_KEYS:
        components[key] = find_file(key, in_dir)

    def complete(keys):
        return all(
            components[k] is not None
            for k in keys
        )

    if complete(T4_KEYS):

        matrix = "T4"
        keys = T4_KEYS

    elif complete(T3_KEYS):

        matrix = "T3"
        keys = T3_KEYS

    else:

        missing = {}

        for name, keys in [
            ("T3", T3_KEYS),
            ("T4", T4_KEYS)
        ]:

            missing[name] = [
                k for k in keys
                if components[k] is None
            ]

        raise FileNotFoundError(
            f"Invalid coherency input directory: {in_dir}\n"
            f"Missing components: {missing}"
        )

    input_filepaths = [
        components[k] for k in keys
    ]

    extensions = {
        os.path.splitext(p)[1].lower()
        for p in input_filepaths
    }

    if len(extensions) != 1:
        raise ValueError(
            "All coherency components must use the same format."
        )

    return input_filepaths
def T3_C3_mat(T):

    U = np.array(
        [
            [1, 0, 1],
            [1, 0, -1],
            [0, np.sqrt(2), 0]
        ],
        dtype=np.float32
    ) / np.sqrt(2.0)

    return np.einsum(
        "ji,jk...,kl->il...",
        U,
        T,
        U
    )
def T4_C4_mat(T):

    U = np.array(
        [
            [1, 0, 0, 1],
            [1, 0, 0, -1],
            [0, 1, 1, 0],
            [0, 1, -1, 0]
        ],
        dtype=np.float32
    ) / np.sqrt(2.0)

    return np.einsum(
        "ji,jk...,kl->il...",
        U,
        T,
        U
    )
def reconstruct_T3(chunks):

    diagonal = [0, 5, 8]

    off_diagonal = {
        (0, 1): (1, 2),
        (0, 2): (3, 4),
        (1, 2): (6, 7)
    }

    shape = np.asarray(chunks[0]).shape

    T = np.zeros(
        (3, 3) + shape,
        dtype=np.complex64
    )

    for i, idx in enumerate(diagonal):

        T[i, i] = chunks[idx]

    for (i, j), (real_idx, imag_idx) in off_diagonal.items():

        value = (
            np.asarray(chunks[real_idx])
            + 1j * np.asarray(chunks[imag_idx])
        )

        T[i, j] = value
        T[j, i] = np.conj(value)

    return T
def reconstruct_T4(chunks):

    diagonal = [0, 7, 12, 15]

    off_diagonal = {
        (0, 1): (1, 2),
        (0, 2): (3, 4),
        (0, 3): (5, 6),
        (1, 2): (8, 9),
        (1, 3): (10, 11),
        (2, 3): (13, 14)
    }

    shape = np.asarray(chunks[0]).shape

    T = np.zeros(
        (4, 4) + shape,
        dtype=np.complex64
    )

    for i, idx in enumerate(diagonal):

        T[i, i] = chunks[idx]

    for (i, j), (real_idx, imag_idx) in off_diagonal.items():

        value = (
            np.asarray(chunks[real_idx])
            + 1j * np.asarray(chunks[imag_idx])
        )

        T[i, j] = value
        T[j, i] = np.conj(value)

    return T
def process_chunk_t3(chunks, *args, **kwargs):

    window_size = args[-6]
    cf = args[-5]
    matrix = args[-4]
    recip = args[-3]
    azlks = args[-2]
    rglks = args[-1]

    kernel = get_filter_kernel(window_size)

    # Reconstruct T3

    T = reconstruct_T3(chunks)

    # Multilooking and filtering

    T = process_C3_matrix(
        T,
        azlks,
        rglks,
        kernel
    )

    # T3 -> C3

    C = T3_C3_mat(T)

    if matrix == "T3":

        return matrix_to_components(T)

    elif matrix == "C3":

        return matrix_to_components(C)

    elif matrix in ("C2", "C2HX", "C2VX", "C2HV"):

        if matrix == "C2HX":
            indices = [0, 1]

        elif matrix == "C2VX":
            indices = [2, 1]

        else:
            indices = [0, 2]

        C2 = C[np.ix_(indices, indices)]

        return matrix_to_components(C2)

    elif matrix == "C2CP":

        Cp = C3_C2CP_mat(
            C,
            kwargs.get("phi", 0.0),
            kwargs.get("chi", 0.0)
        )

        return matrix_to_components(Cp)

    else:

        raise ValueError(
            f"Unsupported T3 conversion: {matrix}"
        )


def process_chunk_t4(chunks, *args, **kwargs):

    window_size = args[-6]
    cf = args[-5]
    matrix = args[-4]
    recip = args[-3]
    azlks = args[-2]
    rglks = args[-1]

    kernel = get_filter_kernel(window_size)

    # Reconstruct T4

    T = reconstruct_T4(chunks)

    # Multilooking and filtering

    T = process_C4_matrix(
        T,
        azlks,
        rglks,
        kernel
    )

    # T4 -> C4

    C = T4_C4_mat(T)

    if matrix == "T4":

        return matrix_to_components(T)

    elif matrix == "C4":

        return matrix_to_components(C)

    elif matrix == "C3":

        C3 = C4_C3_mat(C)

        return matrix_to_components(C3)

    elif matrix in ("C2", "C2HX", "C2VX", "C2HV"):

        C2 = C4_C2_mat(
            C,
            matrix
        )

        return matrix_to_components(C2)

    elif matrix == "C2CP":

        C3 = C4_C3_mat(C)

        Cp = C3_C2CP_mat(
            C3,
            kwargs.get("phi", 0.0),
            kwargs.get("chi", 0.0)
        )

        return matrix_to_components(Cp)

    else:

        raise ValueError(
            f"Unsupported T4 conversion: {matrix}"
        )


@time_it
def convert_T(
    in_dir,
    mat="C3",
    azlks=1,
    rglks=1,
    win=1,
    fmt="tif",
    cog=False,
    ovr=[2, 4, 8, 16],
    comp=False,
    recip=True,
    cf=1,
    out_dir=None,
    max_workers=None,
    block_size=(512, 512),
    phi=0.0,
    chi=0.0,
    progress_callback=None
):

    # Detect input matrix

    input_filepaths = get_t_input_filepaths(in_dir)

    in_mat = {
        9: "T3",
        16: "T4"
    }[len(input_filepaths)]

    print(
        f"Detected {in_mat} matrix and converting to {mat} matrix"
    )

    # Validate conversion

    if mat not in SUPPORTED_T_CONVERSIONS[in_mat]:

        raise NotImplementedError(
            f"Conversion {in_mat} -> {mat} is not supported.\n"
            f"Supported conversions: "
            f"{SUPPORTED_T_CONVERSIONS[in_mat]}"
        )

    # Validate parameters

    if not isinstance(azlks, (int, np.integer)) or azlks < 1:
        raise ValueError("azlks must be a positive integer.")

    if not isinstance(rglks, (int, np.integer)) or rglks < 1:
        raise ValueError("rglks must be a positive integer.")

    if not isinstance(win, (int, np.integer)) or win < 1:
        raise ValueError("win must be a positive integer.")

    if fmt not in ("tif", "bin"):
        raise ValueError("fmt must be 'tif' or 'bin'.")

    if cog and fmt != "tif":
        raise ValueError("COG output requires fmt='tif'.")

    if (
        not isinstance(block_size, (tuple, list))
        or len(block_size) != 2
        or any(
            not isinstance(v, (int, np.integer)) or v <= 0
            for v in block_size
        )
    ):
        raise ValueError(
            "block_size must contain two positive integers."
        )

    # Validate raster properties

    datasets = []

    try:

        for filepath in input_filepaths:

            ds = gdal.Open(
                filepath,
                gdal.GA_ReadOnly
            )

            if ds is None:
                raise FileNotFoundError(filepath)

            datasets.append(ds)

        ref_ds = datasets[0]

        in_cols = ref_ds.RasterXSize
        in_rows = ref_ds.RasterYSize

        in_geotransform = ref_ds.GetGeoTransform()
        in_projection = ref_ds.GetProjection()

        if in_cols < rglks or in_rows < azlks:
            raise ValueError(
                "Multilooking factors exceed raster dimensions."
            )

        for filepath, ds in zip(input_filepaths, datasets):

            if (
                ds.RasterXSize != in_cols
                or ds.RasterYSize != in_rows
            ):
                raise ValueError(
                    f"Raster dimensions mismatch: {filepath}"
                )

            if ds.GetGeoTransform() != in_geotransform:
                raise ValueError(
                    f"Geotransform mismatch: {filepath}"
                )

            if ds.GetProjection() != in_projection:
                raise ValueError(
                    f"Projection mismatch: {filepath}"
                )

    finally:

        datasets.clear()

    # Output dimensions

    out_x_size = in_cols // rglks
    out_y_size = in_rows // azlks

    # Output geotransform

    out_geotransform = list(in_geotransform)

    out_geotransform[1] *= rglks
    out_geotransform[2] *= rglks

    out_geotransform[4] *= azlks
    out_geotransform[5] *= azlks

    out_geotransform = tuple(out_geotransform)

    # Align block size

    def align_block(value, factor):

        return max(
            factor,
            math.ceil(value / factor) * factor
        )

    block_size = (
        align_block(block_size[0], rglks),
        align_block(block_size[1], azlks)
    )

    # Output filepaths

    output_filepaths = get_output_filepaths(
        in_dir,
        out_dir,
        mat,
        fmt
    )

    # Select processing function

    if in_mat == "T3":

        process_func = process_chunk_t3

    else:

        process_func = process_chunk_t4

    # Parallel processing

    process_chunks_parallel(
        input_filepaths,
        list(output_filepaths),

        win,
        True,

        process_func,

        *[
            win,
            cf,
            mat,
            recip,
            azlks,
            rglks,
        ],

        block_size=block_size,
        max_workers=max_workers,
        num_outputs=len(output_filepaths),

        cog=cog,
        ovr=ovr,
        comp=comp,

        out_x_size=out_x_size,
        out_y_size=out_y_size,

        out_geotransform=out_geotransform,
        out_projection=in_projection,

        azlks=azlks,
        rglks=rglks,

        phi=phi,
        chi=chi,

        progress_callback=progress_callback
    )
