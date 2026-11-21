import os
import math
import numpy as np

from osgeo import gdal

from polsartools.utils.utils import conv2d, time_it
from polsartools.utils.io_utils import mlook_arr
from polsartools.utils.proc_utils import process_chunks_parallel
from polsartools.utils.convert_matrices import C3_T3_mat

gdal.UseExceptions()


C2_KEYS = [
    "C11", "C12_real", "C12_imag", "C22"
]

C3_KEYS = [
    "C11", "C12_real", "C12_imag",
    "C13_real", "C13_imag",
    "C22", "C23_real", "C23_imag",
    "C33"
]

C4_KEYS = [
    "C11", "C12_real", "C12_imag",
    "C13_real", "C13_imag",
    "C14_real", "C14_imag",
    "C22", "C23_real", "C23_imag",
    "C24_real", "C24_imag",
    "C33", "C34_real", "C34_imag",
    "C44"
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


OUTPUT_KEYS = {
    "T3": [
        "T11", "T12_real", "T12_imag",
        "T13_real", "T13_imag",
        "T22", "T23_real", "T23_imag",
        "T33"
    ],

    "C3": C3_KEYS,

    "C2": C2_KEYS,

    "C2HX": C2_KEYS,
    "C2VX": C2_KEYS,
    "C2HV": C2_KEYS,
    "C2CP": C2_KEYS,

    "T2": [
        "T11", "T12_real",
        "T12_imag", "T22"
    ],

    "T4": T4_KEYS
}


SUPPORTED_CONVERSIONS = {
    "C2": {"C2", "T2"},

    "C3": {
        "T3", "C2",
        "C2HX", "C2VX", "C2HV","C2CP",
    },

    "C4": {
        "T4", "C3",
        "C2", "C2HX",
        "C2VX", "C2HV","C2CP"
    }
}


def find_file(base, in_dir):

    paths = [
        os.path.join(in_dir, base + ".bin"),
        os.path.join(in_dir, base + ".tif")
    ]

    existing = [p for p in paths if os.path.isfile(p)]

    if len(existing) > 1:
        raise ValueError(
            f"Multiple files found for component {base}: {existing}"
        )

    return existing[0] if existing else None


def get_c_input_filepaths(in_dir):

    components = {}

    for key in C4_KEYS:
        components[key] = find_file(key, in_dir)

    def complete(keys):
        return all(components[k] is not None for k in keys)

    if complete(C4_KEYS):
        matrix = "C4"
        keys = C4_KEYS

    elif complete(C3_KEYS):
        matrix = "C3"
        keys = C3_KEYS

    elif complete(C2_KEYS):
        matrix = "C2"
        keys = C2_KEYS

    else:

        missing = {}

        for name, keys in [
            ("C2", C2_KEYS),
            ("C3", C3_KEYS),
            ("C4", C4_KEYS)
        ]:

            absent = [
                k for k in keys
                if components[k] is None
            ]

            missing[name] = absent

        raise FileNotFoundError(
            f"Invalid covariance input directory: {in_dir}\n"
            f"Missing components: {missing}"
        )

    input_filepaths = [
        components[k] for k in keys
    ]

    # Require a consistent input format.
    extensions = {
        os.path.splitext(p)[1].lower()
        for p in input_filepaths
    }

    if len(extensions) != 1:
        raise ValueError(
            "All covariance components must use the same format."
        )

    return input_filepaths


def get_in_matrix_type(in_dir):

    input_filepaths = get_c_input_filepaths(in_dir)

    return {
        4: "C2",
        9: "C3",
        16: "C4"
    }[len(input_filepaths)]


def get_output_filepaths(
    in_dir,
    out_dir,
    matrix,
    fmt
):

    if matrix not in OUTPUT_KEYS:
        raise ValueError(
            f"Unsupported output matrix: {matrix}"
        )

    if fmt not in ("bin", "tif"):
        raise ValueError(
            "fmt must be either 'bin' or 'tif'."
        )

    if out_dir is None:
        out_dir = os.path.join(in_dir, matrix)

    os.makedirs(out_dir, exist_ok=True)

    ext = "." + fmt

    return [
        os.path.join(out_dir, name + ext)
        for name in OUTPUT_KEYS[matrix]
    ]
def apply_multilook(arr, azlks, rglks):

    if azlks == 1 and rglks == 1:
        return arr

    return mlook_arr(
        arr,
        azlks,
        rglks
    )


def apply_filter(arr, kernel):

    if kernel is None:
        return arr

    if np.iscomplexobj(arr):

        return (
            conv2d(arr.real, kernel)
            + 1j * conv2d(arr.imag, kernel)
        )

    return conv2d(arr, kernel)


def get_filter_kernel(window_size):

    if window_size <= 1:
        return None

    return np.ones(
        (window_size, window_size),
        dtype=np.float32
    ) / (window_size * window_size)


def process_dual_pol(
    chunks,
    idx11,
    idx12_real,
    idx12_imag,
    idx22,
    scale12=1.0,
    scale22=1.0,
    azlks=1,
    rglks=1,
    kernel=None
):

    C11 = np.asarray(
        chunks[idx11],
        dtype=np.float32
    )

    C12 = (
        np.asarray(chunks[idx12_real])
        + 1j * np.asarray(chunks[idx12_imag])
    )

    C12 = C12 * scale12

    C22 = (
        np.asarray(chunks[idx22], dtype=np.float32)
        * scale22
    )

    C11 = apply_multilook(C11, azlks, rglks)
    C12 = apply_multilook(C12, azlks, rglks)
    C22 = apply_multilook(C22, azlks, rglks)

    C11 = apply_filter(C11, kernel)
    C12 = apply_filter(C12, kernel)
    C22 = apply_filter(C22, kernel)

    return (
        np.asarray(C11, dtype=np.float32),
        np.asarray(C12.real, dtype=np.float32),
        np.asarray(C12.imag, dtype=np.float32),
        np.asarray(C22, dtype=np.float32)
    )
def process_C3_matrix(C, azlks, rglks, kernel):

    result = np.zeros_like(C)

    for i in range(3):

        for j in range(i, 3):

            value = apply_multilook(
                C[i, j],
                azlks,
                rglks
            )

            value = apply_filter(
                value,
                kernel
            )

            if i == j:
                value = value.real

            result[i, j] = value
            result[j, i] = np.conj(value)

    return result


def process_chunk_c3(chunks, *args, **kwargs):

    window_size = args[-6]
    cf = args[-5]
    matrix = args[-4]
    recip = args[-3]
    azlks = args[-2]
    rglks = args[-1]

    kernel = get_filter_kernel(window_size)

    if matrix == "T3":

        # Construct C3 matrix

        C11 = np.asarray(chunks[0], dtype=np.float32)

        C12 = (
            np.asarray(chunks[1])
            + 1j * np.asarray(chunks[2])
        )

        C13 = (
            np.asarray(chunks[3])
            + 1j * np.asarray(chunks[4])
        )

        C22 = np.asarray(chunks[5], dtype=np.float32)

        C23 = (
            np.asarray(chunks[6])
            + 1j * np.asarray(chunks[7])
        )

        C33 = np.asarray(chunks[8], dtype=np.float32)

        # Apply multilooking

        C11 = apply_multilook(C11, azlks, rglks)
        C12 = apply_multilook(C12, azlks, rglks)
        C13 = apply_multilook(C13, azlks, rglks)
        C22 = apply_multilook(C22, azlks, rglks)
        C23 = apply_multilook(C23, azlks, rglks)
        C33 = apply_multilook(C33, azlks, rglks)

        # Reconstruct Hermitian covariance matrix

        C21 = np.conj(C12)
        C31 = np.conj(C13)
        C32 = np.conj(C23)

        C = np.array([
            [C11, C12, C13],
            [C21, C22, C23],
            [C31, C32, C33]
        ])

        # C3 -> T3 transformation

        T = C3_T3_mat(C)

        # Apply spatial filtering

        if kernel is not None:

            T11 = apply_filter(T[0, 0], kernel)
            T12 = apply_filter(T[0, 1], kernel)
            T13 = apply_filter(T[0, 2], kernel)

            T22 = apply_filter(T[1, 1], kernel)
            T23 = apply_filter(T[1, 2], kernel)

            T33 = apply_filter(T[2, 2], kernel)

            T = np.array([
                [T11, T12, T13],
                [np.conj(T12), T22, T23],
                [np.conj(T13), np.conj(T23), T33]
            ])

        return (
            np.asarray(T[0, 0].real, dtype=np.float32),
            np.asarray(T[0, 1].real, dtype=np.float32),
            np.asarray(T[0, 1].imag, dtype=np.float32),
            np.asarray(T[0, 2].real, dtype=np.float32),
            np.asarray(T[0, 2].imag, dtype=np.float32),
            np.asarray(T[1, 1].real, dtype=np.float32),
            np.asarray(T[1, 2].real, dtype=np.float32),
            np.asarray(T[1, 2].imag, dtype=np.float32),
            np.asarray(T[2, 2].real, dtype=np.float32)
        )

    elif matrix == "C2":

        return process_dual_pol(
            chunks,
            0, 1, 2, 5,
            scale12=1 / np.sqrt(2),
            scale22=0.5,
            azlks=azlks,
            rglks=rglks,
            kernel=kernel
        )

    elif matrix == "C2HX":

        return process_dual_pol(
            chunks,
            0, 1, 2, 5,
            scale12=1 / np.sqrt(2),
            scale22=0.5,
            azlks=azlks,
            rglks=rglks,
            kernel=kernel
        )

    elif matrix == "C2VX":

        return process_dual_pol(
            chunks,
            8, 6, 7, 5,
            scale12=1 / np.sqrt(2),
            scale22=0.5,
            azlks=azlks,
            rglks=rglks,
            kernel=kernel
        )

    elif matrix == "C2HV":

        return process_dual_pol(
            chunks,
            0, 3, 4, 8,
            scale12=1.0,
            scale22=1.0,
            azlks=azlks,
            rglks=rglks,
            kernel=kernel
        )
    elif matrix == "C2CP":

        # Compact-pol orientation and ellipticity angles.
        phi = kwargs.get("phi", 0.0)
        chi = kwargs.get("chi", 0.0)

        # Reconstruct C3.

        C = np.zeros(
            (3, 3) + np.asarray(chunks[0]).shape,
            dtype=np.complex64
        )

        C[0, 0] = chunks[0]

        C[0, 1] = (
            chunks[1] + 1j * chunks[2]
        )

        C[0, 2] = (
            chunks[3] + 1j * chunks[4]
        )

        C[1, 1] = chunks[5]

        C[1, 2] = (
            chunks[6] + 1j * chunks[7]
        )

        C[2, 2] = chunks[8]

        # Restore Hermitian symmetry.

        C[1, 0] = np.conj(C[0, 1])
        C[2, 0] = np.conj(C[0, 2])
        C[2, 1] = np.conj(C[1, 2])

        # Multilooking and filtering.

        C = process_C3_matrix(
            C,
            azlks,
            rglks,
            kernel
        )

        # C3 -> Compact-pol C2.

        Cp = C3_C2CP_mat(
            C,
            phi,
            chi
        )

        return matrix_to_components(Cp)

    else:

        raise ValueError(
            f"Unsupported C3 conversion: {matrix}"
        )
def process_chunk_c2(chunks, *args, **kwargs):

    window_size = args[-6]
    cf = args[-5]
    matrix = args[-4]
    recip = args[-3]
    azlks = args[-2]
    rglks = args[-1]

    kernel = get_filter_kernel(window_size)

    C11 = np.asarray(chunks[0], dtype=np.float32)

    C12 = (
        np.asarray(chunks[1])
        + 1j * np.asarray(chunks[2])
    )

    C22 = np.asarray(chunks[3], dtype=np.float32)

    # Multilooking

    C11 = apply_multilook(C11, azlks, rglks)
    C12 = apply_multilook(C12, azlks, rglks)
    C22 = apply_multilook(C22, azlks, rglks)

    if matrix == "C2":

        C11 = apply_filter(C11, kernel)
        C12 = apply_filter(C12, kernel)
        C22 = apply_filter(C22, kernel)

        return (
            C11.astype(np.float32),
            C12.real.astype(np.float32),
            C12.imag.astype(np.float32),
            C22.astype(np.float32)
        )

    elif matrix == "T2":

        # C2 -> T2

        T11 = 0.5 * (
            C11 + C22 + 2 * C12.real
        )

        T22 = 0.5 * (
            C11 + C22 - 2 * C12.real
        )

        T12 = (
            0.5 * (C11 - C22)
            - 1j * C12.imag
        )

        # Spatial filtering

        T11 = apply_filter(T11, kernel)
        T12 = apply_filter(T12, kernel)
        T22 = apply_filter(T22, kernel)

        return (
            T11.astype(np.float32),
            T12.real.astype(np.float32),
            T12.imag.astype(np.float32),
            T22.astype(np.float32)
        )

    else:

        raise ValueError(
            f"Unsupported C2 conversion: {matrix}"
        )
def reconstruct_C4(chunks):

    diagonal = [
        0, 7, 12, 15
    ]

    off_diagonal = {
        (0, 1): (1, 2),
        (0, 2): (3, 4),
        (0, 3): (5, 6),
        (1, 2): (8, 9),
        (1, 3): (10, 11),
        (2, 3): (13, 14)
    }

    shape = np.asarray(chunks[0]).shape

    C = np.zeros(
        (4, 4) + shape,
        dtype=np.complex64
    )

    for i, idx in enumerate(diagonal):

        C[i, i] = np.asarray(
            chunks[idx],
            dtype=np.float32
        )

    for (i, j), (real_idx, imag_idx) in off_diagonal.items():

        value = (
            np.asarray(chunks[real_idx], dtype=np.float32)
            + 1j * np.asarray(chunks[imag_idx], dtype=np.float32)
        )

        C[i, j] = value
        C[j, i] = np.conj(value)

    return C


def C4_C2_mat(C, matrix):

    if matrix == "C2HX":
        indices = [0, 1]

    elif matrix == "C2VX":
        indices = [3, 2]

    elif matrix == "C2HV":
        indices = [0, 3]

    elif matrix == "C2":
        indices = [0, 3]

    else:
        raise ValueError(
            f"Unsupported C4 dual-pol conversion: {matrix}"
        )

    return C[np.ix_(indices, indices)]


def C4_C3_mat(C):

    A = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 1]
        ],
        dtype=np.float32
    )

    A[1] /= np.sqrt(2.0)

    return np.einsum(
        "ij,jk...,lk->il...",
        A,
        C,
        A
    )

def matrix_to_components(M):

    n = M.shape[0]

    output = []

    for i in range(n):

        for j in range(i, n):

            if i == j:

                output.append(
                    np.asarray(
                        M[i, j].real,
                        dtype=np.float32
                    )
                )

            else:

                output.extend([
                    np.asarray(
                        M[i, j].real,
                        dtype=np.float32
                    ),

                    np.asarray(
                        M[i, j].imag,
                        dtype=np.float32
                    )
                ])

    return tuple(output)

def C4_T4_mat(C):

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
        "ij,jk...,lk->il...",
        U,
        C,
        U
    )
def process_C4_matrix(C, azlks, rglks, kernel):

    n = C.shape[0]

    out_shape = (
        n,
        n
    ) + apply_multilook(
        C[0, 0],
        azlks,
        rglks
    ).shape

    result = np.zeros(
        out_shape,
        dtype=np.complex64
    )

    for i in range(n):

        for j in range(i, n):

            value = apply_multilook(
                C[i, j],
                azlks,
                rglks
            )

            value = apply_filter(
                value,
                kernel
            )

            if i == j:
                value = value.real

            result[i, j] = value
            result[j, i] = np.conj(value)

    return result

def process_chunk_c4(chunks, *args, **kwargs):

    window_size = args[-6]
    cf = args[-5]
    matrix = args[-4]
    recip = args[-3]
    azlks = args[-2]
    rglks = args[-1]

    kernel = get_filter_kernel(window_size)

    # --------------------------------------------------
    # 1. Reconstruct C4
    # --------------------------------------------------

    C = reconstruct_C4(chunks)

    # --------------------------------------------------
    # 2. Multilooking and spatial filtering
    # --------------------------------------------------

    C = process_C4_matrix(
        C,
        azlks,
        rglks,
        kernel
    )

    # --------------------------------------------------
    # 3. Perform requested conversion
    # --------------------------------------------------

    if matrix == "T4":

        # C4 -> T4

        T = C4_T4_mat(C)

        return matrix_to_components(T)

    elif matrix == "C3":

        # C4 -> C3

        C3 = C4_C3_mat(C)

        return matrix_to_components(C3)

    elif matrix in ("C2", "C2HX", "C2VX", "C2HV"):

        # C4 -> C2

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
            f"Unsupported C4 conversion: {matrix}"
        )

def C3_C2CP_mat(C, phi, chi):

    """
    Convert C3 covariance matrix to compact-pol C2.

    Parameters
    ----------
    C : ndarray
        Complex Hermitian C3 matrix.
        Shape: (3, 3, rows, cols)

    phi : float
        Compact-pol orientation angle in radians.

    chi : float
        Compact-pol ellipticity angle in radians.

    Returns
    -------
    Cp : ndarray
        Compact-pol C2 covariance matrix.
        Shape: (2, 2, rows, cols)
    """

    # --------------------------------------------------
    # Trigonometric terms
    # --------------------------------------------------

    cos_term = (
        np.cos(2 * phi)
        * np.cos(2 * chi)
    )

    sin_phi = (
        np.sin(2 * phi)
        * np.cos(2 * chi)
    )

    sin_chi = np.sin(2 * chi)

    a = 1.0 + cos_term
    b = 1.0 - cos_term

    # --------------------------------------------------
    # Extract C3 elements
    # --------------------------------------------------

    C11 = C[0, 0]
    C12 = C[0, 1]
    C13 = C[0, 2]

    C22 = C[1, 1]
    C23 = C[1, 2]

    C33 = C[2, 2]

    # --------------------------------------------------
    # Cp11
    # --------------------------------------------------

    Cp11 = 0.5 * (
        a * C11
        + 0.5 * b * C22
        + (sin_phi / np.sqrt(2))
        * (C12 + np.conj(C12))
        + (1j * sin_chi / np.sqrt(2))
        * (C12 - np.conj(C12))
    )

    # --------------------------------------------------
    # Cp12
    # --------------------------------------------------

    Cp12 = 0.5 * (
        (a / np.sqrt(2)) * C12
        + (b / np.sqrt(2)) * C23
        + sin_phi * (C13 + 0.5 * C22)
        + 1j * sin_chi * (C13 - 0.5 * C22)
    )

    # --------------------------------------------------
    # Cp22
    # --------------------------------------------------

    Cp22 = 0.5 * (
        0.5 * a * C22
        + b * C33
        + (sin_phi / np.sqrt(2))
        * (C23 + np.conj(C23))
        + (1j * sin_chi / np.sqrt(2))
        * (C23 - np.conj(C23))
    )

    # --------------------------------------------------
    # Construct Hermitian compact-pol C2
    # --------------------------------------------------

    Cp = np.empty(
        (2, 2) + C.shape[2:],
        dtype=np.complex64
    )

    Cp[0, 0] = Cp11.real
    Cp[0, 1] = Cp12
    Cp[1, 0] = np.conj(Cp12)
    Cp[1, 1] = Cp22.real

    return Cp


@time_it
def convert_C(
    in_dir,
    mat="T3",
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

    """
    Convert covariance matrices (C2, C3, C4) into coherency (T2, T3, T4)
    or covariance (C2, C3, C2HX, C2VX, C2HV, C2CP) matrices.

    Supports GeoTIFF and PolSARpro-compatible binary output formats,
    multilooking and spatial filtering.

    Examples
    --------
    >>> # Convert C3 to T3 matrix
    >>> convert_C("/path/to/C3_data", mat="T3")

    >>> # Convert C4 to T4 matrix
    >>> convert_C("/path/to/C4_data", mat="T4")

    >>> # Convert C4 to C3 matrix
    >>> convert_C("/path/to/C4_data", mat="C3")

    >>> # Extract dual-pol C2HV from C4
    >>> convert_C("/path/to/C4_data", mat="C2HV")

    >>> # Convert C3 to compact-pol C2
    >>> convert_C(
    ...     "/path/to/C3_data",
    ...     mat="C2CP",
    ...     phi=0.0,
    ...     chi=0.0
    ... )

    >>> # Apply multilooking and spatial filtering
    >>> convert_C(
    ...     "/path/to/C3_data",
    ...     mat="T3",
    ...     azlks=4,
    ...     rglks=4,
    ...     win=5
    ... )

    >>> # Generate Cloud Optimized GeoTIFF output
    >>> convert_C(
    ...     "/path/to/C4_data",
    ...     mat="T4",
    ...     fmt="tif",
    ...     cog=True,
    ...     comp=True
    ... )

    Parameters
    ----------
    in_dir : str
        Path to the input directory containing covariance matrix
        components. The input matrix type (C2, C3, or C4) is
        automatically detected based on the available components.

        Supported input components:

        C2:
            C11, C12_real, C12_imag, C22

        C3:
            C11, C12_real, C12_imag,
            C13_real, C13_imag,
            C22, C23_real, C23_imag, C33

        C4:
            C11, C12_real, C12_imag,
            C13_real, C13_imag,
            C14_real, C14_imag,
            C22, C23_real, C23_imag,
            C24_real, C24_imag,
            C33, C34_real, C34_imag, C44

        All input components must have the same file format
        (.bin or .tif), dimensions, and georeferencing.

    mat : str, default='T3'
        Desired output matrix.

        Supported conversions:

        C2 input:
            - 'C2'
            - 'T2'

        C3 input:
            - 'T3'
            - 'C2'
            - 'C2HX'
            - 'C2VX'
            - 'C2HV'
            - 'C2CP'

        C4 input:
            - 'T4'
            - 'C3'
            - 'C2'
            - 'C2HX'
            - 'C2VX'
            - 'C2HV'

    azlks : int, default=1
        Number of looks in the azimuth direction.
        Must be a positive integer.

    rglks : int, default=1
        Number of looks in the range direction.
        Must be a positive integer.

    win : int, default=1
        Size of the square moving-average spatial filtering window.
        A value of 1 disables spatial filtering.

    fmt : {'tif', 'bin'}, default='tif'
        Output file format:
        - 'tif': GeoTIFF
        - 'bin': PolSARpro-compatible binary format.

    cog : bool, default=False
        If True, generates Cloud Optimized GeoTIFF output.
        Requires fmt='tif'.

    ovr : list[int], default=[2, 4, 8, 16]
        Overview pyramid levels used during COG generation.

    comp : bool, default=False
        If True, applies LZW compression to GeoTIFF output files.

    recip : bool, default=True
        If True, applies reciprocal symmetry to the scattering
        matrix. Retained for compatibility with the processing
        interface.

    cf : float, default=1
        Calibration factor (linear) retained for compatibility
        with the processing interface.

    out_dir : str or None, default=None
        Path to the output directory. If None, creates an output
        directory named after the requested matrix inside in_dir.

    max_workers : int or None, default=None
        Maximum number of parallel worker processes.
        If None, uses the default worker count provided by
        process_chunks_parallel.

    block_size : tuple[int, int], default=(512, 512)
        Processing block size in pixels, specified as
        (range, azimuth). Block dimensions are automatically
        adjusted to align with the multilooking factors.

    phi : float, default=0.0
        Compact-pol orientation angle in radians.
        Used only for C3-to-C2CP conversion.

    chi : float, default=0.0
        Compact-pol ellipticity angle in radians.
        Used only for C3-to-C2CP conversion.

    progress_callback : callable or None, default=None
        Optional callback function for monitoring processing
        progress, primarily intended for QGIS plugin integration.
        Receives a float between 0.0 and 1.0 representing
        the fraction of completed processing blocks.

    Returns
    -------
    None
        Writes the converted matrix components to the output
        directory. Each matrix component is saved as an individual
        raster file.

    Raises
    ------
    ValueError
        If input parameters are invalid or input raster properties
        are inconsistent.

    FileNotFoundError
        If the required covariance matrix components are missing
        or an input raster cannot be opened.

    NotImplementedError
        If the requested input-to-output matrix conversion
        is not supported.
    """ 




    # --------------------------------------------------
    # 1. Validate parameters
    # --------------------------------------------------

    if not isinstance(azlks, (int, np.integer)) or azlks < 1:
        raise ValueError("azlks must be a positive integer.")

    if not isinstance(rglks, (int, np.integer)) or rglks < 1:
        raise ValueError("rglks must be a positive integer.")

    if not isinstance(win, (int, np.integer)) or win < 1:
        raise ValueError("win must be a positive integer.")

    if not isinstance(block_size, (tuple, list)) or len(block_size) != 2:
        raise ValueError("block_size must contain two integers.")

    if any(
        not isinstance(v, (int, np.integer)) or v <= 0
        for v in block_size
    ):
        raise ValueError("block_size values must be positive integers.")

    if cog and fmt != "tif":
        raise ValueError("COG output requires fmt='tif'.")

    # --------------------------------------------------
    # 2. Detect input matrix
    # --------------------------------------------------

    input_filepaths = get_c_input_filepaths(in_dir)

    in_mat = {
        4: "C2",
        9: "C3",
        16: "C4"
    }[len(input_filepaths)]

    print(
        f"Detected {in_mat} matrix and converting to {mat} matrix"
    )

    # --------------------------------------------------
    # 3. Validate conversion
    # --------------------------------------------------

    if mat not in SUPPORTED_CONVERSIONS.get(in_mat, set()):

        raise NotImplementedError(
            f"Conversion {in_mat} -> {mat} is not supported yet.\n"
            f"Supported conversions: "
            f"{SUPPORTED_CONVERSIONS.get(in_mat, set())}"
        )

    # --------------------------------------------------
    # 4. Validate input raster properties
    # --------------------------------------------------

    datasets = []

    try:

        for filepath in input_filepaths:

            ds = gdal.Open(
                filepath,
                gdal.GA_ReadOnly
            )

            if ds is None:
                raise FileNotFoundError(
                    f"Cannot open input file: {filepath}"
                )

            datasets.append(ds)

        ref_ds = datasets[0]

        in_cols = ref_ds.RasterXSize
        in_rows = ref_ds.RasterYSize

        in_geotransform = ref_ds.GetGeoTransform()
        in_projection = ref_ds.GetProjection()

        if in_cols < rglks or in_rows < azlks:
            raise ValueError(
                "Multilooking factors exceed input raster dimensions."
            )

        # Check all component dimensions and georeferencing.

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

    # --------------------------------------------------
    # 5. Calculate output dimensions
    # --------------------------------------------------

    out_x_size = in_cols // rglks
    out_y_size = in_rows // azlks

    if out_x_size <= 0 or out_y_size <= 0:
        raise ValueError("Invalid output raster dimensions.")

    # --------------------------------------------------
    # 6. Calculate output geotransform
    # --------------------------------------------------

    out_geotransform = list(in_geotransform)

    out_geotransform[1] *= rglks
    out_geotransform[2] *= rglks

    out_geotransform[4] *= azlks
    out_geotransform[5] *= azlks

    out_geotransform = tuple(out_geotransform)

    # --------------------------------------------------
    # 7. Calculate processing block size
    # --------------------------------------------------

    def align_block(value, factor):

        return max(
            factor,
            math.ceil(value / factor) * factor
        )

    block_x_size = align_block(
        block_size[0],
        rglks
    )

    block_y_size = align_block(
        block_size[1],
        azlks
    )

    block_size = (
        block_x_size,
        block_y_size
    )

    # --------------------------------------------------
    # 8. Generate output filepaths
    # --------------------------------------------------

    output_filepaths = get_output_filepaths(
        in_dir,
        out_dir,
        mat,
        fmt
    )

    # --------------------------------------------------
    # 9. Select processing function
    # --------------------------------------------------

    if in_mat == "C2":

        process_func = process_chunk_c2

    elif in_mat == "C3":

        process_func = process_chunk_c3

    else:

        process_func = process_chunk_c4

    # --------------------------------------------------
    # 10. Parallel processing
    # --------------------------------------------------

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
        phi = phi,
        chi = chi,
        progress_callback=progress_callback
    )
