from polsartools.tests.synthetic_data import gen_S2, gen_T3
import os, shutil, logging
from polsartools.preprocess.filters import filter_refined_lee

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_filters_processing(verbose=False, silent=False):
    if not verbose:
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.INFO)

    T3_folder = './tests/test_data/T3'
    os.makedirs(T3_folder, exist_ok=True)

    s11, s12, s22 = gen_S2('./tests/test_data/S2')
    gen_T3(T3_folder, s11, s12, s22)

    window_size = 5
    logger.info("Applying Refined Lee filter...")

    if silent:
        from contextlib import redirect_stdout, redirect_stderr
        with open(os.devnull, 'w') as f, redirect_stdout(f), redirect_stderr(f):
            filter_refined_lee(T3_folder, win=window_size, sub_dir=False)
    else:
        filter_refined_lee(T3_folder, win=window_size, sub_dir=False)

    outFolder = os.path.join(os.path.dirname(T3_folder) + f"_rlee_{window_size}x{window_size}", os.path.basename(T3_folder))
    output_files = [os.path.join(outFolder, f"T{i}{i}.tif") for i in range(1, 4)]

    for file_path in output_files:
        assert os.path.exists(file_path)
        assert os.path.getsize(file_path) > 0
        logger.info(f"Verified {file_path}")
        os.remove(file_path)

    shutil.rmtree(os.path.dirname(outFolder))
    shutil.rmtree('./tests/test_data')
    logger.info(f"Cleaned up {outFolder}")
