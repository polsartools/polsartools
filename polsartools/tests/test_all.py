# polsartools/tests/test_all.py
from polsartools.tests.test_utils import test_utils_processing
from polsartools.tests.test_filters import test_filters_processing
from tqdm import tqdm


def test_all(verbose=False, silent=True):
    steps = [
        ("Utils Test", test_utils_processing),
        ("Filters Test", test_filters_processing),
    ]

    results = []

    for label, func in tqdm(steps, desc="Running Tests", unit="step", disable=False):  # Always show progress bar
        if verbose:
            print(f"\n {label}...")

        try:
            func(verbose=verbose, silent=silent)
            results.append((label, " Passed"))
        except Exception as e:
            results.append((label, f" Failed: {str(e)}"))

    print("\n Test Summary:")
    for label, status in results:
        print(f"• {label}: {status}")

    if all(" Passed" in status for _, status in results):
        print("\n All tests passed.")
    else:
        print("\n Some tests failed !!! Please check the details above.")
