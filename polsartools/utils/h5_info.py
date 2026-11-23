import warnings
warnings.filterwarnings("ignore")
import tables 

def h5_info(file_path, show_unimplemented=True):
    """
    Displays the hierarchical structure and metadata of an HDF5 file.

    Recursively traverses the HDF5 file and prints information about its
    groups and datasets, including dataset paths, shapes, and data types.

    The function also identifies unimplemented or unsupported HDF5
    objects and optionally displays them in the output.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to the input HDF5 file.

    show_unimplemented : bool, optional
        If True (default), displays unimplemented or unsupported HDF5
        objects along with their paths and shapes. If False, these
        objects are skipped.

    Returns
    -------
    None
        Prints the hierarchical structure and metadata of the HDF5
        file to the console. Does not return any values.

    Examples
    --------
    >>> import polsartools as pst

    >>> pst.h5_info("path/to/file.h5")

    >>> pst.h5_info("path/to/file.h5", show_unimplemented=False)
    """

    with tables.open_file(file_path, mode="r") as f:
        def traverse(node):
            if isinstance(node, tables.Group):
                print(f"Group:          {node._v_pathname}")
                for child_name in node._v_children:
                    child_path = (
                        f"/{child_name}"
                        if node._v_pathname == "/"
                        else f"{node._v_pathname.rstrip('/')}/{child_name}"
                    )
                    try:
                        child_node = f.get_node(child_path)
                        traverse(child_node)
                    except Exception as e:
                        print(f"Skipping bad/unsupported node: {child_path} (Error: {e})")
            elif isinstance(node, tables.UnImplemented):
                if show_unimplemented:
                    try:
                        shape = node.shape
                    except Exception:
                        shape = "N/A"
                    print(f"UnImplemented:  {node._v_pathname} (Shape: {shape})")
            else:
                try:
                    shape = node.shape
                except Exception:
                    shape = "N/A"
                try:
                    dtype = node.dtype
                except Exception:
                    dtype = "N/A"
                print(f"Dataset:        {node._v_pathname} (Shape: {shape}, Dtype: {dtype})")

        traverse(f.root)