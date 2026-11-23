import warnings
warnings.filterwarnings("ignore")
import tables 

def h5_info(file_path, show_unimplemented=True):
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