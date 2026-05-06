try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:
    import importlib.util
    from pathlib import Path

    nodes_path = Path(__file__).resolve().with_name("nodes.py")
    spec = importlib.util.spec_from_file_location("seedvr2_canonical_nodes", nodes_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load nodes module from {nodes_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    NODE_CLASS_MAPPINGS = module.NODE_CLASS_MAPPINGS
    NODE_DISPLAY_NAME_MAPPINGS = module.NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
