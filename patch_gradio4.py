"""Patch Gradio 4.44.1 for Python 3.12+ and huggingface_hub 0.25+ compatibility.

Gradio 4.44.1 has two known issues when used with Python 3.12+ and newer huggingface_hub:
1. `gradio/oauth.py` imports `HfFolder` which was removed in huggingface_hub>=0.25
2. `gradio_client/utils.py` `get_type()` crashes on non-dict JSON schemas from newer Pydantic

Run this script after `pip install -r requirements.txt` to apply both patches.
"""

import os
import sys
import importlib


def patch_oauth():
    """Patch gradio/oauth.py to handle missing HfFolder gracefully."""
    import gradio.oauth as oauth_mod
    oauth_path = os.path.abspath(oauth_mod.__file__)

    with open(oauth_path, "r") as f:
        content = f.read()

    # Already patched?
    if "except ImportError:" in content and "HfFolder = None" in content:
        print(f"[OK] {oauth_path} already patched (HfFolder)")
        return

    # Patch import
    old = 'from huggingface_hub import HfFolder, whoami'
    new = '''try:
    from huggingface_hub import HfFolder
except ImportError:
    HfFolder = None
from huggingface_hub import whoami'''
    if old in content:
        content = content.replace(old, new)
    else:
        print(f"[SKIP] {oauth_path}: HfFolder import not found (already changed or different version)")
        return

    # Patch usage
    old_usage = 'token = HfFolder.get_token()'
    new_usage = '''if HfFolder is not None:
        token = HfFolder.get_token()
    else:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")'''
    if old_usage in content:
        content = content.replace(old_usage, new_usage)

    with open(oauth_path, "w") as f:
        f.write(content)
    print(f"[PATCHED] {oauth_path} (HfFolder compatibility)")


def patch_gradio_client():
    """Patch gradio_client/utils.py to handle non-dict JSON schemas (Pydantic + Python 3.12+)."""
    import gradio_client.utils as utils_mod
    utils_path = os.path.abspath(utils_mod.__file__)

    with open(utils_path, "r") as f:
        content = f.read()

    patched = 0

    # Patch 1: get_type()
    old_get_type = 'def get_type(schema: dict):\n    if "const" in schema:'
    new_get_type = 'def get_type(schema: dict):\n    if not isinstance(schema, dict):\n        return type(schema).__name__.lower()\n    if "const" in schema:'
    if old_get_type in content:
        content = content.replace(old_get_type, new_get_type)
        patched += 1
    elif 'if not isinstance(schema, dict):' in content:
        print(f"[OK] {utils_path} already patched (get_type)")
    else:
        print(f"[SKIP] {utils_path}: get_type pattern not found")

    # Patch 2: _json_schema_to_python_type()
    old_js = 'def _json_schema_to_python_type(schema: Any, defs) -> str:\n    """Convert the json schema into a python type hint"""\n    if schema == {}:'
    new_js = 'def _json_schema_to_python_type(schema: Any, defs) -> str:\n    """Convert the json schema into a python type hint"""\n    if not isinstance(schema, dict):\n        return type(schema).__name__\n    if schema == {}:'
    if old_js in content:
        content = content.replace(old_js, new_js)
        patched += 1
    elif 'if not isinstance(schema, dict):' in content:
        print(f"[OK] {utils_path} already patched (_json_schema_to_python_type)")
    else:
        print(f"[SKIP] {utils_path}: _json_schema_to_python_type pattern not found")

    if patched > 0:
        with open(utils_path, "w") as f:
            f.write(content)
        print(f"[PATCHED] {utils_path} ({patched} patch(es) applied)")


def main():
    print("Patching Gradio 4.44.1 for Python 3.12+ / huggingface_hub 0.25+ compatibility...")
    print(f"Python: {sys.version}")
    patch_oauth()
    patch_gradio_client()
    # Reload patched modules
    if 'gradio.oauth' in sys.modules:
        importlib.reload(sys.modules['gradio.oauth'])
    if 'gradio_client.utils' in sys.modules:
        importlib.reload(sys.modules['gradio_client.utils'])
    print("Done!")


if __name__ == "__main__":
    main()
