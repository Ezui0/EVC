"""Patch Gradio 4.44.1 for Python 3.12+/3.13+ and huggingface_hub 0.25+ compatibility.

Gradio 4.44.1 has two known issues when used with Python 3.12+ and newer huggingface_hub:
1. `gradio/oauth.py` imports `HfFolder` which was removed in huggingface_hub>=0.25
2. `gradio_client/utils.py` `get_type()` crashes on non-dict JSON schemas from newer Pydantic

Run this script after `pip install -r requirements.txt` to apply both patches.
This script finds files by path without importing gradio, so it works even if gradio
currently fails to import.
"""

import os
import sys


def find_package_file(package_name, relative_path):
    """Find a package file by walking sys.path, no import needed."""
    for base in sys.path:
        if not base:
            continue
        candidate = os.path.join(base, package_name.replace(".", os.sep), relative_path)
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
        # Also check .dist-info / direct install paths
    # Fallback: use importlib.metadata (available in 3.8+)
    try:
        import importlib.metadata
        dist = importlib.metadata.distribution(package_name)
        if dist.locate_file(relative_path).is_file():
            return str(dist.locate_file(relative_path))
    except Exception:
        pass
    return None


def patch_oauth():
    """Patch gradio/oauth.py to handle missing HfFolder gracefully."""
    oauth_path = find_package_file("gradio", "oauth.py")
    if not oauth_path:
        print("[SKIP] gradio/oauth.py not found")
        return

    with open(oauth_path, "r") as f:
        content = f.read()

    if "HfFolder = None" in content:
        print(f"[OK] {oauth_path} already patched (HfFolder)")
        return

    old = "from huggingface_hub import HfFolder, whoami"
    new = """try:
    from huggingface_hub import HfFolder
except ImportError:
    HfFolder = None
from huggingface_hub import whoami"""
    if old not in content:
        print(f"[SKIP] {oauth_path}: HfFolder import not found")
        return

    content = content.replace(old, new)

    old_usage = "token = HfFolder.get_token()"
    new_usage = """if HfFolder is not None:
        token = HfFolder.get_token()
    else:
        token = os.environ.get(\"HF_TOKEN\") or os.environ.get(\"HUGGING_FACE_HUB_TOKEN\")"""
    if old_usage in content:
        content = content.replace(old_usage, new_usage)

    with open(oauth_path, "w") as f:
        f.write(content)
    print(f"[PATCHED] {oauth_path} (HfFolder compatibility)")


def patch_gradio_client():
    """Patch gradio_client/utils.py to handle non-dict JSON schemas (Pydantic + Python 3.12+)."""
    utils_path = find_package_file("gradio_client", "utils.py")
    if not utils_path:
        print("[SKIP] gradio_client/utils.py not found")
        return

    with open(utils_path, "r") as f:
        content = f.read()

    patched = 0

    # Patch 1: get_type()
    old_gt = 'def get_type(schema: dict):\n    if "const" in schema:'
    new_gt = 'def get_type(schema: dict):\n    if not isinstance(schema, dict):\n        return type(schema).__name__.lower()\n    if "const" in schema:'
    if old_gt in content:
        content = content.replace(old_gt, new_gt)
        patched += 1
    elif "if not isinstance(schema, dict):" in content:
        print(f"[OK] {utils_path} already patched (get_type)")
    else:
        print(f"[SKIP] {utils_path}: get_type pattern not found")

    # Patch 2: _json_schema_to_python_type()
    old_js = 'def _json_schema_to_python_type(schema: Any, defs) -> str:\n    """Convert the json schema into a python type hint"""\n    if schema == {}:'
    new_js = 'def _json_schema_to_python_type(schema: Any, defs) -> str:\n    """Convert the json schema into a python type hint"""\n    if not isinstance(schema, dict):\n        return type(schema).__name__\n    if schema == {}:'
    if old_js in content:
        content = content.replace(old_js, new_js)
        patched += 1
    elif "if not isinstance(schema, dict):" in content:
        print(f"[OK] {utils_path} already patched (_json_schema_to_python_type)")
    else:
        print(f"[SKIP] {utils_path}: _json_schema_to_python_type pattern not found")

    if patched > 0:
        with open(utils_path, "w") as f:
            f.write(content)
        print(f"[PATCHED] {utils_path} ({patched} patch(es) applied)")


def main():
    print("Patching Gradio 4.44.1 for Python 3.12+/3.13+ / huggingface_hub 0.25+ compatibility...")
    print(f"Python: {sys.version}")
    patch_oauth()
    patch_gradio_client()
    print("Done!")


if __name__ == "__main__":
    main()
