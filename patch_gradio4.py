"""Patch Gradio 4.44.1 for Python 3.12+/3.13+ and newer dependency compatibility.

Gradio 4.44.1 has several known issues when used with Python 3.12+/3.13+ and newer deps:
1. `gradio/oauth.py` imports `HfFolder` which was removed in huggingface_hub>=0.25
2. `gradio_client/utils.py` `get_type()` crashes on non-dict JSON schemas from newer Pydantic
3. `jinja2` 3.1.5+/3.2.x `_load_template` creates unhashable cache_key (dict) on some versions
4. `gradio/blocks.py` `get_api_info()` doesn't guard `info` being non-dict from Pydantic

Run this script after `pip install -r requirements.txt` to apply all patches.
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
    try:
        import importlib.metadata
        dist = importlib.metadata.distribution(package_name)
        located = dist.locate_file(relative_path)
        if located and located.is_file():
            return str(located)
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

    # Already patched or import doesn't exist (newer huggingface_hub removed it entirely)
    if "HfFolder = None" in content or "HfFolder" not in content:
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
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")"""
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

    # Patch 3: json_schema_to_python_type() - wrap in try/except for safety
    old_jst = 'def json_schema_to_python_type(schema: Any) -> str:\n    type_ = _json_schema_to_python_type(schema, schema.get("$defs"))\n    return type_.replace(CURRENT_FILE_DATA_FORMAT, "filepath")'
    new_jst = 'def json_schema_to_python_type(schema: Any) -> str:\n    try:\n        if not isinstance(schema, dict):\n            return type(schema).__name__\n        type_ = _json_schema_to_python_type(schema, schema.get("$defs"))\n        return type_.replace(CURRENT_FILE_DATA_FORMAT, "filepath")\n    except Exception:\n        return "Any"'
    if old_jst in content:
        content = content.replace(old_jst, new_jst)
        patched += 1
    elif "except Exception:\n        return \"Any\"" in content:
        print(f"[OK] {utils_path} already patched (json_schema_to_python_type try/except)")
    else:
        print(f"[SKIP] {utils_path}: json_schema_to_python_type pattern not found")

    if patched > 0:
        with open(utils_path, "w") as f:
            f.write(content)
        print(f"[PATCHED] {utils_path} ({patched} patch(es) applied)")


def patch_jinja2():
    """Patch jinja2 to handle unhashable cache keys in _load_template.

    Some jinja2 versions (3.1.5+, 3.2.x) create a cache_key that includes a
    globals dict, which is unhashable. This causes TypeError when Gradio
    tries to render the main page template.
    """
    env_path = find_package_file("jinja2", "environment.py")
    if not env_path:
        print("[SKIP] jinja2/environment.py not found")
        return

    with open(env_path, "r") as f:
        content = f.read()

    # Check if already patched
    if "_gradio_hashable_key" in content:
        print(f"[OK] {env_path} already patched (cache_key)")
        return

    helper_code = '''

def _gradio_hashable_key(key):
    """Make a cache key hashable for jinja2 template cache compatibility."""
    try:
        hash(key)
        return key
    except TypeError:
        if isinstance(key, dict):
            return tuple(sorted((str(k), _gradio_hashable_key(v)) for k, v in key.items()))
        if isinstance(key, (list, tuple)):
            return tuple(_gradio_hashable_key(v) for v in key)
        return str(key)

'''

    class_marker = "class Environment"
    if class_marker not in content:
        print(f"[SKIP] {env_path}: 'class Environment' not found")
        return

    content = content.replace(class_marker, helper_code + class_marker)

    lines = content.split('\n')
    in_load_template = False
    patch_count = 0
    result = []

    for line in lines:
        if 'def _load_template(' in line:
            in_load_template = True
        elif in_load_template and line.strip().startswith('def '):
            in_load_template = False

        if in_load_template and 'cache_key =' in line and '_gradio_hashable_key' not in line:
            indent = len(line) - len(line.lstrip())
            stripped = line.strip()
            lhs, rhs = stripped.split('=', 1)
            patched_line = ' ' * indent + lhs + ' = _gradio_hashable_key(' + rhs + ')'
            result.append(patched_line)
            patch_count += 1
        else:
            result.append(line)

    if patch_count == 0:
        print(f"[SKIP] {env_path}: no cache_key assignment found in _load_template")
        content = content.replace(helper_code, '')
    else:
        content = '\n'.join(result)
        print(f"[PATCHED] {env_path} (cache_key hashability, {patch_count} line(s))")

    with open(env_path, "w") as f:
        f.write(content)


def patch_blocks():
    """Patch gradio/blocks.py get_api_info to guard info being non-dict.

    Newer Pydantic on Python 3.13 may produce component api_info values
    that are not dicts, causing .get() and .split() calls to crash.
    """
    blocks_path = find_package_file("gradio", "blocks.py")
    if not blocks_path:
        print("[SKIP] gradio/blocks.py not found")
        return

    with open(blocks_path, "r") as f:
        content = f.read()

    if "_gradio_safe_info" in content:
        print(f"[OK] {blocks_path} already patched (get_api_info)")
        return

    patched = 0

    # Guard 1: line with info.get("description", "") for outputs
    # Original: info.get("description", "")
    # We need to ensure info is a dict. Find the specific pattern.
    old_out_desc = 'info.get("description", "")'
    new_out_desc = '(info.get("description", "") if isinstance(info, dict) else "")'
    if old_out_desc in content:
        content = content.replace(old_out_desc, new_out_desc)
        patched += 1

    # Guard 2: line with (info or {}).get("additional_description", "")
    # This is already guarded with (info or {}), so it's safe.

    # Guard 3: Ensure json_schema_to_python_type gets a dict
    # We look for: python_type = client_utils.json_schema_to_python_type(info)
    # and wrap info to be a dict if it isn't
    old_pt = 'python_type = client_utils.json_schema_to_python_type(info)'
    new_pt = 'python_type = client_utils.json_schema_to_python_type(info if isinstance(info, dict) else {})'
    if old_pt in content:
        content = content.replace(old_pt, new_pt)
        patched += 1

    if patched > 0:
        with open(blocks_path, "w") as f:
            f.write(content)
        print(f"[PATCHED] {blocks_path} (get_api_info non-dict guard, {patched} line(s))")
    else:
        print(f"[SKIP] {blocks_path}: patterns not found")


def main():
    print("Patching Gradio 4.44.1 for Python 3.12+/3.13+ compatibility...")
    print(f"Python: {sys.version}")
    patch_oauth()
    patch_gradio_client()
    patch_jinja2()
    patch_blocks()
    print("Done!")


if __name__ == "__main__":
    main()
