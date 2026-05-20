"""Estimate Nuitka onefile footprint: static import closure from main.py."""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

SKIP_PREFIXES = (
    "test",
    "pytest",
    "unittest",
    "pdb",
    "distutils",
)


def pkg_name(module: str) -> str:
    return module.split(".")[0]


def find_py_module(module: str) -> Path | None:
    if module == "main":
        p = ROOT / "main.py"
        return p if p.exists() else None
    rel = Path(*module.split("."))
    for base in (ROOT / "src", ROOT / "cua_mcp", ROOT):
        for candidate in (
            base / f"{rel}.py",
            base / rel / "__init__.py",
        ):
            if candidate.is_file():
                return candidate
    spec = importlib.util.find_spec(module)
    if spec and spec.origin and spec.origin.endswith(".py"):
        return Path(spec.origin)
    return None


def imports_in(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                out.add(node.module)
            for alias in node.names:
                if alias.name == "*":
                    continue
    return out


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def package_size(top: str) -> tuple[int, Path | None]:
    try:
        spec = importlib.util.find_spec(top)
    except (ImportError, ModuleNotFoundError, ValueError):
        return 0, None
    if not spec or not spec.origin:
        return 0, None
    origin = Path(spec.origin)
    if origin.name == "__init__.py":
        base = origin.parent
    else:
        base = origin.parent
    # stdlib / single-file modules
    if str(base).startswith(str(Path(sys.base_prefix))):
        lib = Path(sys.base_prefix) / "Lib"
        if base.is_relative_to(lib):
            return dir_size(base), base
        return origin.stat().st_size, origin
    return dir_size(base), base


def main() -> None:
    queue = ["main"]
    seen_modules: set[str] = set()
    project_modules: set[str] = set()
    third_party_tops: set[str] = set()

    while queue:
        mod = queue.pop()
        if mod in seen_modules:
            continue
        seen_modules.add(mod)
        top = pkg_name(mod)
        if top.startswith(SKIP_PREFIXES):
            continue

        py = find_py_module(mod)
        if py and py.is_relative_to(ROOT):
            project_modules.add(mod)
            try:
                names = imports_in(py)
            except SyntaxError:
                continue
            for name in sorted(names):
                if name.startswith(SKIP_PREFIXES):
                    continue
                if find_py_module(name) or find_py_module(pkg_name(name)):
                    if name not in seen_modules:
                        queue.append(name)
                else:
                    third_party_tops.add(pkg_name(name))
        else:
            third_party_tops.add(top)

    data_dirs = [
        ROOT / "cua_mcp" / "read_screen_text",
        ROOT / "cua_mcp" / "best.onnx",
    ]
    data_bytes = sum(dir_size(d) for d in data_dirs)

    pkg_bytes: dict[str, int] = {}
    for top in sorted(third_party_tops):
        sz, _ = package_size(top)
        if sz:
            pkg_bytes[top] = sz

    # Likely pulled by nuitka include flags even if not imported
    for extra in ("opencc",):
        if extra not in pkg_bytes:
            sz, _ = package_size(extra)
            if sz:
                pkg_bytes[extra] = sz

    total_pkgs = sum(pkg_bytes.values())
    print("=== Static import estimate (main.py closure) ===")
    print(f"Project modules traced: {len(project_modules)}")
    print(f"Third-party top-level packages: {len(pkg_bytes)}")
    print(f"Included data dirs: {data_bytes / (1024**2):.1f} MB")
    print(f"Third-party on disk (upper bound, no dedup): {total_pkgs / (1024**2):.1f} MB")
    print()
    print("Top packages by size:")
    for name, sz in sorted(pkg_bytes.items(), key=lambda x: -x[1])[:25]:
        print(f"  {name:20} {sz / (1024**2):7.1f} MB")
    print()
    rough = total_pkgs + data_bytes
    # Nuitka onefile: CPython runtime + compression overhead
    print(f"Rough uncompressed payload: {rough / (1024**2):.0f} MB")
    print(f"Estimated onefile .exe (compressed): {rough * 0.55 / (1024**2):.0f}–{rough * 0.75 / (1024**2):.0f} MB")
    print(f"Estimated standalone folder (--standalone, no onefile): {rough * 1.05 / (1024**2):.0f}–{rough * 1.2 / (1024**2):.0f} MB")
    print()
    not_in_closure = []
    for maybe in ("torch", "ultralytics", "skimage", "scipy", "pytest"):
        if maybe not in pkg_bytes:
            not_in_closure.append(maybe)
    if not_in_closure:
        print("NOT in static closure (likely excluded unless follow-imports grabs more):")
        print(" ", ", ".join(not_in_closure))


if __name__ == "__main__":
    main()
