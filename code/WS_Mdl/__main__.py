"""CLI dispatcher for the WS_Mdl package.

This module enables commands like:
    WS_Mdl <function_name> [arguments ...]

How it works:
1) Scans the package tree for Python files.
2) Reads each file's ``__all__`` definition (without importing modules).
3) Builds a map: exported_name -> [module(s)].
4) Imports candidate module(s) lazily and calls the requested function.

Why this design:
- Keeps startup lightweight.
- Avoids importing heavy modules unless required.
- Gives users a consistent way to call explicitly exported functions.
"""

import ast
import importlib
import sys
from pathlib import Path

# Package metadata used by the discovery logic.
PACKAGE_NAME = 'WS_Mdl'
PACKAGE_ROOT = Path(__file__).resolve().parent
# Skip directories that should not participate in CLI export discovery.
SKIP_TOP_LEVEL_DIRS = {'scripts', '__pycache__'}


def _module_name_from_path(Pa: Path) -> str:
    """Convert a file path inside ``PACKAGE_ROOT`` to a dotted module path.

    Examples:
    - ``WS_Mdl/core/style.py`` -> ``WS_Mdl.core.style``
    - ``WS_Mdl/core/__init__.py`` -> ``WS_Mdl.core``
    - ``WS_Mdl/__init__.py`` -> ``WS_Mdl``
    """

    Pa_Rel = Pa.relative_to(PACKAGE_ROOT)

    # ``__init__.py`` represents the package itself, so drop the filename.
    if Pa.name == '__init__.py':
        parts = Pa_Rel.parts[:-1]
    else:
        # Normal module file: remove the .py suffix and keep all path parts.
        parts = Pa_Rel.with_suffix('').parts

    return PACKAGE_NAME if not parts else f'{PACKAGE_NAME}.{".".join(parts)}'


def _extract_all_names(py_path: Path) -> list[str]:
    """Extract literal string names from a module's ``__all__`` assignment.

    Parsing is done with ``ast`` so we don't execute module code during discovery.

    Returns an empty list if:
    - the file can't be read/parsed,
    - ``__all__`` is missing,
    - or ``__all__`` is dynamic/non-literal (for safety and predictability).
    """

    try:
        source = py_path.read_text(encoding='utf-8')
        tree = ast.parse(source, filename=str(py_path))
    except Exception:
        # Treat unreadable or syntactically invalid files as non-exporting.
        return []

    # Only inspect top-level statements; nested assignments are not relevant.
    for node in tree.body:
        # ``__all__ = [...]``
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        # ``__all__: list[str] = [...]``
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        else:
            continue

        if value is None:
            continue
        if not any(isinstance(target, ast.Name) and target.id == '__all__' for target in targets):
            continue

        # Accept only static containers of string literals.
        if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            names: list[str] = []
            for item in value.elts:
                if isinstance(item, ast.Constant) and isinstance(item.value, str):
                    names.append(item.value)
                else:
                    # Non-literal entries are ignored to avoid executing code.
                    return []
            return names

    return []


def _discover_exports() -> dict[str, list[str]]:
    """Build mapping from exported symbol name to module path(s).

    Returns:
        dict[str, list[str]]
            Example:
            {
                "set_verbose": ["WS_Mdl.core.style"],
                "sprint": ["WS_Mdl.core.style"]
            }
    """

    exports: dict[str, list[str]] = {}

    # Discover all package files in a deterministic order.
    for py_path in sorted(PACKAGE_ROOT.rglob('*.py')):
        rel = py_path.relative_to(PACKAGE_ROOT)

        # Skip CLI module itself.
        if py_path.name == '__main__.py':
            continue

        # Skip top-level package ``__init__`` to avoid exposing package names
        # like ``core``/``io`` as callable CLI commands.
        if py_path == PACKAGE_ROOT / '__init__.py':
            continue

        # Skip explicit directories that are not part of command discovery.
        if rel.parts and rel.parts[0] in SKIP_TOP_LEVEL_DIRS:
            continue

        names = _extract_all_names(py_path)
        if not names:
            continue

        module_name = _module_name_from_path(py_path)
        for name in names:
            # Keep list values so collisions are visible/handled later.
            exports.setdefault(name, []).append(module_name)

    return exports


def _print_usage() -> None:
    """Print CLI usage help."""
    print('Usage: WS_Mdl <function_name> [arguments ...]')
    print('       WS_Mdl --list')


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint.

    Args:
        argv: Optional list of arguments excluding executable name.
              If omitted, values are taken from ``sys.argv[1:]``.

    Returns:
        Process-style exit code (0 = success, 1 = error/invalid usage).
    """

    # Allow both direct calls (tests/snippets) and normal CLI invocation.
    argv = sys.argv[1:] if argv is None else argv

    # Help / no-argument path.
    if not argv or argv[0] in {'-h', '--help', 'help'}:
        _print_usage()
        return 1

    # Discover what this CLI is allowed to call.
    exports = _discover_exports()

    # List available exported symbols.
    if argv[0] in {'--list', 'list'}:
        names = sorted(exports)
        if not names:
            print('No functions are currently exported via __all__.')
            return 0
        print('Functions exported via __all__:')
        for name in names:
            print(f' - {name}')
        return 0

    # Treat first argument as function name, remaining args as raw positional
    # string arguments passed directly to the target callable.
    function_name = argv[0]
    args = argv[1:]

    # Find candidate module(s) exporting this name.
    candidate_modules = exports.get(function_name, [])
    if not candidate_modules:
        print(
            f"Error: Function '{function_name}' is not exported via __all__ in the WS_Mdl package. "
            "Use 'WS_Mdl --list' to see available functions."
        )
        return 1

    import_errors: list[str] = []
    for module_name in candidate_modules:
        try:
            # Lazy import: only import modules that may contain the function.
            module = importlib.import_module(module_name)
        except Exception as exc:
            # Keep searching other candidate modules before failing.
            import_errors.append(f'{module_name}: {exc}')
            continue

        # Extra safety: ensure symbol actually exists in imported module.
        if not hasattr(module, function_name):
            continue

        func = getattr(module, function_name)
        if not callable(func):
            print(f"Error: '{function_name}' exists in {module_name} but is not callable.")
            return 1

        try:
            # Arguments are passed as strings; conversion is handled by target func.
            result = func(*args)
        except TypeError as exc:
            # Usually indicates wrong number/type of positional arguments.
            print(f"Error calling '{function_name}' from {module_name}: {exc}")
            return 1

        # Print a return value only when one is explicitly produced.
        if result is not None:
            print(result)
        return 0

    # If imports failed for all candidates, report all failures to help debugging.
    if import_errors:
        print(f"Error: Could not import module(s) exporting '{function_name}':")
        for msg in import_errors:
            print(f' - {msg}')
        return 1

    # Fallback for inconsistent export maps (should be rare).
    print(f"Error: '{function_name}' was exported but no callable implementation was found.")
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
