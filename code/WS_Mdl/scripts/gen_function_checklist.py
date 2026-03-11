"""Generate a checklist tree of files, functions, classes, and methods.

The generated markdown file is meant for manual verification tracking:
- [ ] not checked/tested
- [x] checked/tested

Re-running this script refreshes discovered items and keeps existing checkmarks
by matching stable item IDs embedded in HTML comments.
"""

from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_CHECKBOX_RE = re.compile(r'^\s*- \[(?P<state>[ xX])\].*<!-- id:(?P<id>[^>]+) -->\s*$')


@dataclass(slots=True)
class ClassInfo:
    name: str
    line: int
    methods: list[tuple[str, int]]


@dataclass(slots=True)
class ModuleInfo:
    rel_path: str
    functions: list[tuple[str, int]]
    classes: list[ClassInfo]


def _iter_python_files(package_root: Path) -> list[Path]:
    """Return all Python files under package_root, excluding cache folders."""
    files: list[Path] = []
    for py_path in sorted(package_root.rglob('*.py')):
        rel_parts = py_path.relative_to(package_root).parts
        if '__pycache__' in rel_parts:
            continue
        files.append(py_path)
    return files


def _collect_module_info(py_path: Path, package_root: Path) -> ModuleInfo | None:
    """Parse one module and collect top-level functions/classes/methods."""
    try:
        source = py_path.read_text(encoding='utf-8')
        tree = ast.parse(source, filename=str(py_path))
    except Exception:
        # Skip unreadable or syntactically invalid files.
        return None

    functions: list[tuple[str, int]] = []
    classes: list[ClassInfo] = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append((node.name, node.lineno))
            continue

        if isinstance(node, ast.ClassDef):
            methods: list[tuple[str, int]] = []
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.append((child.name, child.lineno))

            classes.append(ClassInfo(name=node.name, line=node.lineno, methods=methods))

    if not functions and not classes:
        return None

    rel_path = py_path.relative_to(package_root).as_posix()
    return ModuleInfo(rel_path=rel_path, functions=functions, classes=classes)


def _load_checked_ids(output_path: Path) -> dict[str, bool]:
    """Load checkbox states from an existing checklist file."""
    checked_by_id: dict[str, bool] = {}
    if not output_path.exists():
        return checked_by_id

    for line in output_path.read_text(encoding='utf-8').splitlines():
        m = _CHECKBOX_RE.match(line)
        if not m:
            continue
        checked_by_id[m.group('id')] = m.group('state').lower() == 'x'

    return checked_by_id


def _checkbox(checked_by_id: dict[str, bool], item_id: str) -> str:
    return 'x' if checked_by_id.get(item_id, False) else ' '


def _append_item(lines: list[str], checked_by_id: dict[str, bool], item_id: str, label: str, indent: int = 0) -> None:
    pad = '  ' * indent
    state = _checkbox(checked_by_id, item_id)
    lines.append(f'{pad}- [{state}] {label} <!-- id:{item_id} -->')


def _render_markdown(package_root: Path, modules: list[ModuleInfo], checked_by_id: dict[str, bool]) -> str:
    """Create the checklist markdown content."""
    now_utc = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')

    lines: list[str] = []
    lines.append('# WS_Mdl Function Checklist')
    lines.append('')
    lines.append(f'Generated: {now_utc}')
    lines.append(f'Package root: {package_root.as_posix()}')
    lines.append('')
    lines.append('How to use:')
    lines.append('- Tick items with [x] when checked/tested.')
    lines.append('- Re-run this script anytime; existing checks are preserved by item ID.')
    lines.append('')

    if not modules:
        lines.append('No files with top-level functions/classes were found.')
        return '\n'.join(lines) + '\n'

    for module in modules:
        lines.append(f'## {module.rel_path}')
        file_id = f'file:{module.rel_path}'
        _append_item(lines, checked_by_id, file_id, f'FILE {module.rel_path}')

        for func_name, line in module.functions:
            func_id = f'func:{module.rel_path}:{func_name}'
            _append_item(lines, checked_by_id, func_id, f'FUNCTION {func_name} (L{line})', indent=1)

        for cls in module.classes:
            class_id = f'class:{module.rel_path}:{cls.name}'
            _append_item(lines, checked_by_id, class_id, f'CLASS {cls.name} (L{cls.line})', indent=1)

            for method_name, line in cls.methods:
                method_id = f'method:{module.rel_path}:{cls.name}.{method_name}'
                _append_item(
                    lines,
                    checked_by_id,
                    method_id,
                    f'METHOD {cls.name}.{method_name} (L{line})',
                    indent=2,
                )

        lines.append('')

    return '\n'.join(lines) + '\n'


def build_checklist(package_root: Path, output_path: Path) -> tuple[int, int]:
    """Generate checklist and return (module_count, item_count)."""
    checked_by_id = _load_checked_ids(output_path)

    modules: list[ModuleInfo] = []
    for py_path in _iter_python_files(package_root):
        info = _collect_module_info(py_path, package_root)
        if info is not None:
            modules.append(info)

    markdown = _render_markdown(package_root, modules, checked_by_id)
    output_path.write_text(markdown, encoding='utf-8')

    item_count = 0
    for module in modules:
        item_count += 1  # file item
        item_count += len(module.functions)
        item_count += len(module.classes)
        item_count += sum(len(cls.methods) for cls in module.classes)

    return len(modules), item_count


def main() -> int:
    package_default = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description='Generate file/function checklist markdown for WS_Mdl.')
    parser.add_argument(
        '--package-root',
        type=Path,
        default=package_default,
        help='Path to the package root to scan (default: WS_Mdl).',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=package_default / 'Auxi/FUNCTION_CHECKLIST.md',
        help='Output markdown file path.',
    )

    args = parser.parse_args()
    package_root = args.package_root.resolve()
    output_path = args.output.resolve()

    module_count, item_count = build_checklist(package_root, output_path)
    print(f'Checklist written to: {output_path}')
    print(f'Modules listed: {module_count}')
    print(f'Checklist items: {item_count}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
