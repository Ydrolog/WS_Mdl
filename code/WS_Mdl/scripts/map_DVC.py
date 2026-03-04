#!/usr/bin/env python

from pathlib import Path

import yaml


def main():
    base = Path('.')
    paths = set()

    for dvc_file in base.rglob('*.dvc'):
        with dvc_file.open('r') as f:
            meta = yaml.safe_load(f)
            for out in meta.get('outs', []):
                rel_path = (dvc_file.parent / out['path']).resolve(strict=False).relative_to(base.resolve())
                paths.add(rel_path)

    for dvc_yaml in base.rglob('dvc.yaml'):
        with dvc_yaml.open() as f:
            meta = yaml.safe_load(f)
            for stage in meta.get('stages', {}).values():
                for out in stage.get('outs', []):
                    if isinstance(out, dict):
                        rel_path = (dvc_yaml.parent / out['path']).resolve(strict=False).relative_to(base.resolve())
                        paths.add(rel_path)
                    else:
                        paths.add(Path(out))

    collapsed = set()
    for path in paths:
        parts = path.parts

        if path.is_file() or '.' in path.name:  # likely a file
            collapsed.add(path)
        elif len(parts) >= 2:
            collapsed.add(Path(parts[0]) / parts[1])
        elif parts:
            collapsed.add(Path(parts[0]))

    for path in sorted(collapsed, key=lambda p: p.as_posix()):
        print(path.as_posix())


if __name__ == '__main__':
    main()
