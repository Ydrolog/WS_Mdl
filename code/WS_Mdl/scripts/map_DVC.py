#!/usr/bin/env python

import os
import yaml

def main():
    paths = set()

    for root, dirs, files in os.walk(".", topdown=True):
        for file in files:
            if file.endswith(".dvc"):
                with open(os.path.join(root, file), "r") as f:
                    meta = yaml.safe_load(f)
                    for out in meta.get("outs", []):
                        full_path = os.path.normpath(os.path.join(root, out["path"]))
                        rel_path = os.path.relpath(full_path, start=".")
                        paths.add(rel_path)

    for root, dirs, files in os.walk(".", topdown=True):
        if "dvc.yaml" in files:
            with open(os.path.join(root, "dvc.yaml")) as f:
                meta = yaml.safe_load(f)
                for stage in meta.get("stages", {}).values():
                    for out in stage.get("outs", []):
                        if isinstance(out, dict):
                            full_path = os.path.normpath(os.path.join(root, out["path"]))
                            rel_path = os.path.relpath(full_path, start=".")
                            paths.add(rel_path)
                        else:
                            paths.add(out)

    collapsed = set()
    for path in paths:
        parts = os.path.normpath(path).split(os.sep)

        if os.path.isfile(path) or '.' in parts[-1]:  # likely a file
            collapsed.add(path)
        elif len(parts) >= 2:
            collapsed.add(os.path.join(parts[0], parts[1]))
        elif parts:
            collapsed.add(parts[0])

    for path in sorted(collapsed):
        print(path.replace(os.sep, "/"))


if __name__ == "__main__":
    main()
