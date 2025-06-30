#!/usr/bin/env python3

import os
import subprocess
import sys
from WS_Mdl import utils as U


def run(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode:
        print(f"Error running command: {cmd}\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()


def main():
    # Detect active conda environment
    conda_prefix = os.getenv('CONDA_PREFIX')
    conda_name = os.getenv('CONDA_DEFAULT_ENV') or os.path.basename(conda_prefix or '')
    if not conda_prefix:
        print("No active conda environment detected. Activate an environment first.", file=sys.stderr)
        sys.exit(1)

    MdlN = U.get_last_MdlN()
    repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    package_name = "WS_Mdl"

    # Export conda environment without builds
    yml_file = f"WS_env_{MdlN}.yml"
    with open(yml_file, 'w') as f:
        subprocess.run([
            'conda', 'env', 'export', '--prefix', conda_prefix, '--no-builds'
        ], stdout=f, check=True)

    # Read and modify YAML
    lines = open(yml_file).read().splitlines(keepends=True)
    new_lines = []
    # Inject name and remove prefix
    for line in lines:
        if line.startswith('prefix:'):
            continue
        new_lines.append(line)
    # Ensure name is first non-comment
    for idx, line in enumerate(new_lines):
        if not line.lstrip().startswith('#'):
            new_lines.insert(idx, f'name: {conda_name}\n')
            break

    # Determine Git version: tag or commit
    try:
        version = run(f'git -C "{repo_path}" describe --tags --exact-match')
    except SystemExit:
        version = run(f'git -C "{repo_path}" rev-parse HEAD')
    remote_url = run(f'git -C "{repo_path}" remote get-url origin')

    # Inject pip section
    out = []
    for line in new_lines:
        out.append(line)
        if line.strip() == 'dependencies:':
            out.append('  - pip:\n')
            out.append(f'    - git+{remote_url}@{version}#egg={package_name}\n')

    with open(yml_file, 'w') as f:
        f.writelines(out)

    print(f"Done. Generated: {yml_file} (env name: '{conda_name}')")


if __name__ == '__main__':
    main()
