#!/usr/bin/env python3

import os
from os import listdir as LD, makedirs as MDs
from os.path import join as PJ, basename as PBN, dirname as PDN, exists as PE
import subprocess
import sys
import pandas as pd
from WS_Mdl import utils as U

def run(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running command: {cmd}\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()

def main():
    # Fixed environment name
    env_name = "WS"
    MdlN = U.get_last_MdlN()

    # Fixed local Git repository path
    Pa_repo = r"C:\OD\WS_Mdl"
    package_name = "WS_Mdl"
    # Export conda environment
    print(f"Exporting conda environment '{env_name}' to environment.yml...")
    try:
        yml = f'WS_conda_env_{MdlN}.yml'
        with open(yml, 'w') as f:
            subprocess.run(['conda', 'env', 'export', '--name', env_name, '--no-builds'], stdout=f, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error exporting conda environment: {e}", file=sys.stderr)
        sys.exit(1)
    # Freeze pip packages
    print("Freezing pip packages to pip_env.txt...")
    pip_freeze_env = f'WS_pip_env_{MdlN}.txt'
    try:
        with open(pip_freeze_env, 'w') as f:
            subprocess.run(['pip', 'freeze'], stdout=f, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error freezing pip packages: {e}", file=sys.stderr)
        sys.exit(1)
    # Verify Git repo path and extract remote URL + commit
    if not os.path.isdir(PJ(Pa_repo, '.git')):
        print(f"Error: {Pa_repo} is not a Git repository.", file=sys.stderr)
        sys.exit(1)

    remote_url = run(f'git -C "{Pa_repo}" remote get-url origin')
    commit_hash = run(f'git -C "{Pa_repo}" rev-parse HEAD')
    git_line = f"git+{remote_url}@{commit_hash}#egg={package_name}"
    print(f"Appending local Git package: {git_line}")
    with open(pip_freeze_env, 'a') as f:
        f.write(git_line + '\n')

    print(f"Done. Files generated: {yml},  {pip_freeze_env}")

if __name__ == '__main__':
    main()