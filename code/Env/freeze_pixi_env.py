#!/usr/bin/env python3
import subprocess
import sys
from datetime import datetime as DT
from pathlib import Path

# from WS_Mdl import utils as U

# Pa_toml, Pa_lock, Pa_WS_Mdl = U.

# Paths relative to repo root
FILES_TO_TRACK = ['pixi.toml', 'pixi.lock', 'WS_Mdl/']


def run_cmd(cmd, check=True, capture=False):
    return subprocess.run(cmd, check=check, capture_output=capture, text=True)


def main():
    try:
        # Ensure we are in repo root
        repo_root = run_cmd(['git', 'rev-parse', '--show-toplevel'], capture=True).stdout.strip()
        print(f'Repo root: {repo_root}')
        Path(repo_root)  # not really needed but keeps structure

        # Check for changes in the relevant files
        diff_cmd = ['git', 'status', '--porcelain'] + FILES_TO_TRACK
        changes = run_cmd(diff_cmd, capture=True).stdout.strip()

        if not changes:
            print('No changes to tracked env/code files. Nothing to commit.')
            return

        print('Changes detected:\n' + changes)

        # Stage changes
        run_cmd(['git', 'add'] + FILES_TO_TRACK)
        print('Staged changes.')

        # Commit with timestamp
        now = DT.now().strftime('%Y-%m-%d %H:%M:%S')
        commit_msg = f'Model run prep – {now}'
        run_cmd(['git', 'commit', '-m', commit_msg])
        print(f"Committed changes with message: '{commit_msg}'")

    except subprocess.CalledProcessError as e:
        print(f'Error running command: {e}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
