#!/usr/bin/env python3

import subprocess, sys, argparse, filecmp, os
from pathlib import Path
from WS_Mdl import utils as U


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Export conda environment to YAML file.')
    parser.add_argument('--MdlN', type=str, help='Model number for file naming')
    parser.add_argument(
        '--no-dedup', dest='dedup', action='store_false', help='Disable identical-file pruning'
    )
    parser.set_defaults(dedup=True)  # deduplication ON by default
    args = parser.parse_args()

    # Get the current active conda environment
    try:
        active_env = subprocess.run(
            ['conda', 'info', '--env'], capture_output=True, text=True, check=True
        )

        # Parse output to find the active environment (marked with *)
        for line in active_env.stdout.splitlines():
            if '*' in line:
                env_name = line.split()[0]
                if env_name == '*':  # Handle case where * is separate
                    env_name = line.split()[1]
                break
        else:
            print('No active conda environment found', file=sys.stderr)
            sys.exit(1)

        print(f'Active conda environment: {env_name}')

        # Get model number for file naming - either from args or default
        if args.MdlN:
            MdlN = args.MdlN
            print(f'Using provided model number: {MdlN}')
        else:
            MdlN = U.get_last_MdlN()
            print(f'Using default model number: {MdlN}')

        # Export conda environment to YAML (same folder as this script)
        script_dir = Path(__file__).resolve().parent
        Fi = script_dir / f'WS_env_{MdlN}.yml'
        print(f'Exporting conda environment to {Fi} …')

        with Fi.open('w') as f:
            subprocess.run(
                ['conda', 'env', 'export', '--name', env_name, '--no-builds'], stdout=f, check=True
            )

        print(f'Environment successfully exported to {Fi}')
        # optional deduplication
        if args.dedup:
            others = [p for p in script_dir.glob('WS_env_*.yml') if p != Fi]
            if others:
                latest = max(others, key=lambda p: p.stat().st_mtime)
                if filecmp.cmp(Fi, latest, shallow=False):
                    print(f'{Fi.name} is identical to {latest.name} -> deleting redundant file.')
                    os.remove(Fi)
                    return
        print(f'Kept {Fi.name}')

    except subprocess.CalledProcessError as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
