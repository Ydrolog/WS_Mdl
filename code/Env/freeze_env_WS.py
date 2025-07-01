#!/usr/bin/env python3

import subprocess
import sys
from WS_Mdl import utils as U

def main():
    # Get the current active conda environment
    try:
        active_env = subprocess.run(['conda', 'info', '--env'], 
                                  capture_output=True, text=True, check=True)
        
        # Parse output to find the active environment (marked with *)
        for line in active_env.stdout.splitlines():
            if '*' in line:
                env_name = line.split()[0]
                if env_name == '*':  # Handle case where * is separate
                    env_name = line.split()[1]
                break
        else:
            print("No active conda environment found", file=sys.stderr)
            sys.exit(1)
            
        print(f"Active conda environment: {env_name}")
        
        # Get model number for file naming
        MdlN = U.get_last_MdlN()
        
        # Export conda environment to YAML
        Fi = f'WS_env_{MdlN}.yml'
        print(f"Exporting conda environment to {Fi}...")
        
        with open(Fi, 'w') as f:
            subprocess.run(['conda', 'env', 'export', '--name', env_name, '--no-builds'], 
                          stdout=f, check=True)
            
        print(f"Environment successfully exported to {Fi}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
