import subprocess
from pathlib import Path

def main():
    repo_root = Path.cwd().resolve()

    # Get .gitignore'd files
    result = subprocess.run(
        ["git", "ls-files", "--others", "--ignored", "--exclude-standard"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )

    folders = set()
    for line in result.stdout.strip().splitlines():
        file_path = Path(line).resolve()

        # Skip files outside the repo root
        try:
            rel_path = file_path.relative_to(repo_root)
        except ValueError:
            continue

        parent = rel_path.parent
        parts = parent.parts

        # Collapse to first 2 folder levels
        if len(parts) >= 2:
            folders.add(Path(parts[0]) / parts[1])
        elif parts:
            folders.add(Path(parts[0]))

    for folder in sorted(folders):
        print(folder.as_posix())
