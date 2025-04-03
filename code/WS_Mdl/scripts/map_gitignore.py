import subprocess
from pathlib import Path

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

    def insert(self, parts):
        node = self
        for part in parts:
            if node.is_end:
                return False
            node = node.children.setdefault(part, TrieNode())
        node.is_end = True
        node.children.clear()
        return True

def main():
    result = subprocess.run(
        ["git", "ls-files", "--others", "--ignored", "--exclude-standard"],
        stdout=subprocess.PIPE, text=True, check=True
    )

    base = Path.cwd().resolve()
    parents = {Path(p).parent.resolve() for p in result.stdout.splitlines()}

    # Apply manual override: reduce anything under `.dvc` to `.dvc`
    simplified = set()
    for p in parents:
        try:
            i = p.parts.index(".dvc")
            simplified.add(Path(*p.parts[:i+1]))  # e.g., Path(".dvc")
        except ValueError:
            simplified.add(p)

    trie = TrieNode()
    collapsed = []

    for path in sorted(simplified):
        if trie.insert(path.parts):
            collapsed.append(path)

    for path in collapsed:
        print(path.relative_to(base).as_posix())

if __name__ == "__main__":
    main()
