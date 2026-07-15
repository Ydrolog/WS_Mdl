import re
from pathlib import Path


__all__ = ['r_Txt_Lns', 'replace_MdlN']


def replace_MdlN(value: str | Path, old: str, new: str) -> str | Path:
    """Replace a model number without matching the start of a larger number."""

    if not old:
        raise ValueError('old model number cannot be empty')

    replaced = re.sub(rf'{re.escape(old)}(?!\d)', lambda _: new, str(value))
    return Path(replaced) if isinstance(value, Path) else replaced


def r_Txt_Lns(Pa: Path | str) -> list[str]:
    """Reads a text file and returns its lines as a list."""
    with open(Pa, 'r', encoding='utf-8') as f:
        l_Ln = f.readlines()
    return l_Ln
