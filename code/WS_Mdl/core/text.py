import pathlib as Path


def r_Txt_Lns(Pa: Path | str) -> list[str]:
    """Reads a text file and returns its lines as a list."""
    with open(Pa, 'r', encoding='utf-8') as f:
        l_Ln = f.readlines()
    return l_Ln
