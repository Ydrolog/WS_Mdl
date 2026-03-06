from pathlib import Path

from WS_Mdl.core.path import MdlN_Pa
from WS_Mdl.core.style import sprint


def as_d(Pa_INI: Path | str) -> dict:
    """
    Reads INI file (used for preparing the model) and converts it to a dictionary. Keys are converted to upper-case.
    Common use:
    d_INI = as_d(Pa_INI)
    Xmin, Ymin, Xmax, Ymax = [float(i) for i in d_INI['WINDOW'].split(',')]
    cellsize = float(d_INI['CELLSIZE'])
    N_R, N_C = int( - (Ymin - Ymax) / cellsize ), int( (Xmax - Xmin) / cellsize )
    """
    d_INI = {}
    with open(Pa_INI, 'r', encoding='utf-8') as file:
        for Ln in file:
            Ln = Ln.strip()
            if Ln and not Ln.startswith('#'):  # Ignore empty lines and comments
                k, v = Ln.split('=', 1)  # Split at the first '='
                d_INI[k.strip().upper()] = v.strip()  # Remove extra spaces

    sprint(f'🟢 - INI file {Pa_INI} read successfully. Dictionary created with {len(d_INI)} keys.')
    return d_INI


class INIView(dict):
    """Dictionary of INI values with attribute-style key access."""

    __slots__ = ('_path',)

    def __init__(self, Pa_INI: Path | str):
        self._path = Path(Pa_INI)
        super().__init__(as_d(self._path))

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __getitem__(self, key):
        if isinstance(key, str):
            key = key.upper()
        return super().__getitem__(key)

    def get(self, key, default=None):
        if isinstance(key, str):
            key = key.upper()
        return super().get(key, default)

    def __contains__(self, key):
        if isinstance(key, str):
            key = key.upper()
        return super().__contains__(key)


def Mdl_Dmns(Pa_INI):
    d = as_d(Pa_INI)

    Xmin, Ymin, Xmax, Ymax = map(float, d['WINDOW'].split(','))
    cellsize = float(d['CELLSIZE'])
    N_R = int(-(Ymin - Ymax) / cellsize)
    N_C = int((Xmax - Xmin) / cellsize)

    return Xmin, Ymin, Xmax, Ymax, cellsize, N_R, N_C


def CeCes(MdlN: str):
    """Get centroids of the model grid from the INI file of the model.
    Returns x_CeCes, y_CeCes."""
    import numpy as np

    Xmin, Ymin, Xmax, Ymax, cellsize, N_R, N_C = Mdl_Dmns(MdlN_Pa(MdlN)['INI'])
    dx = float(cellsize)
    dy = -float(cellsize)

    return np.arange(Xmin + dx / 2, Xmax, dx), np.arange(Ymax + dy / 2, Ymin, dy)
