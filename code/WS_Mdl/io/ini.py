from pathlib import Path

from WS_Mdl.core.style import sprint


def r_as_d(Pa_INI: Path | str) -> dict:
    """
    Reads INI file (used for preparing the model) and converts it to a dictionary. Keys are converted to upper-case.
    Common use:
    d_INI = INI_to_d(Pa_INI)
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


class INIView:
    """MdlN accessor/view."""

    __slots__ = ('_d', '_path')

    def __init__(self, Pa_INI):
        self._path = Path(Pa_INI)
        self._d = r_as_d(self._path)

    def __getattr__(self, name: str):
        try:
            return self._d[name.upper()]
        except KeyError as e:
            raise AttributeError(name) from e

    def __getitem__(self, key: str):
        return self._d[key.upper()]

    def keys(self):
        return self._d.keys()


def Mdl_Dmns_from_INI(Pa_INI):
    d = r_as_d(Pa_INI)

    Xmin, Ymin, Xmax, Ymax = map(float, d['WINDOW'].split(','))
    cellsize = float(d['CELLSIZE'])
    N_R = int(-(Ymin - Ymax) / cellsize)
    N_C = int((Xmax - Xmin) / cellsize)

    return Xmin, Ymin, Xmax, Ymax, cellsize, N_R, N_C
