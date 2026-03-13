# ---------- Model Number related Functions ----------
import re

from WS_Mdl.imod.ini import INIView, Mdl_Dmns

from .log import get_B
from .path import MdlN_PaView, imod_V
from .style import set_verbose

_MdlN_pattern = re.compile(r'^(?P<alias>[A-Za-z]+)(?P<N>\d+)$')


class Mdl_N:
    """
    Class representing a Model Number (MdlN) with an alias and a numeric component.
    Spelled as Mdl_N to avoid confict with MdlN argument used in most other functions.
    Provides attributes to access:
     - related paths
     - INI file content
     - dimensions
    """

    __slots__ = ('MdlN', 'alias', 'N', 'iMOD5', 'V', 'Pa', 'INI', 'Dmns', 'B', 'Pa_B')

    def __init__(self, MdlN: str, iMOD5: bool | None = None):
        # MdlN format validation and extraction of alias and number using regex
        m = _MdlN_pattern.match(MdlN)
        if not m:
            raise ValueError(f"Invalid MdlN format: {MdlN}. Expected format: Alias followed by number, e.g., 'Mdl123'.")

        # iMOD 5 type warning.
        if iMOD5 is not None and not isinstance(iMOD5, bool):
            raise TypeError('iMOD5 should be None, True, or False.')

        self.MdlN = MdlN
        self.alias = m.group('alias')
        self.N = int(m.group('N'))
        self.iMOD5 = iMOD5

        self.V = 'imod5' if iMOD5 is True else ('imod_python' if iMOD5 is False else imod_V(MdlN))
        self.Pa = MdlN_PaView(MdlN, iMOD5=(self.V == 'imod5'))

        set_verbose(False)
        self.INI = INIView(self.Pa.INI)
        self.Dmns = Mdl_Dmns(self.Pa.INI)
        set_verbose(True)

        self.B = get_B(MdlN)
        self.Pa_B = MdlN_PaView(self.B, iMOD5=(self.V == 'imod5'))
