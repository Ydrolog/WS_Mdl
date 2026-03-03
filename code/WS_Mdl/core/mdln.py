# ---------- Model Number related Functions ----------
import re
from dataclasses import dataclass

_MdlN_pattern = re.compile(r'^(?P<alias>[A-Za-z]+)(?P<N>\d+)$')


@dataclass(frozen=True, slots=True)
class MdlN:
    """
    Class representing a Model Number (MdlN) with an alias and a numeric component. Provides properties to access:
     - related paths
     - INI file content
     - dimensions
    """

    MdlN: str

    def __post_init__(self):
        m = _MdlN_pattern.match(self.MdlN)
        if not m:
            raise ValueError(
                f"Invalid MdlN format: {self.MdlN}. Expected format: Alias followed by number, e.g., 'Mdl123'."
            )
        object.__setattr__(self, 'alias', m.group('alias'))
        object.__setattr__(self, 'N', int(m.group('N')))

    @property
    def Pa(self):
        from WS_Mdl.core.path import MdlN_PaView

        return MdlN_PaView(self.MdlN)

    @property
    def INI(self):
        from WS_Mdl.io.ini import INIView

        return INIView(self.Pa.Pa_INI)

    @property
    def Dmns(self):
        from WS_Mdl.io.ini import Mdl_Dmns_from_INI

        return Mdl_Dmns_from_INI(self.Pa.P_INI)

    @property
    def V(self):
        from WS_Mdl.core.path import imod_V

        return imod_V(self.MdlN)
