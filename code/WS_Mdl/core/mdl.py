# ---------- Model Number related Functions ----------
import re
from dataclasses import dataclass, field

_MdlN_pattern = re.compile(r'^(?P<alias>[A-Za-z]+)(?P<N>\d+)$')


@dataclass(frozen=True, slots=True)
class Mdl_N:
    """
    Class representing a Model Number (MdlN) with an alias and a numeric component.
    It's spelled as Mdl_N to avoid confict with MdlN argument used in most other functions.
    Provides properties to access:
     - related paths
     - INI file content
     - dimensions
    """

    MdlN: str
    iMOD5: bool | None = field(default=None, repr=False, compare=False)
    alias: str = field(init=False)
    N: int = field(init=False)
    _pa_cache: object = field(init=False, repr=False, compare=False, default=None)
    _ini_cache: object = field(init=False, repr=False, compare=False, default=None)
    _v_cache: str | None = field(init=False, repr=False, compare=False, default=None)

    def __post_init__(self):
        m = _MdlN_pattern.match(self.MdlN)
        if not m:
            raise ValueError(
                f"Invalid MdlN format: {self.MdlN}. Expected format: Alias followed by number, e.g., 'Mdl123'."
            )
        if self.iMOD5 is not None and not isinstance(self.iMOD5, bool):
            raise TypeError('iMOD5 should be None, True, or False.')

        object.__setattr__(self, 'alias', m.group('alias'))
        object.__setattr__(self, 'N', int(m.group('N')))
        object.__setattr__(self, '_pa_cache', None)
        object.__setattr__(self, '_ini_cache', None)
        object.__setattr__(
            self,
            '_v_cache',
            'imod5' if self.iMOD5 is True else ('imod_python' if self.iMOD5 is False else None),
        )

    @property
    def Pa(self):
        """Returns a dictionary of paths related to the model number. But also makes them accessible as attributes."""
        cached = self._pa_cache
        if cached is None:
            from WS_Mdl.core.path import MdlN_PaView

            cached = MdlN_PaView(self.MdlN, iMOD5=(self.V == 'imod5'))
            object.__setattr__(self, '_pa_cache', cached)

        return cached

    @property
    def INI(self):
        """Returns the content of the INI file as a dictionary."""
        cached = self._ini_cache
        if cached is None:
            from WS_Mdl.imod.ini import INIView

            cached = INIView(self.Pa.INI)
            object.__setattr__(self, '_ini_cache', cached)

        return cached

    @property
    def Dmns(self):
        """Returns the model dimensions as a tuple (Xmin, Ymin, Xmax, Ymax, cellsize, N_R, N_C)."""
        from WS_Mdl.imod.ini import Mdl_Dmns

        return Mdl_Dmns(self.Pa.INI)

    @property
    def V(self):
        """Returns the iMOD version of the model."""
        cached = self._v_cache
        if cached is not None:
            return cached

        from WS_Mdl.core.path import imod_V

        cached = imod_V(self.MdlN)
        object.__setattr__(self, '_v_cache', cached)
        return cached

    @property
    def B(self):
        """Returns the Baseline Sim."""
        from WS_Mdl.core.log import get_B

        return get_B(self.MdlN)

    @property
    def Pa_B(self):
        """Returns the paths of the Baseline Sim."""
        from WS_Mdl.core.path import MdlN_PaView

        return MdlN_PaView(self.MdlN, iMOD5=(self.V == 'imod5')).B(self.B)
