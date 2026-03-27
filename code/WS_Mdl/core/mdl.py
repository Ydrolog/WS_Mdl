# ---------- Model Number related Functions ----------
import re
from datetime import datetime as DT

from WS_Mdl.imod.ini import INIView, Mdl_Aa, Mdl_Dmns

from .log import get_B
from .path import MdlN_PaView, imod_V
from .style import set_verbose

_MdlN_pattern = re.compile(r'^(?P<alias>[A-Za-z]+)(?P<N>\d+)$')


class Mdl_N:
    """
    Class representing a Model Number (MdlN). Spelled as Mdl_N to avoid confict with MdlN argument used in most other functions.
    """

    __slots__ = {
        'MdlN': 'str: Model Number (e.g., "NBr21")',
        'alias': 'str: Alias part of the Model Number (e.g., "NBr")',
        'N': 'int: Numeric part of the Model Number (e.g., 21)',
        'iMOD5': 'bool: Indicates if the model is an iMOD5 model',
        'V': 'str: Model version string ("imod5"/"imod_python")',
        'Pa': 'Paths view for the model',
        'INI': 'INI file content view',
        'Dmns': 'tuple: Xmin, Ymin, Xmax, Ymax, dx, dy (Model dimensions)',
        'Mdl_Aa': 'float: Model Area',
        'B': 'str: Baseline Model Number (e.g., "NBr18")',
        'Pa_B': 'Paths view for the Baseline Model',
        '__dict__': 'Allow dynamic attributes',
    }

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

        self.V = 'imod5' if iMOD5 is True else ('imod_python' if iMOD5 is False else imod_V(MdlN, iMOD5=iMOD5))
        self.Pa = MdlN_PaView(MdlN, iMOD5=(self.V == 'imod5'))

        set_verbose(False)
        self.INI = INIView(self.Pa.INI)
        if self.INI:
            self.Dmns = Mdl_Dmns(self.Pa.INI)
            self.Mdl_Aa = Mdl_Aa(self.Pa.INI)
            self.Xmin, self.Ymin, self.Xmax, self.Ymax, self.cellsize, self.N_R, self.N_C = Mdl_Dmns(self.Pa.INI)
            self.SP_1st, self.SP_last = [
                DT.strftime(DT.strptime(self.INI[f'{i}'], '%Y%m%d'), '%Y-%m-%d') for i in ['SDATE', 'EDATE']
            ]
        set_verbose(True)

        self.B = get_B(MdlN)
        self.Pa_B = MdlN_PaView(self.B, iMOD5=(self.V == 'imod5'))
