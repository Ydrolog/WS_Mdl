# ---------- Model Number related Functions ----------
import re
from dataclasses import dataclass
from datetime import datetime as DT
from functools import cached_property

from WS_Mdl.imod.ini import CeCes, INIView, Mdl_area, Mdl_Dmns

from .path import MdlN_PaView, imod_V
from .style import set_verbose

_MdlN_pattern = re.compile(r'^(?P<alias>[A-Za-z]+)(?P<N>\d+)$')


@dataclass
class Sim:
    verbose: bool = False
    Bin_Ins: bool = True
    save_budget: str = 'last'  # 'last', 'all', None, or number specifying frequency
    save_head: str = 'last'  # 'last', 'all', None, or number specifying frequency


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
        'Mdl_area': 'float: Model Area',
        'Xmin': 'float: Minimum X coordinate',
        'Ymin': 'float: Minimum Y coordinate',
        'Xmax': 'float: Maximum X coordinate',
        'Ymax': 'float: Maximum Y coordinate',
        'Xs': 'float: X coordinates of the model grid',
        'Ys': 'float: Y coordinates of the model grid',
        'cellsize': 'float: Model cell size',
        'N_R': 'int: Number of rows in the model grid',
        'N_C': 'int: Number of columns in the model grid',
        'SP_1st': 'str: Start date of the model simulation period (YYYY-MM-DD)',
        'SP_last': 'str: End date of the model simulation period (YYYY-MM-DD)',
        'SP_1st_DT': 'datetime: Start date as datetime object',
        'SP_last_DT': 'datetime: End date as datetime object',
        'Sim': 'Sim settings container for model-specific simulation options',
        '__dict__': 'dict: allow dynamic attributes',
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
        self.Sim = Sim()

        set_verbose(False)  # To avoid INI prints
        self.INI = INIView(self.Pa.INI)
        if self.INI:
            self.Dmns = Mdl_Dmns(self.Pa.INI)
            self.Mdl_area = Mdl_area(self.Pa.INI)
            self.Xmin, self.Ymin, self.Xmax, self.Ymax, self.cellsize, self.N_R, self.N_C = Mdl_Dmns(self.Pa.INI)
            self.Xs, self.Ys = CeCes(self.MdlN)
            self.N_L_cells = self.N_R * self.N_C
            self.SP_1st, self.SP_last = [
                DT.strftime(DT.strptime(self.INI[f'{i}'], '%Y%m%d'), '%Y-%m-%d') for i in ['SDATE', 'EDATE']
            ]
            self.SP_1st_DT, self.SP_last_DT = (
                DT.strptime(self.SP_1st, '%Y-%m-%d'),
                DT.strptime(self.SP_last, '%Y-%m-%d'),
            )
            self.cell_area = self.cellsize**2
        set_verbose(True)

    @cached_property
    def B(self):
        from .log import get_B

        return get_B(self.MdlN)

    @cached_property
    def Pa_B(self):
        return MdlN_PaView(self.B, iMOD5=(self.V == 'imod5'))

    @cached_property
    def MSW_In(self):
        from WS_Mdl.imod.msw.input import MSW_In

        return MSW_In(self)

    @cached_property
    def Pkgs(self):  # Refers to MF6 packages, as the pack
        from WS_Mdl.imod.mf6.nam import l_Pkgs

        return l_Pkgs(self.MdlN)

    @property
    def _PRJ_bundle(self):
        from WS_Mdl.imod.prj import r_with_OBS

        return r_with_OBS(
            self.Pa.PRJ
        )  # , season_to_DT=False) # Fix for ufunc 'greater_equal' did not contain a loop with signature matching types (<class 'numpy.dtypes.Float64DType'>, <class 'numpy.dtypes.DateTime64DType'>) -> None

    @property
    def PRJ(self):
        return self._PRJ_bundle[0]

    @property
    def PRJ_OBS(self):
        return self._PRJ_bundle[1]

    @property
    def N_L(self):
        return self.PRJ['(bot)']['n_system']

    @property
    def vars(self):
        """Return all slotted attributes and their current values as a dict."""
        return {k: getattr(self, k, None) for k in self.__slots__}
