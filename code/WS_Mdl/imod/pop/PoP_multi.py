# PoP_multi.py - standalone PoP for scenarios (ensemble members pooled): hd_multi -> diff_multi -> upload

import sys
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True, write_through=True)
sys.stderr.reconfigure(encoding='utf-8', line_buffering=True, write_through=True)

import types
sys.modules['primod'] = types.ModuleType('primod')   # temporary stub, remove when primod is installed

from WS_Mdl.imod.pop.hd_multi import (
    c_HD_Bin_AVGs, c_HD_Bin_GXGs, c_HD_Bin_Pctls, c_HD_Bin_Monthly,
)

from WS_Mdl.imod.pop.diff_multi import Diff_PoP_ScN

# --- Input ---
REF = {'REF_TEST': ['NBr128', 'NBr129']}                 # 1 reference scenario: ScN -> its members (MdlNs)
FUT = {'Hd_2050_TEST': ['NBr131', 'NBr132']}             # 1 or more future scenarios, same structure
l_Ls = [1]                                               # layers;
MdlN_T = 'NBr119' 					 # template for Up_MM_ScN
STEPS = {'hd': False, 'diff': False, 'qgis': False, 'upload': True}   # switch steps on/off
start_year = end_year = 'from_INI'                       # per scenario taken from the first member's INI


def run_hd(ScN, l_MdlN):
    kw = dict(ScN=ScN, l_Ls=l_Ls, start_year=start_year, end_year=end_year)
    c_HD_Bin_AVGs(l_MdlN, yearly=False, **kw)    # yearly=False: year labels of different members are different weather years
    c_HD_Bin_GXGs(l_MdlN, **kw)
    c_HD_Bin_Pctls(l_MdlN, **kw)
    c_HD_Bin_Monthly(l_MdlN, **kw)


if __name__ == '__main__':
    scenarios = REF | FUT
    ScN_ref = next(iter(REF))
    MdlN_any = next(iter(REF.values()))[0]       # any member, only supplies paths/alias

    if STEPS['hd']:
        for ScN, l_MdlN in scenarios.items():
            print(f'===== hd_multi: {ScN} ({", ".join(l_MdlN)}) =====')
            run_hd(ScN, l_MdlN)

    if STEPS['diff']:
        for ScN, l_MdlN in FUT.items():
            print(f'===== diff_multi: {ScN} minus {ScN_ref} =====')
            Diff_PoP_ScN(ScN, ScN_ref, l_MdlN[0])

    if STEPS['upload']:
        from WS_Mdl.core.defaults import Pa_WS
        from WS_Mdl.io.ibridges import Upl_Scn_Out
        Upl_Scn_Out(list(scenarios), MdlN_any, l_extra=[str(p.relative_to(Pa_WS)) for p in l_qgz])