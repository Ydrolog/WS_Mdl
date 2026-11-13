"""
PoP_multy.py - PoP for KNMI'23 scenarios (members pooled), run once per VM.

Each VM runs ONE scenario with all its members (the pooled statistics need the bins of all members on the same machine).
The reference scenario is shared via Yoda.

Order of the steps (switch them on/off in STEPS):
    hd            Pooled statistics per scenario (AVG, GXG, percentiles, monthly) -> PoP/Out/<ScN>/
    download_ref  Download the reference scenario from Yoda (only needed on a future-scenario VM)
    diff          Diff = future scenario minus reference scenario -> next to the future scenario files
    upload        Upload PoP/Out/<ScN> (incl. Diff files) to Yoda

How to use:
    1. Set LOCAL (ScN -> members), ScN_ref and l_Ls at the top of the script.
    2. REF VM:
         STEPS: hd=True, upload=True, rest False
    3. Future-scenario VM, first pass (so the result is on Yoda):
         STEPS: hd=True, upload=True, rest False
    4. Future-scenario VM, second pass, AFTER the REF VM has finished uploading:
         STEPS: download_ref=True, diff=True, upload=True, hd=False
    5. Run from the pixi shell:
         python -u run_PoP.py 2>&1 | Tee-Object log_PoP.txt

Checks:
    - The members of a scenario must have the same period and be complete (o_HD_Bin_Multi stops otherwise).
    - Scenario names (ScN) must be identical on all VMs, because they are part of the file names.
    - After download_ref, check that PoP/Out/<ScN_ref> has all subfolders (GW_HD_AVGs, GXG, GW_HD_Pct, GW_HD_Monthly).
    - Not included: WB (water balance), Outlet_TS, p_HD_OBS_TS, QGIS (Up_MM).
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True, write_through=True)
sys.stderr.reconfigure(encoding='utf-8', line_buffering=True, write_through=True)

import types
sys.modules['primod'] = types.ModuleType('primod')  # temporary stub, remove when primod is installed

from WS_Mdl.core.mdl import Mdl_N
from WS_Mdl.imod.pop.diff_multi import Diff_PoP_ScN
from WS_Mdl.imod.pop.hd_multi import c_HD_Bin_AVGs, c_HD_Bin_GXGs, c_HD_Bin_Monthly, c_HD_Bin_Pctls

# --- Input (change per VM) ---
LOCAL = {'Hd_2050': ['NBr131', 'NBr132','NBr133']}  # scenario(s) whose runs are on this VM: ScN -> members
ScN_ref = 'REF_TEST'  # reference scenario. If it is in LOCAL, no diff is made here
l_Ls = [1]  # layers; 
STEPS = {'hd': False, 'download_ref': False, 'diff': True, 'upload': False}
start_year = end_year = 'from_INI'  # per scenario taken from the first member's INI


def run_hd(ScN, l_MdlN):
    kw = dict(ScN=ScN, l_Ls=l_Ls, start_year=start_year, end_year=end_year)
    c_HD_Bin_AVGs(l_MdlN, yearly=False, **kw)  # yearly=False: year labels of different members are different weather years
    c_HD_Bin_GXGs(l_MdlN, **kw)
    c_HD_Bin_Pctls(l_MdlN, **kw)
    c_HD_Bin_Monthly(l_MdlN, **kw)


if __name__ == '__main__':
    MdlN_any = next(iter(LOCAL.values()))[0]  # any local member, only supplies paths/alias
    l_FUT = [ScN for ScN in LOCAL if ScN != ScN_ref]

    if STEPS['hd']:
        for ScN, l_MdlN in LOCAL.items():
            print(f'===== hd_multi: {ScN} ({", ".join(l_MdlN)}) =====')
            run_hd(ScN, l_MdlN)

    if STEPS['download_ref'] and ScN_ref not in LOCAL:
        from WS_Mdl.io.ibridges import Dl, iB_session

        print(f'===== download {ScN_ref} from Yoda =====')
        Dl(f'models/{Mdl_N(MdlN_any).alias}/PoP/Out/{ScN_ref}', iB_session(), overwrite=False)

    if STEPS['diff']:
        for ScN in l_FUT:
            print(f'===== diff_multi: {ScN} minus {ScN_ref} =====')
            Diff_PoP_ScN(ScN, ScN_ref, LOCAL[ScN][0])

    if STEPS['upload']:
        from WS_Mdl.io.ibridges import Upl_Scn_Out

        Upl_Scn_Out(list(LOCAL), MdlN_any)