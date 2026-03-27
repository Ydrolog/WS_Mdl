import os
from concurrent.futures import ProcessPoolExecutor as PPE
from datetime import datetime as DT
from os import makedirs as MDs
from os.path import basename as PBN
from os.path import join as PJ
from pathlib import Path

import imod
import numpy as np
import pandas as pd
import xarray as xra
from WS_Mdl.core.mdl import Mdl_N
from WS_Mdl.core.style import Sep, set_verbose, sprint
from WS_Mdl.imod.idf import HD_Out_to_DF
from WS_Mdl.xr.compare import Diff_MBTIF
from WS_Mdl.xr.convert import to_MBTIF, to_TIF


def HD_Bin_GXG_to_MBTIF(
    MdlN, start_year='from_INI', end_year='from_INI', IDT='from_INI', GVG=False, l_Ls=[1, 3, 5, 7, 9]
):
    """
    - start_year: 'YYYY' (inclusive) or 'from_INI' to read from INI file
    - end_year: 'YYYY' (inclusive) or 'from_INI' to read from INI file
    - IDT: Number of days per SP. #666 this should be upgraded to use ITT too, to allow for other time units (e.g. hour).
    - GVG: Boolean, whether to include GVG in the output
    """
    sprint(Sep)
    set_verbose(False)

    # Load standard imod paths and variables
    M = Mdl_N(MdlN)

    start_year = int(M.INI.SDATE[:4]) if start_year == 'from_INI' else int(start_year)
    end_year = int(M.INI.EDATE[:4]) if end_year == 'from_INI' else int(end_year)
    IDT = int(M.INI.IDT) if IDT == 'from_INI' else int(IDT)

    l_years = [i for i in range(start_year, end_year + 1)]
    set_verbose(True)

    # 1. Load & trim Bin HD file
    DA_HD = imod.mf6.open_hds(hds_path=M.Pa.HD_Out_Bin, grb_path=M.Pa.DIS_GRB)
    dates = pd.date_range(start=str(start_year), periods=DA_HD.time.size, freq=f'{IDT}D')
    DA_HD = DA_HD.assign_coords(time=dates)  # Assign to DA_HD
    DA_HD = DA_HD.where(DA_HD.time.dt.year.isin(l_years), drop=True).sel(layer=l_Ls)  # Select specific years and layers
    sprint(f' 🟢🟢 - Loaded HD file from {M.Pa.HD_Out_Bin}')

    # 2. GXG
    ##  Calculate GXG
    d_GXG = {}
    for L in l_Ls:
        DA_HD_L = DA_HD.sel(layer=L)
        GXG = imod.evaluate.calculate_gxg(DA_HD_L).load()
        GXG = GXG.rename_vars({var: var.upper() for var in GXG.data_vars})

        # Get N_years
        N_years_GXG = np.unique(GXG.N_YEARS_GXG.values).max()
        N_years_GVG = np.unique(GXG.N_YEARS_GVG.values).max()

        # Calculate GW range: GHG-GLG
        GXG['GHG_m_GLG'] = GXG['GHG'] - GXG['GLG']
        GXG = GXG[['GHG', 'GLG', 'GHG_m_GLG', 'GVG']] if GVG else GXG[['GHG', 'GLG', 'GHG_m_GLG']]

        # Collect results
        for var in GXG.data_vars:
            if var not in d_GXG:
                d_GXG[var] = []
            d_GXG[var].append(GXG[var])
    ## Concatenate
    for var in d_GXG:
        if isinstance(d_GXG[var], list):
            d_GXG[var] = xra.concat(d_GXG[var], dim=pd.Index(l_Ls, name='layer'))
    sprint(f' 🟢🟢 - Calculated GXG for {MdlN}')

    # 3. Save to MBTIF
    (M.Pa.PoP_Out_MdlN / 'GXG').mkdir(parents=True, exist_ok=True)

    for K, GXG in d_GXG.items():
        L_min, L_max = GXG.layer.values.min(), GXG.layer.values.max()

        Pa_Out = M.Pa.PoP_Out_MdlN / 'GXG' / f'{K}_L{L_min}-{L_max}_{MdlN}.tif'

        d_MtDt = {
            f'{K}_L{L_min}-{L_max}_{MdlN}': {
                'AVG': float(GXG.mean().values),
                'coordinates': GXG.coords,
                'period': f'{start_year}-{end_year}',
                'N_years': N_years_GVG if K == 'GVG' else N_years_GXG,
                'variable': Pa_Out.name[0],
                'details': f'{MdlN} {K} calculated from (path: {M.Pa.HD_Out_Bin}), via function described in: https://deltares.github.io/imod-python/api/generated/evaluate/imod.evaluate.calculate_gxg.html',
            }
        }

        to_MBTIF(GXG, Pa_Out, d_MtDt, _print=False)
        sprint(f'  🟢 - Saved {K} to {Pa_Out}')
    sprint(f'🟢🟢🟢 - HD_Bin_GXG_to_MBTIF finished successfully for {MdlN}.')
    sprint(Sep)


def GXG_Diff(MdlN_1, MdlN_2):
    """
    Calcs Diff between GXG of two MdLNs, and saves the Diff as a TIF in the same dir as the MdlN_1 GXG files.
    Warning!: It doesn't check if the dates used for calculating the GXGs are the same.
    """

    Pa_PoP_GXG = Mdl_N(MdlN_1).Pa.PoP_Out_MdlN / 'GXG'
    l_GXG_Fi = [i for i in Pa_PoP_GXG.iterdir() if i.is_file() and i.suffix == '.tif']

    for Fi in l_GXG_Fi:
        Pa_TIF_1 = Pa_PoP_GXG / Fi.name
        Pa_TIF_2 = Pa_PoP_GXG / Path(str(Fi).replace(MdlN_1, MdlN_2))
        Pa_Out = Pa_TIF_1.parent / Pa_TIF_1.name.replace(
            MdlN_1, f'{MdlN_1}m{"".join(i for i in MdlN_2 if i.isdigit())}'
        )
        # print(Pa_TIF_1, Pa_TIF_2, Pa_Out, end='\n---------------\n', sep='\n')
        try:
            Diff_MBTIF(Pa_TIF_1, Pa_TIF_2, Pa_Out)
        except (FileNotFoundError, OSError) as e:
            print(f'🔴 - Missing/unreadable file(s) for {Fi}: {e}')


def HD_IDF_GXG_to_TIF(MdlN: str, N_cores: int = None, CRS: str = None, rules: str = None, iMOD5=False):
    """Reads Sim Out IDF files from the model directory and calculates GXG for each L. Saves them as MultiBand TIF files - each band representing one of the GXG params for a L."""

    def _HD_IDF_GXG_to_TIF_per_L(DF, L, MdlN, Pa_HD, CRS):
        """Only for use within HD_IDF_GXG_to_TIF - to utilize multiprocessing."""

        # Load HD files corresponding to the L to an XA
        l_IDF_L = list(DF.loc[DF['L'] == L, 'path'])
        XA = imod.idf.open(l_IDF_L)

        # Calculate Variables
        GXG = imod.evaluate.calculate_gxg(XA.squeeze())
        GXG = GXG.rename_vars({var: var.upper() for var in GXG.data_vars})
        N_years_GXG = (
            GXG['N_YEARS_GXG'].values
            if GXG['N_YEARS_GXG'].values.max() != GXG['N_YEARS_GXG'].values.min()
            else int(GXG['N_YEARS_GXG'].values[0, 0])
        )
        N_years_GVG = (
            GXG['N_YEARS_GVG'].values
            if GXG['N_YEARS_GVG'].values.max() != GXG['N_YEARS_GVG'].values.min()
            else int(GXG['N_YEARS_GVG'].values[0, 0])
        )
        GXG['GHG_m_GLG'] = GXG['GHG'] - GXG['GLG']
        GXG = GXG[['GHG', 'GLG', 'GHG_m_GLG', 'GVG']]

        # Save to TIF
        MDs(PJ(M.Pa.PoP_Out_MdlN, 'GXG'), exist_ok=True)
        for V in GXG.data_vars:
            Pa_Out = PJ(M.Pa.PoP_Out_MdlN, 'GXG', f'{V}_L{L}_{MdlN}.tif')

            d_MtDt = {
                f'{V}_L{L}_{MdlN}': {
                    'AVG': float(GXG[V].mean().values),
                    'coordinates': XA.coords,
                    'N_years': N_years_GVG if V == 'GVG' else N_years_GXG,
                    'variable': os.path.splitext(PBN(Pa_Out))[0],
                    'details': f'{MdlN} {V} calculated from (path: {Pa_HD}), via function described in: https://deltares.github.io/imod-python/api/generated/evaluate/imod.evaluate.calculate_gxg.html',
                }
            }

            DA = GXG[V]
            to_TIF(DA, Pa_Out, d_MtDt, CRS=CRS, _print=False)

        return f'L{L} 🟢'

    sprint(Sep)
    sprint(f'*** {MdlN} *** - HD_IDF_GXG_to_TIF\n')

    if CRS is None:
        from WS_Mdl.core.defaults import CRS

    # Get paths
    M = Mdl_N(MdlN)

    # Read DF and apply rules to DF if rules is not None.
    DF = HD_Out_to_DF(M.Pa.Out_HD)
    if rules is not None:
        DF = DF.query(rules)

    if N_cores is None:
        N_cores = max(os.cpu_count() - 2, 1)
    start = DT.now()  # Start time

    with PPE(max_workers=N_cores) as E:
        futures = [E.submit(_HD_IDF_GXG_to_TIF_per_L, DF, L, MdlN, M.Pa.Out_HD, CRS) for L in DF['L'].unique()]
        for f in futures:
            sprint('\t', f.result(), '- Elapsed time (from start):', DT.now() - start)

    sprint('🟢🟢🟢 | Total elapsed:', DT.now() - start)
    sprint(Sep)
