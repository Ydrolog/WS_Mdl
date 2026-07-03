from pathlib import Path

import xarray as xra

from WS_Mdl.core.style import set_verbose, sprint
from WS_Mdl.xr.convert import to_MBTIF

__all__ = ['Diff_MBTIF', 'Diff_PoP_Param']


def Diff_MBTIF(Pa_TIF1, Pa_TIF2, Pa_TIF_Out=None, verbose=True):
    """
    Calculates the difference between two Multi-band TIF files (TIF1 - TIF2).
    Assumes both files have the same number of bands and dimensions.
    """

    Pa_TIF1, Pa_TIF2 = Path(Pa_TIF1), Path(Pa_TIF2)

    if Pa_TIF_Out is None:
        Pa_TIF_Out = Pa_TIF1.parent / f'{Pa_TIF1.stem}_diff.tif'

    if verbose:
        sprint(f'Calculating difference: {Pa_TIF1.name} - {Pa_TIF2.name}')

    ds1 = xra.open_dataset(Pa_TIF1, engine='rasterio')
    ds2 = xra.open_dataset(Pa_TIF2, engine='rasterio')
    try:
        da1 = ds1['band_data']
        da2 = ds2['band_data']

        if da1.shape != da2.shape:
            raise ValueError(f'Shapes do not match: {da1.shape} vs {da2.shape}')

        da_diff = da1 - da2

        d_MtDt = {f'Diff_Band_{i + 1}': {'description': f'Difference Band {i + 1}'} for i in range(da_diff.shape[0])}

        set_verbose(False)
        try:
            to_MBTIF(da_diff, Pa_TIF_Out, d_MtDt, _print=verbose)
        finally:
            set_verbose(True)

        if verbose:
            sprint(f'🟢 - Difference saved to {Pa_TIF_Out}')
    finally:
        ds1.close()
        ds2.close()


def Diff_PoP_Param(
    MdlN,
    B: str = None,
    Param: str = 'all',
):  # Improve to be able to do all parameters in PoP/Out folder.
    """
    Calcs Diff between TIF files of two MdLNs stored in M.Pa.PoP_Out_MdlN/Param folder. Saves the Diff as a TIF in the same dir as the MdlN PoP_Out files.
    Warning!: It doesn't check if the dates of the PoP_Out of the two MdLNs are the same.
    """
    from WS_Mdl.core.mdl import Mdl_N

    M = Mdl_N(MdlN)

    B = B if B is not None else M.B

    Pa_PoP = M.Pa.PoP_Out_MdlN / Param
    l_Fi = [i for i in Pa_PoP.rglob('*.tif') if MdlN in i.name]
    # print(*l_GXG_Fi, sep='\n')

    if not l_Fi:
        raise FileNotFoundError(f"🔴 - No .tif files found for '{MdlN}' in: {Pa_PoP}")

    for Fi in l_Fi:
        Fi_2 = Pa_PoP / Path(str(Fi).replace(MdlN, B))
        Pa_Out = Fi.parent / Fi.name.replace(MdlN, f'{MdlN}m{"".join(i for i in B if i.isdigit())}')
        # print(Pa_TIF_1, Pa_TIF_2, Pa_Out, end='\n---------------\n', sep='\n')
        try:
            Diff_MBTIF(Fi, Fi_2, Pa_Out)
        except (FileNotFoundError, OSError) as e:
            print(f'🔴 - Missing/unreadable file(s) for:\n{Fi}:\n {e}', end='------------\n')
