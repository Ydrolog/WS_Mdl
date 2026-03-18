from pathlib import Path

import imod
import xarray as xra
from tqdm import tqdm
from WS_Mdl.core import Mdl_N, blue, style_reset
from WS_Mdl.xr.spatial import clip_Mdl_area


def to_XA(DF_meteo, Par, MdlN, clip=False):
    """
    Reads multiple .asc files listed in one of DF_meteo's columns and combines them into a single xarray DataArray.

    Example usage:
    from WS_Mdl.imod.msw.mete_grid import to_DF
    from WS_Mdl.imod.msw.meteo import to_XA
    from WS_Mdl.imod.prj import r_with_OBS

    PRJ = r_with_OBS(M.Pa.PRJ)[0] # [0], cause [1] is the OBS
    DF_P = to_DF(PRJ)
    A_P = to_XA(DF_P, 'P', MdlN)
    """

    M = Mdl_N(MdlN)
    base_dir = Path(M.Pa.PRJ).parent

    Rel_Paths = DF_meteo[Par].astype(str).tolist()
    times = DF_meteo['DT'].to_numpy()

    Abs_Paths = [base_dir / p for p in Rel_Paths]

    L_DA = []
    Y_Desc = None

    for file_path in tqdm(
        Abs_Paths, total=len(Abs_Paths), desc=f' -- {blue}{MdlN}{style_reset}: Loading {Par}', miniters=20
    ):
        DA = imod.rasterio.open(file_path)

        if 'band' in DA.dims:
            DA = DA.squeeze('band', drop=True)

        if Y_Desc is None:
            Y = DA['y'].values
            Y_Desc = (Y.size > 1) and (Y[0] > Y[-1])

        if Y_Desc:
            DA = DA.isel(y=slice(None, None, -1))

        if clip:
            DA = clip_Mdl_area(DA, MdlN)

        L_DA.append(DA)

    A = xra.concat(
        L_DA,
        dim=xra.Variable('time', times),
        coords='minimal',
        compat='override',
        combine_attrs='drop_conflicts',
    )
    return A
