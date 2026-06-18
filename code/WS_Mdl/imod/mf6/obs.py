import re
from pathlib import Path

import numpy as np
import pandas as pd
import WS_Mdl.core.df  # noqa: F401
from shapely.geometry import Point
from WS_Mdl.core.defaults import CRS
from WS_Mdl.core.mdl import Mdl_N
from WS_Mdl.core.style import Sep, sprint
from WS_Mdl.imod.mf6.nam import add_Pkg


def add_GWL_OBS(MdlN: str = None, M: Mdl_N = None, Opt: str = 'BEGIN OPTIONS\nEND OPTIONS'):
    """
    Adds OBS file(s) from PRJ file OBS block to Mdl Sim (which iMOD can't do). Thus the OBS file needs to be written, and then a link to the OBS file needs to be created within the NAM file.
    Assumes OBS IPF file contains the following parameters/columns: 'Id', 'L', 'x', 'y'
    for iMOD5 option check WS_Mdl.utils.MdlN_Pa() description.
    """
    from WS_Mdl.imod.ipf import as_DF
    from WS_Mdl.imod.prj import r_with_OBS

    sprint(Sep)
    sprint('Running add_GWL_OBS ...')
    M = Mdl_N(MdlN) if M is None else M

    # Read PRJ file to extract OBS block info - list of OBS files to be added.
    l_OBS_lines = r_with_OBS(M.Pa.PRJ)[1]
    pattern = r"['\",]([^'\",]*?\.ipf)"  # Regex pattern to extract file paths ending in .ipf
    l_IPF = [
        match.group(1) for line in l_OBS_lines for match in re.finditer(pattern, line)
    ]  # Find all IPF files of the OBS block.

    # Iterate through OBS files of OBS blocks and add them to the Sim
    for i, path in enumerate(l_IPF):
        Pa_OBS_IPF = (M.Pa.PRJ.parent / path).resolve()  # path of IPF file. To be read.
        OBS_IPF_Fi = Pa_OBS_IPF.name  # Filename of OBS file to be added to Sim (to be added without ending)
        if i == 0:
            Pa_OBS = M.Pa.Sim_In / f'{M.MdlN}_GWL.OBS6'  # path of OBS file. To be written.
        else:
            Pa_OBS = M.Pa.Sim_In / f'{M.MdlN}_GWL_N{i}.OBS6'  # path of OBS file. To be written.

        DF_OBS_IPF = as_DF(
            Pa_OBS_IPF
        )  # Get list of OBS items (without temporal dimension, as it's uneccessary for the OBS file, and takes ages to load)
        DF_OBS_IPF_Mdlarea = DF_OBS_IPF.loc[
            ((DF_OBS_IPF['X'] > M.Xmin) & (DF_OBS_IPF['X'] < M.Xmax))
            & ((DF_OBS_IPF['Y'] > M.Ymin) & (DF_OBS_IPF['Y'] < M.Ymax))
        ].copy()  # Slice to OBS within the Mdl area (using INI window)

        DF_OBS_IPF_Mdlarea['C'] = ((DF_OBS_IPF_Mdlarea['X'] - M.Xmin) / M.cellsize).astype(
            np.int32
        ) + 1  # Calculate Cs. Xmin at the origin of the model.
        DF_OBS_IPF_Mdlarea['R'] = (-(DF_OBS_IPF_Mdlarea['Y'] - M.Ymax) / M.cellsize).astype(
            np.int32
        ) + 1  # Calculate Rs. Ymax at the origin of the model.

        DF_OBS_IPF_Mdlarea.sort_values(
            by=['L', 'R', 'C'], ascending=[True, True, True], inplace=True
        )  # Let's sort the DF by L, R, C

        with open(Pa_OBS, 'w') as f:  # write OBS file(s)
            # sprint(M.Pa.MdlN, path, Pa_OBS_IPF, sep='\n')
            f.write(f'# created from {Pa_OBS_IPF}\n')
            f.write(Opt.encode().decode('unicode_escape'))  # write optional block
            f.write(f'\n\nBEGIN CONTINUOUS FILEOUT GWL_OBS_{M.MdlN}({OBS_IPF_Fi.split(".")[0]}).csv\n')

            for _, row in DF_OBS_IPF_Mdlarea.drop_duplicates(subset=['Id', 'L', 'R', 'C']).iterrows():
                f.write(f' {row["Id"]} HEAD {row["L"]} {row["R"]} {row["C"]}\n')

            f.write('END CONTINUOUS\n')

        Pa_OBS_Rel = Path(Pa_OBS).relative_to(M.Pa.NAM_Sim.parent)
        add_Pkg(M.MdlN, f'  OBS6 .\\{Pa_OBS_Rel} GWHD_OBS_Pnt')  # Add to NAM

        sprint(f'🟢 - {Pa_OBS} has been added successfully!')
    sprint(Sep)


def add_within_polygon(
    Pa_Shp: str | Path, MdlN: str, Pkg: str, Opt="""BEGIN OPTIONS\n  DIGITS 4\n  PRINT_INPUT\nEND OPTIONS\n\n"""
):
    """Adds all Pkg elements within a polygon (shapefile) as OBS to the Sim in the steps below:
    -"""
    # Init
    sprint(Sep, verbose_out=False)
    import geopandas as gpd
    from WS_Mdl.imod.mf6.bin import to_DF
    from WS_Mdl.imod.mf6.write import add_OBS_to_MF_In

    from .headers import d_Pkg_Specs

    M = Mdl_N(MdlN)

    pkg_cols = [c[0] for c in d_Pkg_Specs[Pkg.upper()]]

    def _read_pkg_table(
        path: Path, skiprows: int = 0, skipfooter: int = 0
    ) -> pd.DataFrame:  # 666 This can be improved: move it to a new file, so it can be re-used for all MF6 In loading.
        DF = pd.read_csv(
            path,
            sep=r'\s+',
            header=None,
            skiprows=skiprows,
            skipfooter=skipfooter,
            engine='python',
        )

        n_file = DF.shape[1]
        n_base = len(pkg_cols)

        if n_file < n_base:
            raise ValueError(f'{path.name}: expected at least {n_base} cols, found {n_file}')

        extra = [f'aux{i}' for i in range(1, n_file - n_base + 1)]
        DF.columns = pkg_cols + extra
        return DF

    # Load Shp
    if Pa_Shp is not None:
        GDF_Shp = gpd.read_file(Pa_Shp)
        GDF_Shp.crs = CRS
        print(f'Loaded shapefile with {len(GDF_Shp)} features')
        print(f'CRS: {GDF_Shp.crs}')
        print(f'Bounds: {GDF_Shp.bounds}')

    # Load DF
    d = (
        {f.parent.name: {'path': f, 'DF': to_DF(f, Pkg)} for f in M.Pa.Sim_In.rglob(f'{Pkg.lower()}*.bin')}
        if M.V == 'imod_python'
        else {
            f.parents[1].name + '_' + f.parent.name: {
                'path': f,
                'DF': _read_pkg_table(f, skipfooter=12),  # Footer contains dimensions
            }
            for f in M.Pa.Sim_In.rglob(f'{Pkg.lower()}*.arr')
            if not f.open().readline().startswith('# DIMENSIONS')  # Skip empty files (only have dimensions)
        }
    )

    if (
        M.V == 'imod_python' and not d
    ):  # imod_python and no files -> caused by Bin_Ins=False; written as .dat instead. Try to load .dat.
        d = {
            f.parent.name: {
                'path': f,
                'DF': _read_pkg_table(f, skiprows=1),
            }
            for f in M.Pa.Sim_In.rglob(f'{Pkg.lower()}*0.dat')
        }

    if not d:
        raise FileNotFoundError(f'🔴 - No {Pkg} files found  in {M.Pa.Sim_In}')

    for S in d:
        d[S]['DF']['N'] = d[S]['DF']['i'].index + 1
        Sys = re.findall(r'\d+', S)[-1]
        d[S]['DF'] = d[S]['DF'].ws.Calc_XY(Xmin=M.Xmin, Ymax=M.Ymax, cellsize=M.cellsize)

        # Create geometry for DRN points
        d[S]['DF']['geometry'] = d[S]['DF'].apply(lambda row: Point(row['x'], row['y']), axis=1)
        GDF = gpd.GeoDataFrame(d[S]['DF'], crs=CRS)

        # Store init counts before filtering
        # N_init = len(d[S]['DF'])

        # Clip to shapefile
        GDF = gpd.sjoin(GDF, GDF_Shp, how='inner', predicate='within')

        if not GDF.empty:
            # Write
            DF_w = pd.DataFrame()
            DF_w['obsname'] = GDF.apply(lambda row: f'{Pkg}_L{int(row["k"])}_R{int(row["i"])}_C{int(row["j"])}', axis=1)
            DF_w['obstype'] = Pkg
            DF_w['id'] = GDF.apply(lambda row: f'{int(row["k"])} {int(row["i"])} {int(row["j"])}', axis=1)

            Fi = f'{MdlN}.{S}.OBS6'
            with open(M.Pa.Sim_In / Fi, 'w') as f:
                f.write(Opt)
                f.write(f'BEGIN CONTINUOUS FILEOUT {Pkg}_OBS_Sys_{Sys}.CSV\n')
                f.write(DF_w.ws.to_MF_block())
                f.write('END CONTINUOUS FILEOUT\n')

            # Add to MF_In
            text = (
                f' OBS6 FILEIN ./imported_model/{Fi}'
                if M.V == 'imod_python'
                else f' OBS6 FILEIN ./GWF_1/MODELINPUT/{Fi}'
            )
            Pa = (
                M.Pa.Sim_In / f'{S}.{Pkg.lower()}'
                if M.V == 'imod_python'
                else M.Pa.Sim_In / f'{MdlN}_{S.split("_")[-1].upper()}.{Pkg.upper()}6'
            )
            add_OBS_to_MF_In(str_OBS=text, Pa=Pa, iMOD5=False)


def add_L_HD_OBS(MdlN: str, l_L: int, Opt: str = 'BEGIN OPTIONS\n  DIGITS 5\nEND OPTIONS\n\n'):
    """
    Adds HD OBS for all active cells in specified layers based on B .grb file.
    Checks if B domain matches the S domain. If False, stops.
    """
    M = Mdl_N(MdlN)  # Load Mdl_N instance

    from imod.mf6 import read_grb

    ID = read_grb(M.Pa_B.GRB)['idomain']  # Load B domain

    if ID.shape != (M.N_L, M.N_R, M.N_C):  # Ensure dimensions match
        raise ValueError(
            f'B ({M.B}) domain dimensions {ID.shape} do not match {MdlN} {(M.N_L, M.N_R, M.N_C)}. Check if B domain matches S domain.'
        )

    # Make DF from L R C of active cells in selected layers
    ID = (
        ID.sel(layer=l_L)
        .rename({'y': 'R', 'x': 'C', 'layer': 'L'})
        .assign_coords(
            R=('R', ((ID.y[0] - ID.y) / M.cellsize + 1).astype(int).values),
            C=('C', ((ID.x - ID.x[0]) / M.cellsize + 1).astype(int).values),
        )
    )

    LRC = (
        ID.where(ID == 1, drop=True)
        .stack(cell=('L', 'R', 'C'))
        .dropna('cell')
        .cell.to_index()
        .map(lambda x: ' '.join(map(str, x)))
    )

    DF = pd.DataFrame({'obsname': LRC.map(lambda x: 'HD_' + x.replace(' ', '_')), 'obstype': 'HEAD', 'id': LRC})

    Pa_OBS = M.Pa.Sim_In / f'L_HD_OBS_{MdlN}.OBS6'

    with open(Pa_OBS, 'w') as f:
        f.write(f'# created with {M.Pa_B.GRB}\n')
        f.write(Opt)  # write optional block
        f.write(f'BEGIN CONTINUOUS FILEOUT L_HD_OBS_{MdlN}.csv\n')
        f.write(DF.ws.to_MF_block())
        f.write('END CONTINUOUS\n')

    Pa_OBS_Rel = Path(Pa_OBS).relative_to(M.Pa.NAM_Sim.parent)
    add_Pkg(M.MdlN, f'  OBS6 .\\{Pa_OBS_Rel} GWHD_OBS_L')  # Add to NAM
