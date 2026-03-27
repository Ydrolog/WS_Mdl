import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import WS_Mdl.core.df  # noqa: F401
from filelock import FileLock as FL
from shapely.geometry import Point
from WS_Mdl.core.defaults import CRS
from WS_Mdl.core.mdl import Mdl_N
from WS_Mdl.core.style import Sep, sprint


def add(MdlN: str, Opt: str = 'BEGIN OPTIONS\nEND OPTIONS', iMOD5=False):
    """
    Adds OBS file(s) from PRJ file OBS block to Mdl Sim (which iMOD can't do). Thus the OBS file needs to be written, and then a link to the OBS file needs to be created within the NAM file.
    Assumes OBS IPF file contains the following parameters/columns: 'Id', 'L', 'X', 'Y'
    for iMOD5 option check WS_Mdl.utils.MdlN_Pa() description.
    """
    from WS_Mdl.core.path import MdlN_PaView
    from WS_Mdl.imod.ipf import as_DF
    from WS_Mdl.imod.prj import r_with_OBS

    sprint(Sep)
    sprint('Running add_OBS ...')
    M = Mdl_N(MdlN)
    Pa = M.Pa if iMOD5 == (M.V == 'imod5') else MdlN_PaView(MdlN, iMOD5=iMOD5)

    # Extract info from INI file.
    d_INI = M.INI
    Xmin, Ymin, Xmax, Ymax = [float(i) for i in d_INI['WINDOW'].split(',')]
    cellsize = float(d_INI['CELLSIZE'])
    # N_R, N_C = int( - (Ymin - Ymax) / cellsize ), int( (Xmax - Xmin) / cellsize ),

    # Read PRJ file to extract OBS block info - list of OBS files to be added.
    l_OBS_lines = r_with_OBS(M.Pa.PRJ)[1]
    pattern = r"['\",]([^'\",]*?\.ipf)"  # Regex pattern to extract file paths ending in .ipf
    l_IPF = [
        match.group(1) for line in l_OBS_lines for match in re.finditer(pattern, line)
    ]  # Find all IPF files of the OBS block.

    # Iterate through OBS files of OBS blocks and add them to the Sim
    for i, path in enumerate(l_IPF):
        Pa_OBS_IPF = (M.Pa.MdlN / path).resolve()  # path of IPF file. To be read.
        OBS_IPF_Fi = Pa_OBS_IPF.name  # Filename of OBS file to be added to Sim (to be added without ending)
        if i == 0:
            Pa_OBS = M.Pa.MdlN / f'GWF_1/MODELINPUT/{MdlN}.OBS6'  # path of OBS file. To be written.
        else:
            Pa_OBS = M.Pa.MdlN / f'GWF_1/MODELINPUT/{MdlN}_N{i}.OBS6'  # path of OBS file. To be written.

        DF_OBS_IPF = as_DF(
            Pa_OBS_IPF
        )  # Get list of OBS items (without temporal dimension, as it's uneccessary for the OBS file, and takes ages to load)
        DF_OBS_IPF_MdlAa = DF_OBS_IPF.loc[
            ((DF_OBS_IPF['X'] > Xmin) & (DF_OBS_IPF['X'] < Xmax))
            & ((DF_OBS_IPF['Y'] > Ymin) & (DF_OBS_IPF['Y'] < Ymax))
        ].copy()  # Slice to OBS within the Mdl Aa (using INI window)

        DF_OBS_IPF_MdlAa['C'] = ((DF_OBS_IPF_MdlAa['X'] - Xmin) / cellsize).astype(
            np.int32
        ) + 1  # Calculate Cs. Xmin at the origin of the model.
        DF_OBS_IPF_MdlAa['R'] = (-(DF_OBS_IPF_MdlAa['Y'] - Ymax) / cellsize).astype(
            np.int32
        ) + 1  # Calculate Rs. Ymax at the origin of the model.

        DF_OBS_IPF_MdlAa.sort_values(
            by=['L', 'R', 'C'], ascending=[True, True, True], inplace=True
        )  # Let's sort the DF by L, R, C

        with open(Pa_OBS, 'w') as f:  # write OBS file(s)
            # sprint(M.Pa.MdlN, path, Pa_OBS_IPF, sep='\n')
            f.write(f'# created from {Pa_OBS_IPF}\n')
            f.write(Opt.encode().decode('unicode_escape'))  # write optional block
            f.write(f'\n\nBEGIN CONTINUOUS FILEOUT OBS_{OBS_IPF_Fi.split(".")[0]}.csv\n')

            for _, row in DF_OBS_IPF_MdlAa.drop_duplicates(subset=['Id', 'L', 'R', 'C']).iterrows():
                f.write(f' {row["Id"]} HEAD {row["L"]} {row["R"]} {row["C"]}\n')

            f.write('END CONTINUOUS\n')

        # Open NAM file and add OBS file to it
        lock = FL(f'{Pa.NAM_Mdl}.lock')  # Create a file lock to prevent concurrent writes
        with lock, open(Pa.NAM_Mdl, 'r+') as f:
            l_NAM = f.read().split('END PACKAGES')
            f.seek(0)
            f.truncate()  # overwrite in-place
            Pa_OBS_Rel = Path(Pa_OBS).relative_to(M.Pa.MdlN)

            f.write(l_NAM[0])
            f.write(rf' OBS6 .\{Pa_OBS_Rel} OBS_{OBS_IPF_Fi.split(".")[0]}')
            f.write('\nEND PACKAGES')

            f.flush()
            os.fsync(f.fileno())  # ensure it’s on disk
            # lock is released automatically when the with-block closes
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

    from .defaults import d_Pkg_Cols

    M = Mdl_N(MdlN)

    # Load Shp
    if Pa_Shp is not None:
        GDF_Shp = gpd.read_file(Pa_Shp)
        print(f'Loaded shapefile with {len(GDF_Shp)} features')
        print(f'CRS: {GDF_Shp.crs}')
        print(f'Bounds: {GDF_Shp.bounds}')
        GDF_Shp.crs = CRS

    # Load DF
    d = (
        {f.parent.name: {'path': f, 'DF': to_DF(f, Pkg)} for f in M.Pa.Sim_In.rglob(f'{Pkg.lower()}*.bin')}
        if M.V == 'imod_python'
        else {
            f.parents[1].name + '_' + f.parent.name: {
                'path': f,
                'DF': pd.read_csv(
                    f,
                    sep=r'\s+',
                    names=d_Pkg_Cols[Pkg.upper()],
                    usecols=range(len(d_Pkg_Cols[Pkg.upper()])),
                    header=None,
                ),
            }
            for f in M.Pa.Sim_In.rglob(f'{Pkg.lower()}*.arr')
        }
    )

    for S in d:
        d[S]['DF']['N'] = d[S]['DF']['i'].index + 1
        Sys = re.findall(r'\d+', S)[-1]
        d[S]['DF'] = d[S]['DF'].ws.Calc_XY(Xmin=M.Xmin, Ymax=M.Ymax, cellsize=M.cellsize)

        # Create geometry for DRN points
        d[S]['DF']['geometry'] = d[S]['DF'].apply(lambda row: Point(row['X'], row['Y']), axis=1)
        GDF = gpd.GeoDataFrame(d[S]['DF'], crs=CRS)

        # Store init counts before filtering
        N_init = len(d[S]['DF'])

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
            text = f' OBS6 FILEIN ./imported_model/{Fi}' if M.V == 'imod_python' else f' OBS6 ./GWF_1/MODELIMPUT/{Fi}'
            Pa = (
                M.Pa.Sim_In / f'{S}.{Pkg.lower()}'
                if M.V == 'imod_python'
                else M.Pa.Sim_In / f'{S.upper()}.{Pkg.upper()}6'
            )
            add_OBS_to_MF_In(str_OBS=text, Pa=Pa, iMOD5=False)
