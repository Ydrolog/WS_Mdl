import math
import os
import tempfile
from pathlib import Path

import geopandas as gpd
import imod
import numpy as np
import pandas as pd
import primod
import xarray as xra
from imod import mf6, msw
from WS_Mdl.core.defaults import CRS
from WS_Mdl.core.mdl import Mdl_N
from WS_Mdl.core.path import MdlN_PaView, Pa_WS
from WS_Mdl.core.runtime import timed_Exe
from WS_Mdl.core.style import Sep, sprint
from WS_Mdl.imod.ini import CeCes, Mdl_Dmns
from WS_Mdl.imod.mf6.solution import moderate_settings
from WS_Mdl.imod.msw.mete_grid import Cvt_to_AbsPa, add_missing_Cols
from WS_Mdl.xr.convert import to_MBTIF as xr_to_MBTIF
from WS_Mdl.xr.convert import to_TIF as xr_to_TIF

xra.set_options(use_new_combine_kwarg_defaults=True)

# import importlib as IL


def r_with_OBS(
    Pa_PRJ, remove_SS=True, season_to_DT=True
):  # 666 Turn this into a class. And add other functions as methods.
    """
    imod.formats.prj.read_projectfile struggles with .prj files that contain OBS blocks. This will read the PRJ file and return a tuple. The first item is a PRJ dictionary (as imod.formats.prj would return) and also a list of the OBS block lines.

    Pa_PRJ: a the path of the PRJ file.
    remove_SS: if True, removes any steady state periods from the PRJ file. This is useful when working with transient models only, as steady state periods can cause issues with some packages (e.g. RIV, DRN) when using imod python.
    season_to_DT: CAUTION!!! this can be set to True so that imod.formats.prj.read_projectfile() doesn't fail when it encounters season names (winter, summer) in the PRJ blocks, but it won't apply them properly (as iMOD5 would do). This function would need to be upgraded for that to happen.
    """
    Pa_PRJ = Path(Pa_PRJ)  # Convert to Path object for easier manipulation
    with open(Pa_PRJ, 'r') as f:
        lines = f.readlines()

    l_filtered_Lns, l_OBS_Lns = [], []
    skip_block = False

    for line in lines:  # Separate OBS block from the rest of the PRJ file
        if '(obs)' in line.lower():  # Start of OBS block
            skip_block = True
            l_OBS_Lns.append(line)  # Keep the header
        elif skip_block and line.strip() == '':  # End of OBS block
            skip_block = False
        elif skip_block:
            l_OBS_Lns.append(line)  # Store OBS content
        else:
            l_filtered_Lns.append(line)  # Keep everything else

    get_SS_N, N_SS, l_filtered_Lns1 = False, 0, []
    if remove_SS:
        for i, Ln in enumerate(l_filtered_Lns[:]):  # Iterate over a copy of the list to allow removal
            if 'STEADY-STATE' in Ln.upper():  # Identify steady state period lines
                get_SS_N = True
                LL = l_filtered_Lns1[-1].split(',')  # LL: Last Line
                l_filtered_Lns1[-1] = ','.join([f'{int(LL[0]) - 1:03}'] + LL[1:])
            elif get_SS_N:
                l_SS_Ln = Ln.replace('\n', '').split(',')
                N_SS = math.prod([int(x) for x in l_SS_Ln])
                get_SS_N = False
            elif N_SS > 0:
                N_SS -= 1
            else:
                l_filtered_Lns1.append(Ln)  # Keep non-steady state lines

    # Replace seasons
    l_filtered_Lns2 = []
    if season_to_DT:
        season_map = {}
        record_periods = False
        for i, Ln in enumerate(l_filtered_Lns1[:]):  # Iterate over a copy of the list to allow modification
            if 'periods' in Ln.lower():  # Identify season lines
                record_periods = True
                # l_filtered_Lns.pop(i)
            elif record_periods and Ln.strip() == '':  # End of season block
                record_periods = False
            elif record_periods:
                if 'winter\n' == Ln.lower():
                    date_str = l_filtered_Lns1[i + 1].strip()
                    try:
                        dt = pd.to_datetime(date_str, dayfirst=True)
                        season_map[Ln] = dt.strftime('%Y-%m-%d %H:%M:%S') + '\n'
                    except Exception:
                        season_map[Ln] = date_str + '\n'
                elif 'summer\n' == Ln.lower():
                    date_str = l_filtered_Lns1[i + 1].strip()
                    try:
                        dt = pd.to_datetime(date_str, dayfirst=True)
                        season_map[Ln] = dt.strftime('%Y-%m-%d %H:%M:%S') + '\n'
                    except Exception:
                        season_map[Ln] = date_str + '\n'
        l_filtered_Lns2 = [season_map.get(x, x) for x in l_filtered_Lns1]

    # Write to a temporary file
    l_filtered_Lns = l_filtered_Lns2
    # Create the temp file in the same directory as the original PRJ so relative paths resolve correctly
    with tempfile.NamedTemporaryFile(delete=False, mode='w', dir=Pa_PRJ.parent, suffix='.prj') as temp_file:
        temp_file.writelines(l_filtered_Lns)
        Pa_PRJ_temp = temp_file.name

    try:
        PRJ = imod.formats.prj.read_projectfile(Pa_PRJ_temp)  # Load the PRJ file without OBS
    except Exception as e:
        sprint(f'Error reading PRJ file: {e}')
        PRJ = None
    Path(Pa_PRJ_temp).unlink()  # Delete temp PRJ file as it's not needed anymore.

    return PRJ, l_OBS_Lns


def to_DF(MdlN):
    """Leverages r_PRJ_with_OBS to produce a DF with the PRJ data.
    Could have been included in utils.py based on dependencies, but utils_imod.py fits it better as it's almost always used after r_PRJ_with_OBS (so the libs will be already loaded)."""

    M = Mdl_N(MdlN)
    Pa = M.Pa

    Pa_AppData = Path(os.getenv('APPDATA')).parent
    t_Pa_replace = (
        str(Pa_AppData),
        str(Pa_WS / 'models' / M.alias),
    )  # For some reason imod.idf.read reads the path incorrectly, so I have to replace the incorrect part.

    d_PRJ, OBS = r_with_OBS(Pa.PRJ)

    columns = [
        'package',
        'parameter',
        'time',
        'active',
        'is_constant',
        'layer',
        'factor',
        'addition',
        'constant',
        'path',
    ]
    DF = pd.DataFrame(columns=columns)  # Main DF to store all the packages

    sprint(' --- Reading PRJ Packages into DF ---')
    for Pkg_name in list(d_PRJ.keys()):  # Iterate over packages
        sprint(f'\t{Pkg_name:<7}\t...\t', end='')
        try:
            Pkg = d_PRJ[Pkg_name]

            if int(Pkg['active']):  # if the package is active, process it
                l_Par = [
                    k for k in Pkg.keys() if k not in {'active', 'n_system', 'time'}
                ]  # Make list from package keys/parameters
                for Par in l_Par[:]:  # Iterate over parameters
                    for N, L in enumerate(Pkg[Par]):  # differentiate between packages (have time) and modules.
                        Ln_DF_path = {
                            **L,
                            'package': Pkg_name,
                            'parameter': Par,
                        }  # , 'Pa_type':L['path'].suffix.lower()} #, "metadata": L}

                        if 'time' in d_PRJ[Pkg_name].keys():
                            if Pkg['n_system'] > 1:
                                DF.loc[f'{Pkg_name.upper()}_{Par}_Sys{(N) % Pkg["n_system"] + 1}_{L["time"]}'] = (
                                    Ln_DF_path
                                )
                            elif Pkg['n_system'] == 1:
                                DF.loc[f'{Pkg_name.upper()}_{Par}'] = Ln_DF_path
                        else:
                            if Pkg['n_system'] > 1:
                                DF.loc[f'{Pkg_name.upper()}_{Par}_Sys{(N) % Pkg["n_system"] + 1}'] = Ln_DF_path
                            elif Pkg['n_system'] == 1:
                                DF.loc[f'{Pkg_name.upper()}_{Par}'] = Ln_DF_path
                sprint('🟢')
            else:
                sprint('\u2012 the package is innactive.')
        except Exception as e:
            DF.loc[f'{Pkg_name.upper()}'] = '-'
            DF.loc[f'{Pkg_name.upper()}', 'active'] = f'Failed to read package: {e}'
            sprint('🟡')
    sprint('🟢🟢')
    sprint(f' {"-" * 100}')

    DF['package'] = DF['package'].str.replace('(', '').str.replace(')', '').str.upper()
    DF['suffix'] = DF['path'].apply(
        lambda x: x.suffix.lower() if hasattr(x, 'suffix') else '-'
    )  # Check if 'suffix' exists # Make suffix column so that paths can be categorized
    DF['path'] = DF['path'].astype(
        'string'
    )  # Convert path to string so that the wrong part of the path can be .replace()ed
    DF['MdlN'] = DF['path'].str.split('_').str[-1].str.split('.').str[0]
    DF['path'] = DF['path'].str.replace(
        *t_Pa_replace, regex=False
    )  # Replace incorrect part of paths. I'm not sure why iMOD doesn't read them right. Maybe cause they're relative it's assumed they start form a directory which is incorrect.
    DF = DF.loc[
        :, list(DF.columns[:2]) + ['MdlN'] + list(DF.columns[2:-3]) + ['suffix', DF.columns[-3]]
    ]  # Rearrange columns

    return DF


def o_with_OBS(Pa_PRJ, return_OBS=False):
    """
    - Reads PRJ file with OBS block.
    - Returns PRJ file (as imod.formats.prj would), or tuple with PRJ and OBS lines if return_OBS.
    - Why? Because imod.formats.prj.read_projectfile fails on PRJ files that contain OBS blocks.
    - This works for both types of PRJ files - with and without OBS blocks. So it's SAFER to use overall.
    """

    Dir_PRJ = Path(Pa_PRJ).parent  # Directory of the PRJ file

    with open(Pa_PRJ, 'r') as f:
        lines = f.readlines()

    l_filtered_Lns, l_OBS_Lns = [], []
    skip_block = False

    for line in lines:
        if '(obs)' in line.lower():  # Start of OBS block
            skip_block = True
            l_OBS_Lns.append(line)  # Keep the header
        elif skip_block and line.strip() == '':  # End of OBS block
            skip_block = False
        elif skip_block:
            l_OBS_Lns.append(line)  # Store OBS content
        else:
            l_filtered_Lns.append(line)  # Keep everything else

    # Write to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.prj.tmp', dir=Dir_PRJ) as temp_file:
        temp_file.writelines(l_filtered_Lns)
        Pa_PRJ_temp = temp_file.name

    PRJ = imod.prj.open_projectfile_data(Pa_PRJ_temp)  # Load the PRJ file without OBS
    Path(Pa_PRJ_temp).unlink()  # Delete temp PRJ file as it's not needed anymore.

    sprint(f'🟢🟢 - PRJ loaded from {Pa_PRJ}')
    if return_OBS:
        return PRJ, l_OBS_Lns
    else:
        return PRJ


def regrid(PRJ, MdlN: str = None, x_CeCes=None, y_CeCes=None, method='linear'):
    """
    Regrid all spatial data in PRJ to target discretization.
    If x_CeCes and y_CeCes are provided, they will be used as the target grid.
    If MdlN is provided, it will use the model's spatial dimensions from the INI file.
    """

    if x_CeCes is not None and y_CeCes is not None:
        dx = x_CeCes[1] - x_CeCes[0] if len(x_CeCes) > 1 else 0
        dy = y_CeCes[1] - y_CeCes[0] if len(y_CeCes) > 1 else 0

    elif MdlN:  # If MdlN is provided, get the model's spatial dimensions from INI file
        x_CeCes, y_CeCes = CeCes(MdlN)
        dx = x_CeCes[1] - x_CeCes[0] if len(x_CeCes) > 1 else 0
        dy = y_CeCes[1] - y_CeCes[0] if len(y_CeCes) > 1 else 0

    else:
        sprint('🔴🔴🔴 - Either MdlN or x_CeCes and y_CeCes must be provided. Cancelling regridding...')
        return  # Stop regridding if no valid grid is provided

    sprint(f'\nTarget grid: {len(x_CeCes)}x{len(y_CeCes)} cells at {dx:.1f}x{dy:.1f} m resolution')
    sprint(
        f'\nTarget extents: X=[{x_CeCes.min():.1f}, {x_CeCes.max():.1f}], Y=[{y_CeCes.min():.1f}, {y_CeCes.max():.1f}]'
    )

    PRJ_regridded = {}

    for key, data in PRJ.items():
        sprint(f'Processing {key}...')

        if isinstance(data, dict):
            # Handle nested dictionaries (like 'cap', 'bnd')
            PRJ_regridded[key] = {}
            for sub_key, sub_data in data.items():
                PRJ_regridded[key][sub_key] = regrid_DA(sub_data, x_CeCes, y_CeCes, dx, dy, f'{key}.{sub_key}', method)
        else:
            # Handle top-level data
            PRJ_regridded[key] = regrid_DA(data, x_CeCes, y_CeCes, dx, dy, key, method)

    sprint('🟢🟢🟢 - PRJ has been regridded successfully!')
    return PRJ_regridded


def regrid_DA(DA, x_CeCes, y_CeCes, dx, dy, item_name, method='linear'):
    """Handle regridding of individual DA items"""

    item_name_lower = item_name.lower()

    # Skip if not xarray or no spatial dimensions
    if not hasattr(DA, 'dims') or not ('x' in DA.dims and 'y' in DA.dims):
        sprint(f'  {item_name}: ⚪️ - No spatial dims - keeping original')
        return DA

    # Check if DA is already on target grid before doing anything
    DA_x, DA_y = DA.x.values, DA.y.values
    if len(DA_x) > 1 and len(DA_y) > 1:
        DA_dx = DA_x[1] - DA_x[0]
        DA_dy = abs(DA_y[1] - DA_y[0])

        # Compare with tolerance
        same_resolution = np.isclose(DA_dx, dx, atol=1e-6) and np.isclose(DA_dy, abs(dy), atol=1e-6)
        same_size = len(DA_x) == len(x_CeCes) and len(DA_y) == len(y_CeCes)

        # Check if coordinates match (within tolerance)
        coords_match = False
        if same_size:
            try:
                coords_match = np.allclose(DA_x, x_CeCes, atol=1e-6) and np.allclose(DA_y, y_CeCes, atol=1e-6)
            except Exception:
                coords_match = False

        if same_resolution and same_size and coords_match:
            sprint(f'  {item_name}: ⚫️ - Already on target grid')
            return DA

    # Handle special DA types
    if ('riv' in item_name_lower) or ('drn' in item_name_lower):
        method = 'nearest'  # Sparse BC rasters: preserve support; linear can erode NaN-heavy masks.
    elif 'ibound' in item_name_lower:
        method = 'nearest'  # Use nearest neighbor for boundary conditions. They're 1 or 0, so we don't want to be messing with decimals.
    elif any(x in item_name_lower for x in ['landuse', 'soil_unit', 'zone']):
        method = 'nearest'  # Use nearest neighbor for categorical DA
    elif 'area' in item_name_lower:
        # Special handling for area fields - scale by grid ratio
        regridded = DA.interp(x=x_CeCes, y=y_CeCes, method='linear')
        grid_ratio = (len(x_CeCes) * len(y_CeCes)) / (len(DA.x) * len(DA.y))
        regridded = regridded * grid_ratio

        # Attach dx and dy attributes to the regridded DataArray
        regridded = regridded.assign_coords(dx=dx, dy=dy)
        sprint(f'  {item_name}: 🟢 - Area field regridded with grid ratio scaling')
        return regridded

    # Option A: Interpolate first, then clip to target bounds
    # This is simpler but more computationally expensive for large arrays
    try:
        regridded = DA.interp(x=x_CeCes, y=y_CeCes, method=method)

        # Attach dx and dy attributes to the regridded DataArray
        regridded = regridded.assign_coords(dx=dx, dy=dy)
        sprint(f'  {item_name}: 🟢 - {DA.sizes} -> {regridded.sizes}. Method: {method}.')
        return regridded
    except Exception as e:
        sprint(f'  {item_name}: 🔴 - Regridding failed ({e}) - keeping original')
        return DA


def to_TIF(MdlN, iMOD5=False):
    """Converts PRJ file to TIF (multiband if necessary) files by package (only time independent packages).
    The function uses a DF produced by PRJ_to_DF. It needs to follow a specific format.
    Also creates a .csv file with the TIF file paths to be replaced in the QGIS project."""

    # -------------------- Initiate ----------------------------------------------
    M = Mdl_N(MdlN)
    Pa = M.Pa if iMOD5 == (M.V == 'imod5') else MdlN_PaView(MdlN, iMOD5=iMOD5)
    M_alias = M.alias
    Xmin, Ymin, Xmax, Ymax, cellsize, N_R, N_C = Mdl_Dmns(Pa.INI)  # Get dimensions

    DF = to_DF(MdlN)  # Read PRJ file to DF

    # -------------------- Process time-indepenent packages (most) ---------------
    sprint('\n --- Converting time-independant package IDF files to TIF ---')
    DF_Rgu = DF[
        (DF['time'].isna())  # Only keep regular (time independent) packages
        & (DF['path'].notna())
        & (DF['suffix'] == '.idf')
    ]  # Non time packages have NaN in 'time' Fld. Failed packages have '-', so they'll also be excluded.

    for i, Par in enumerate(DF_Rgu['parameter'].unique()[:]):  # Iterate over parameters
        sprint(f'\t{i:<2}, {Par:<30} ... ', end='')

        try:
            DF_Par = DF_Rgu[DF_Rgu['parameter'] == Par]  # Slice DF_Rgu for current parameter.
            DF_Par = DF_Par.drop_duplicates(
                subset='path', keep='first'
            )  # Drop duplicates, keep the first one. imod.formats.idf.open will do that with the list of paths anyway, so the only way to match the paths to the correct metadata is to have only one path per metadata.
            if DF_Par['package'].nunique() > 1:
                sprint('There are multiple packages for the same parameter. Check DF_Rg')
                break
            else:
                Pkg = DF_Par['package'].iloc[0]  # Get the package name

            ## Prepare directoreis and filenames
            Pkg_MdlN = M_alias + str(DF_Par['MdlN'].str.extract(r'(\d+)').astype(int).max().values[0])
            Pa_TIF = Pa.Pa_Mdl / 'PoP' / 'In' / Pkg / Pkg_MdlN / f'{Pkg}_{Par}_{Pkg_MdlN}.tif'  # Full path to TIF file

            ## Build a dictionary mapping each band’s name to its row’s metadata. We're assuming that the order the paths are read into DA is the same as the order in DF_Par.
            d_MtDt = {}

            if Pa_TIF.exists():
                sprint(f'🔴 - {Pa_TIF.name} already exists. Skipping.')
                continue
            else:
                Pa_TIF.parent.mkdir(parents=True, exist_ok=True)  # Make sure the directory exists

                ## Read files-paths to xarray Data Array (DA), then write them to TIF file(s).
                if DF_Par.shape[0] > 1:  # If there are multiple paths for the same parameter
                    for i, R in DF_Par.iterrows():
                        d_MtDt[f'{R["parameter"]}_L{R["layer"]}_{R["MdlN"]}'] = {
                            ('origin_path' if col == 'path' else col): str(val) for col, val in R.items()
                        }
                    DA = imod.formats.idf.open(list(DF_Par['path']), pattern='{name}_L{layer}_').sel(
                        x=slice(Xmin, Xmax), y=slice(Ymax, Ymin)
                    )
                    xr_to_MBTIF(DA, Pa_TIF, d_MtDt)
                    sprint('🟢 - multi-band')
                else:
                    try:
                        DA = imod.formats.idf.open(list(DF_Par['path']), pattern='{name}_L{layer}_').sel(
                            x=slice(Xmin, Xmax), y=slice(Ymax, Ymin)
                        )
                        d_MtDt[
                            f'{DF_Par["parameter"].values[0]}_L{DF_Par["layer"].values[0]}_{DF_Par["MdlN"].values[0]}'
                        ] = {('origin_path' if col == 'path' else col): str(val) for col, val in R.items()}
                        xr_to_TIF(
                            DA.squeeze(drop=True), Pa_TIF, d_MtDt
                        )  # .squeeze cause 2D arrays have extra dimension with size 1 sometimes.
                        sprint('🟢 - single-band with L attribute')
                    except Exception:
                        DA = imod.formats.idf.open(list(DF_Par['path']), pattern='{name}_').sel(
                            x=slice(Xmin, Xmax), y=slice(Ymax, Ymin)
                        )
                        d_MtDt[f'{DF_Par["parameter"].values[0]}_{DF_Par["MdlN"].values[0]}'] = {
                            ('origin_path' if col == 'path' else col): str(val) for col, val in R.items()
                        }
                        xr_to_TIF(
                            DA.squeeze(drop=True), Pa_TIF, d_MtDt
                        )  # .squeeze cause 2D arrays have extra dimension with size 1 sometimes.
                        sprint('🟢 - single-band without L attribute')
        except Exception as e:
            sprint(f'🔴 - Error: {e}')

    # ------------- Process time-dependent packages (RIV, DRN, WEL) ---------------------
    ## RIV & DRN
    sprint('\n --- Converting time dependant packages ---')
    DF_time = DF[
        (DF['time'].notna()) & (DF['time'] != '-') & (DF['path'].notna())
    ]  # Non time packages have NaN in 'time' Fld. Failed packages have '-', so they'll also be excluded.

    for i, R in DF_time[DF_time['package'].isin(('DRN', 'RIV'))].iterrows():
        sprint(f'\t{f"{R['package']}_{R['parameter']}":<30} ... ', end='')

        Pa_TIF = (
            Pa.Pa_Mdl / 'PoP' / 'In' / R['package'] / R['MdlN'] / Path(R['path']).with_suffix('.tif').name
        )  # Full path to TIF file

        if Pa_TIF.exists():
            sprint(f'🔴 - {Pa_TIF.name} already exists. Skipping.')
            continue
        else:
            try:
                Pa_TIF.parent.mkdir(parents=True, exist_ok=True)  # Make sure the directory exists

                ## Build a dictionary mapping each band’s name to its row’s metadata.
                d_MtDt = {
                    f'{R["parameter"]}_L{R["layer"]}_{R["MdlN"]}': {
                        ('origin_path' if col == 'path' else col): str(val) for col, val in R.items()
                    }
                }

                DA = imod.formats.idf.open(R['path'], pattern=f'{{name}}_{M_alias}').sel(
                    x=slice(Xmin, Xmax), y=slice(Ymax, Ymin)
                )
                xr_to_TIF(
                    DA.squeeze(drop=True), Pa_TIF, d_MtDt
                )  # .squeeze cause 2D arrays have extra dimension with size 1 sometimes.
                sprint('🟢 - IDF converted to TIF - single-band without L attribute')
            except Exception as e:
                sprint(f'🔴 - Error: {e}')

    ## WEL
    DF_WEL = DF.loc[DF['package'] == 'WEL']

    for i, R in DF_WEL.iloc[3:6].iterrows():
        sprint(f'\t{R["path"].name:<30} ... ', end='')

        Pa_GPKG = (
            Pa.Pa_Mdl / 'PoP' / 'In' / R['package'] / R['MdlN'] / Path(R['path']).with_suffix('.gpkg').name
        )  # Full path to TIF file

        if Pa_GPKG.exists():
            sprint(f'🔴 - file {Pa_GPKG.name} exists. Skipping.')
            continue
        else:
            try:
                DF_IPF = imod.formats.ipf.read(R['path'])
                DF_IPF = DF_IPF.loc[
                    ((DF_IPF['x'] > Xmin) & (DF_IPF['x'] < Xmax)) & ((DF_IPF['y'] > Ymin) & (DF_IPF['y'] < Ymax))
                ].copy()  # Slice to OBS within the Mdl Aa

                if ('q_m3' in DF_IPF.columns) and ('id' not in DF_IPF.columns):
                    DF_IPF.rename(
                        columns={'q_m3': 'id'}, inplace=True
                    )  # One of the IPF files has q_m3 instead of id in it's fields. Don't ask me why, but it has to be dealt with.

                # 666 I'll only save the average flow now
                DF_IPF_AVG = DF_IPF.groupby('id')[DF_IPF.select_dtypes(include=np.number).columns].agg(np.mean)
                _GDF_AVG = gpd.GeoDataFrame(
                    DF_IPF_AVG, geometry=gpd.points_from_xy(DF_IPF_AVG['x'], DF_IPF_AVG['y'])
                ).set_CRS(CRS=CRS)

                Pa_GPKG.parent.mkdir(parents=True, exist_ok=True)  # Make sure the directory exists
                _GDF_AVG.to_file(Pa_GPKG, driver='GPKG')  # , layer=PBN(Pa_GPKG))
                sprint('🟢 - IPF average values (per id) converted to GPKG')
            except Exception as e:
                sprint(f'🔴 - Error: {e}')
    # -------------------- Process derived packages/parameters (Thk, T) -----------------
    d_Clc_In = {}  # Dictionary to store calculated inputs.

    ## Thk. TOP and BOT files have been QA'd in C:\OD\WS_Mdl\code\PrP\Mdl_In_to_MM\Mdl_In_to_MM.ipynb
    sprint(' --- Converting calculated inputs to TIF ---')

    toP = imod.formats.idf.open(list(DF_Rgu[DF_Rgu['parameter'] == 'top']['path']), pattern='{name}_L{layer}_').sel(
        x=slice(Xmin, Xmax), y=slice(Ymax, Ymin)
    )
    DA_BOT = imod.formats.idf.open(
        list(DF_Rgu[DF_Rgu['parameter'] == 'bottom']['path']), pattern='{name}_L{layer}_'
    ).sel(x=slice(Xmin, Xmax), y=slice(Ymax, Ymin))
    DA_Kh = imod.formats.idf.open(list(DF_Rgu[DF_Rgu['parameter'] == 'kh']['path']), pattern='{name}_L{layer}_').sel(
        x=slice(Xmin, Xmax), y=slice(Ymax, Ymin)
    )

    DA_Thk = (toP - DA_BOT).squeeze(drop=True)  # Let's make a dictionary to store Info about each parameter
    MdlN_Pkg = M.alias + str(
        max(DF_Rgu.loc[DF_Rgu['package'].isin(['TOP', 'BOT']), 'MdlN'].str.extract(r'(\d+)')[0])
    )  # 666 the largest number from the TOP and BOT MdlNs
    d_Clc_In['Thk'] = {
        'Par': 'thickness',
        'DA': DA_Thk,
        'MdlN_Pkg': MdlN_Pkg,
        'MtDt': {
            **{
                f'thickness_L{i + 1}_{MdlN_Pkg}': {'layer': f'L{i + 1}'} for i in range(DA_Thk.shape[0])
            },  # Per-layer metadata
            'all': {
                'description': "Layer thickness calculated as 'top - bottom' per layer.",
                'source_files': f"""{'-' * 200}\nTOP: {' ' * 30} {' | | '.join(DF_Rgu.loc[DF_Rgu['package'] == 'TOP', 'path'])}        {'-' * 200}\nBOT: {' ' * 30} {' | | '.join(DF_Rgu.loc[DF_Rgu['package'] == 'BOT', 'path'])} """,
            },
        },
    }
    ## T
    DA_T = DA_Thk * DA_Kh
    MdlN_Pkg = M.alias + str(
        max(DF_Rgu.loc[DF_Rgu['package'].isin(['TOP', 'BOT', 'NPF']), 'MdlN'].str.extract(r'(\d+)')[0])
    )  # the largest number from the TOP and BOT MdlNs
    d_Clc_In['T'] = {
        'Par': 'transmissivity',
        'DA': DA_T,
        'MdlN_Pkg': MdlN_Pkg,
        'MtDt': {
            **{
                f'transmissivity_L{i + 1}_{MdlN_Pkg}': {'layer': f'L{i + 1}'} for i in range(DA_Thk.shape[0])
            },  # Per-layer metadata
            'all': {
                'description': "Layer transmissivity (horizontal) calculated as '(top - bottom)*Kh' per layer.",
                'source_files': f"""{'-' * 200}TOP: {' ' * 30} {' | | '.join(DF_Rgu.loc[DF_Rgu['package'] == 'TOP', 'path'])} 
                    {'-' * 200}BOT: {' ' * 30} {' | | '.join(DF_Rgu.loc[DF_Rgu['package'] == 'BOT', 'path'])}
                    {'-' * 200}NPF: {' ' * 30} {' | | '.join(DF_Rgu.loc[DF_Rgu['package'] == 'NPF', 'path'])}""",
            },
        },
    }

    for i, Par in enumerate(d_Clc_In.keys()):
        sprint(f'\t{d_Clc_In[Par]["Par"]:<30} ... ', end='')

        Pa_TIF = (
            M.Pa.Pa_Mdl / 'PoP' / 'Clc_In' / Par / d_Clc_In[Par]['MdlN_Pkg'] / f'{Par}_{d_Clc_In[Par]["MdlN_Pkg"]}.tif'
        )  # Full path to TIF file #666 need to think which MdlN to use. It's hard to do the same as with the other packages.

        if Pa_TIF.exists():
            sprint(f'🔴 - {Pa_TIF.name} already exists. Skipping.')
            continue
        else:
            try:
                Pa_TIF.parent.mkdir(parents=True, exist_ok=True)  # Make sure the directory exists

                ## Write DAs to TIF files.
                DA = d_Clc_In[Par]['DA'].squeeze(drop=True)
                d_MtDt = d_Clc_In[Par]['MtDt']

                if not DA.rio.CRS:  # Ensure DA_Thk has a CRS (if missing, set it)
                    DA.rio.write_CRS(CRS, inplace=True)  # Replace with correct CRS

                match len(DA.shape):
                    case 3:
                        xr_to_MBTIF(DA, Pa_TIF, d_MtDt)  # If there are multiple paths for the same parameter
                        sprint('🟢 - multi-band')
                    case 2:
                        xr_to_TIF(
                            DA.squeeze(drop=True), Pa_TIF, d_MtDt
                        )  # .squeeze cause 2D arrays have extra dimension with size 1 sometimes.
                        sprint('🟢 - single-band')
                    case _:
                        raise ValueError(f'Unexpected array rank: {DA.ndim}')

            except Exception as e:
                sprint(f'🔴 - Error: {e}')
    sprint(Sep)


def PrSimP(
    M: Mdl_N,
):
    """
    Prepares Sim Ins for Mdl_N following the process below:
    1. Load and regrid PRJ file to target grid (if necessary).
    2. Load MF6 Simulation from regridded PRJ.
    3. Load MSW Simulation from regridded PRJ and MF6 DIS.
    4. Clip both simulations to AoI.
    5. Create mask from current regridded model (not the old one).
    6. Clean up MF6 packages (remove unused packages, reformat BC packages).
    7. Write Sim In files (MF6 and MSW) to disk.

    Returns Sim_MF6 & MSW_Mdl for potential downstream processes.

    Requires M.verbose, Pa.MF6_DLL, M.Pa.MSW_DLL to be imbued to the Mdl_N object/class. Those are not standard Mdl_N properties.
    """

    # ----- Load and regrid PRJ -----
    PRJ_ = timed_Exe(
        o_with_OBS, M.Pa.PRJ, pre=f'  - Loading {M.Pa.PRJ.name} ...', verbose_in=True, verbose_out=M.verbose
    )
    PRJ, period_data = PRJ_[0], PRJ_[1]

    sprint('  - Regridding PRJ ...', end='', verbose_in=True, verbose_out=M.verbose, set_time=True)
    PRJ_regrid = timed_Exe(regrid, PRJ, M.MdlN)  # Speeds up Mdl load.

    # Filter period_data to avoid DTypePromotionError when repeat labels don't overlap simulation time
    # This happens when imod.util.expand_repetitions returns an empty dict for a repeat.
    # In NumPy 2.x, comparing float64(0.0) with datetime64[ns] raises DTypePromotionError.
    sprint('  - Filtering period_data ...', end='', verbose_in=True, verbose_out=M.verbose)
    from imod.util.expand_repetitions import expand_repetitions
    
    T_min = pd.Timestamp(M.INI.SDATE)
    T_max = pd.Timestamp(M.INI.EDATE)
    
    filtered_period_data = {}
    for pkg_name, pkg_data in period_data.items():
        if not isinstance(pkg_data, dict):
            filtered_period_data[pkg_name] = pkg_data
            continue
            
        filtered_pkg_data = {}
        for repeat_key, repeat_val in pkg_data.items():
            # expand_repetitions returns a dict of {new_date: old_date}
            # If the dict is empty, this repetition is not active during simulation time.
            # We filter it out early to prevent iMOD-Python from creating empty packages
            # that might trigger NumPy 2.x DType promotion errors.
            try:
                repetitions = expand_repetitions(repeat_val, T_min, T_max)
                if repetitions:
                    filtered_pkg_data[repeat_key] = repeat_val
            except Exception:
                # If expansion fails for any reason, keep the original to be safe
                filtered_pkg_data[repeat_key] = repeat_val
        filtered_period_data[pkg_name] = filtered_pkg_data
    period_data = filtered_period_data
    sprint('🟢', verbose_in=True, verbose_out=M.verbose)

    # Set outer boundaries to -1. Otherwise CHD won't be loaded properly.
    BND = PRJ_regrid['bnd']['ibound']
    BND.loc[:, [BND.y[0], BND.y[-1]], :] = -1  # Top and bottom rows
    BND.loc[:, :, [BND.x[0], BND.x[-1]]] = -1  # Left and right columns
    sprint('🟢', verbose_in=True, verbose_out=M.verbose, print_time=True)

    # ----- Load MF6 Simulation
    times = pd.date_range(M.SP_1st, M.SP_last, freq='D')
    Sim_MF6 = timed_Exe(
        mf6.Modflow6Simulation.from_imod5_data,
        PRJ_regrid,
        period_data,
        times,
        pre='  - Loading MF6 Simulation ...',
        post='🟢',
        verbose_in=True,
        verbose_out=M.verbose,
    )
    # Sim_MF6[f'{M.MdlN}'] = Sim_MF6.pop('imported_model')  # Rename imported_model to MdlN. #666

    # Pass the Sim components to objects.
    MF6_Mdl = Sim_MF6['imported_model']
    MF6_Mdl['oc'] = mf6.OutputControl(save_head='last', save_budget='last')
    Sim_MF6['ims'] = moderate_settings()  # Mimic iMOD5's "Moderate" settings.
    MF6_DIS = MF6_Mdl['dis']

    # ----- Load MSW Simulation
    PRJ_MSW = {'cap': PRJ_regrid.copy()['cap'], 'extra': PRJ_regrid.copy()['extra']}  # Isolate MSW keys from PRJ.
    PRJ_MSW['extra']['paths'][2][0] = Cvt_to_AbsPa(M.Pa.PRJ, PRJ)  # mete_grid.inp relative paths fix
    MSW_Mdl = timed_Exe(
        msw.MetaSwapModel.from_imod5_data,
        PRJ_MSW,
        MF6_DIS,
        times,
        pre='  - Loading MSW Simulation ...',
        post='🟢',
        verbose_in=True,
        verbose_out=M.verbose,
    )

    # ----- Clip models
    sprint('  - Clipping models ...', end='', verbose_in=True, verbose_out=M.verbose, set_time=True)
    Sim_MF6_AoI = timed_Exe(Sim_MF6.clip_box, x_min=M.Xmin, x_max=M.Xmax, y_min=M.Ymin, y_max=M.Ymax, pre='')
    MF6_Mdl_AoI = Sim_MF6_AoI['imported_model']
    MSW_Mdl_AoI = timed_Exe(MSW_Mdl.clip_box, x_min=M.Xmin, x_max=M.Xmax, y_min=M.Ymin, y_max=M.Ymax, pre='')
    # clip_box doesn't clip the packages clipped with regrid, but it clips non raster-like packages like WEL and removes packages that are not in the AoI.
    sprint('🟢', verbose_in=True, verbose_out=M.verbose, print_time=True)

    # ----- Load models into memory
    sprint('  - Loading models into memory ...', end='', verbose_in=True, verbose_out=M.verbose, set_time=True)
    for pkg in MF6_Mdl_AoI.values():
        if 'layer' in pkg.dataset.coords and pkg.dataset['layer'].ndim == 1:
            pkg.dataset = pkg.dataset.sortby('layer')
        pkg.dataset.load()
    for pkg in MSW_Mdl_AoI.values():
        pkg.dataset.load()
    sprint('🟢', verbose_in=True, verbose_out=M.verbose, print_time=True)

    # ----- Create mask from current regridded model (not the old one)
    sprint('  - Creating mask ...', end='', verbose_in=True, verbose_out=M.verbose, set_time=True)
    mask = MF6_Mdl_AoI.domain
    # 666 mask needs to be checked and potentially updated with -1 values at the edge of the Mdl Aa.
    Sim_MF6_AoI.mask_all_models(mask)
    DIS_AoI = MF6_Mdl_AoI['dis']
    sprint('🟢', verbose_in=True, verbose_out=M.verbose, print_time=True)

    # ----- MF6 cleanup
    sprint('  - Cleaning up MF6 packages ...', end='', verbose_in=True, verbose_out=M.verbose, set_time=True)
    try:
        for Pkg in [i for i in MF6_Mdl_AoI.keys() if ('riv' in i.lower()) or ('drn' in i.lower())]:
            MF6_Mdl_AoI[Pkg].cleanup(DIS_AoI)
    except Exception:
        print('Failed to cleanup packages. Proceeding without cleanup. Fingers crossed!')
    sprint('🟢', verbose_in=True, verbose_out=M.verbose, print_time=True)

    # ----- MetaSWAP cleanup
    sprint('  - Cleaning up MSW packages ...', end='', verbose_in=True, verbose_out=M.verbose, set_time=True)
    MSW_Mdl_AoI['grid'].dataset['rootzone_depth'] = MSW_Mdl_AoI['grid'].dataset['rootzone_depth'].fillna(1.0)
    sprint('🟢', verbose_in=True, verbose_out=M.verbose, print_time=True)

    # ----- Coupling
    metamod_coupling = timed_Exe(
        primod.MetaModDriverCoupling,
        mf6_model='imported_model',
        mf6_recharge_package='msw-rch',
        mf6_wel_package='msw-sprinkling',
        pre='  - Coupling ...',
        post='🟢',
        verbose_in=True,
        verbose_out=M.verbose,
    )
    metamod = timed_Exe(primod.MetaMod, MSW_Mdl_AoI, Sim_MF6_AoI, coupling_list=[metamod_coupling], pre='')
    M.Pa.MdlN.mkdir(parents=True, exist_ok=True)  # Create simulation directory if it doesn't exist

    # ----- Write Mdl Files
    timed_Exe(
        metamod.write,
        directory=M.Pa.MdlN,
        modflow6_dll=M.Pa.MF6_DLL,
        metaswap_dll=M.Pa.MSW_DLL,
        metaswap_dll_dependency=M.Pa.MF6_DLL.parent,
        pre='  - Writing model files ...',
        post='🟢',
        verbose_in=True,
        verbose_out=M.verbose,
    )
    add_missing_Cols(M.Pa.Pa_MdlN / 'metaswap/mete_grid.inp')

    return Sim_MF6_AoI, MSW_Mdl_AoI
