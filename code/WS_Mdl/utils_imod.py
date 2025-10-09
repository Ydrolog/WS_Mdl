# ***** Similar to utils.py, but those utilize imod, which takes a long time to load. *****
import math
import os
import re
import subprocess as sp
import tempfile
from datetime import datetime as DT
from os import makedirs as MDs
from os.path import basename as PBN
from os.path import dirname as PDN
from os.path import exists as PE
from os.path import join as PJ

import imod
import numpy as np
import pandas as pd
import primod
import xarray as xr
from filelock import FileLock as FL
from imod import mf6, msw
from tqdm import tqdm  # Track progress of the loop

from .utils import (
    INI_to_d,
    Mdl_Dmns_from_INI,
    Pa_WS,
    Pre_Sign,
    Sign,
    get_MdlN_Pa,
    r_IPF_Spa,
    set_verbose,
    vprint,
)

CuCh = {
    '-': '🔴',  # negative
    '0': '🟡',  # neutral
    '+': '🟢',  # positive
    '=': '⚪️',  # no action required
    'x': '⚫️',  # already done
}  # Rule for using multiple e.g. 🟢🟢🟢. Use 2 when a function returns an object. Use 3 for more impactful functions that save a file, or complete a longer process, like commiting git changes. In all other cases use 1.


# PRJ related --------------------------------------------------------------------
def r_PRJ_with_OBS(Pa_PRJ, remove_SS=True, season_to_DT=True):
    """
    imod.formats.prj.read_projectfile struggles with .prj files that contain OBS blocks. This will read the PRJ file and return a tuple. The first item is a PRJ dictionary (as imod.formats.prj would return) and also a list of the OBS block lines.

    Pa_PRJ: a the path of the PRJ file.
    remove_SS: if True, removes any steady state periods from the PRJ file. This is useful when working with transient models only, as steady state periods can cause issues with some packages (e.g. RIV, DRN) when using imod python.
    season_to_DT: CAUTION!!! this can be set to True so that imod.formats.prj.read_projectfile() doesn't fail when it encounters season names (winter, summer) in the PRJ blocks, but it won't apply them properly (as iMOD5 would do). This function would need to be upgraded for that to happen.
    """
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
                l_filtered_Lns.pop(i)
            elif record_periods and Ln.strip() == '':  # End of season block
                record_periods = False
            else:
                if 'winter\n' == Ln.lower():
                    season_map[Ln] = l_filtered_Lns1[i + 1].strip()  # Next line should contain the date
                elif 'summer\n' == Ln.lower():
                    season_map[Ln] = l_filtered_Lns1[i + 1].strip()  # Next line should contain the date
        l_filtered_Lns2 = [season_map.get(x, x) for x in l_filtered_Lns1]

    # Write to a temporary file
    l_filtered_Lns = l_filtered_Lns2
    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.prj') as temp_file:
        temp_file.writelines(l_filtered_Lns)
        Pa_PRJ_temp = temp_file.name

    try:
        PRJ = imod.formats.prj.read_projectfile(Pa_PRJ_temp)  # Load the PRJ file without OBS
    except Exception as e:
        print(f'Error reading PRJ file: {e}')
        PRJ = None
    os.remove(Pa_PRJ_temp)  # Delete temp PRJ file as it's not needed anymore.

    return PRJ, l_OBS_Lns


def PRJ_to_DF(MdlN):
    """Leverages r_PRJ_with_OBS to produce a DF with the PRJ data.
    Could have been included in utils.py based on dependencies, but utils_imod.py fits it better as it's almost always used after r_PRJ_with_OBS (so the libs will be already loaded)."""

    d_Pa = get_MdlN_Pa(MdlN)

    Mdl = ''.join([c for c in MdlN if not c.isdigit()])
    Pa_AppData = os.path.normpath(PJ(os.getenv('APPDATA'), '../'))
    t_Pa_replace = (
        Pa_AppData,
        PJ(Pa_WS, 'models', Mdl),
    )  # For some reason imod.idf.read reads the path incorrectly, so I have to replace the incorrect part.

    d_PRJ, OBS = r_PRJ_with_OBS(d_Pa['PRJ'])

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

    vprint(' --- Reading PRJ Packages into DF ---')
    for Pkg_name in list(d_PRJ.keys()):  # Iterate over packages
        vprint(f'\t{Pkg_name:<7}\t...\t', end='')
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
                vprint('🟢')
            else:
                vprint('\u2012 the package is innactive.')
        except Exception as e:
            DF.loc[f'{Pkg_name.upper()}'] = '-'
            DF.loc[f'{Pkg_name.upper()}', 'active'] = f'Failed to read package: {e}'
            vprint('🟡')
    vprint('🟢🟢')
    vprint(f' {"-" * 100}')

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


def o_PRJ_with_OBS(Pa_PRJ):
    """
    imod.formats.prj.read_projectfile struggles with .prj files that contain OBS blocks. This will read the PRJ file and return a tuple. The first item is a PRJ dictionary (as imod.formats.prj would return), the 2nd is a list of the OBS block lines.
    ATM this is safer, as it can deal with both PRJ files with and without OBS blocks.
    """

    Dir_PRJ = os.path.dirname(Pa_PRJ)  # Directory of the PRJ file

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
    os.remove(Pa_PRJ_temp)  # Delete temp PRJ file as it's not needed anymore.

    vprint(f'🟢🟢 - PRJ loaded from {Pa_PRJ}')
    return PRJ, l_OBS_Lns


def o_PRJ_with_OBS_old(Pa_PRJ):
    """imod.formats.prj.read_projectfile struggles with .prj files that contain OBS blocks. This will read the PRJ file and return a tuple. The first item is a PRJ dictionary (as imod.formats.prj would return), the 2nd is a list of the OBS block lines.
    - This is an old version that uses imod.formats.prj.open_projectfile_data, which is much slower than imod.formats.prj.open_projectfile. Use o_PRJ_with_OBS instead.
    - Actually I think the speed difference is attributed to the fact that the data were already read into memory. There is no significant difference in speed when reading the PRJ file for the first time (betwen the two functions)."""

    Dir_PRJ = os.path.dirname(Pa_PRJ)  # Directory of the PRJ file

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
    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.prj', dir=Dir_PRJ) as temp_file:
        temp_file.writelines(l_filtered_Lns)
        Pa_PRJ_temp = temp_file.name

    PRJ = imod.formats.prj.open_projectfile_data(Pa_PRJ_temp)  # Load the PRJ file without OBS
    os.remove(Pa_PRJ_temp)  # Delete temp PRJ file as it's not needed anymore.

    return PRJ, l_OBS_Lns


def regrid_PRJ(PRJ, MdlN: str = None, x_CeCes=None, y_CeCes=None, method='linear'):
    """
    Regrid all spatial data in PRJ to target discretization.
    If x_CeCes and y_CeCes are provided, they will be used as the target grid.
    If MdlN is provided, it will use the model's spatial dimensions from the INI file.
    """

    if x_CeCes is not None and y_CeCes is not None:
        dx = x_CeCes[1] - x_CeCes[0] if len(x_CeCes) > 1 else 0
        dy = y_CeCes[1] - y_CeCes[0] if len(y_CeCes) > 1 else 0

    elif MdlN:  # If MdlN is provided, get the model's spatial dimensions from INI file
        x_CeCes, y_CeCes = get_CeCes_from_INI(MdlN)
        dx = x_CeCes[1] - x_CeCes[0] if len(x_CeCes) > 1 else 0
        dy = y_CeCes[1] - y_CeCes[0] if len(y_CeCes) > 1 else 0

    else:  # If MdlN is not provided, use default values
        vprint('🔴🔴🔴 - Either MdlN or x_CeCes and y_CeCes must be provided. Cancelling regridding...')
        return  # Stop regridding if no valid grid is provided

    vprint(f'\nTarget grid: {len(x_CeCes)}x{len(y_CeCes)} cells at {dx:.1f}x{dy:.1f} m resolution')
    vprint(
        f'\nTarget extents: X=[{x_CeCes.min():.1f}, {x_CeCes.max():.1f}], Y=[{y_CeCes.min():.1f}, {y_CeCes.max():.1f}]'
    )

    PRJ_regridded = {}

    for key, data in PRJ.items():
        vprint(f'Processing {key}...')

        if isinstance(data, dict):
            # Handle nested dictionaries (like 'cap', 'bnd')
            PRJ_regridded[key] = {}
            for sub_key, sub_data in data.items():
                PRJ_regridded[key][sub_key] = regrid_DA(sub_data, x_CeCes, y_CeCes, dx, dy, f'{key}.{sub_key}', method)
        else:
            # Handle top-level data
            PRJ_regridded[key] = regrid_DA(data, x_CeCes, y_CeCes, dx, dy, key, method)

    vprint('🟢🟢🟢 - PRJ has been regridded successfully!')
    return PRJ_regridded


def regrid_DA(DA, x_CeCes, y_CeCes, dx, dy, item_name, method='linear'):
    """Handle regridding of individual DA items"""

    # Skip if not xarray or no spatial dimensions
    if not hasattr(DA, 'dims') or not ('x' in DA.dims and 'y' in DA.dims):
        vprint(f'  {item_name}: ⚪️ - No spatial dims - keeping original')
        return DA

    # Check if DA is already on target grid before doing anything
    DA_x, DA_y = DA.x.values, DA.y.values
    if len(DA_x) > 1 and len(DA_y) > 1:
        DA_dx = DA_x[1] - DA_x[0]
        DA_dy = abs(DA_y[1] - DA_y[0])

        # Compare with tolerance
        same_resolution = np.isclose(DA_dx, dx, atol=1e-6) and np.isclose(DA_dy, dy, atol=1e-6)
        same_size = len(DA_x) == len(x_CeCes) and len(DA_y) == len(y_CeCes)

        # Check if coordinates match (within tolerance)
        coords_match = False
        if same_size:
            try:
                coords_match = np.allclose(DA_x, x_CeCes, atol=1e-6) and np.allclose(DA_y, y_CeCes, atol=1e-6)
            except Exception:
                coords_match = False

        if same_resolution and same_size and coords_match:
            vprint(f'  {item_name}: ⚫️ - Already on target grid')
            return DA

    # Handle special DA types
    if 'ibound' in item_name.lower():
        # Use nearest neighbor for boundary conditions
        method = 'nearest'
    elif any(x in item_name.lower() for x in ['landuse', 'soil_unit', 'zone']):
        # Use nearest neighbor for categorical DA
        method = 'nearest'
    elif 'area' in item_name.lower():
        # Special handling for area fields - scale by grid ratio
        regridded = DA.interp(x=x_CeCes, y=y_CeCes, method='linear')
        grid_ratio = (len(x_CeCes) * len(y_CeCes)) / (len(DA.x) * len(DA.y))
        regridded = regridded * grid_ratio

        # Attach dx and dy attributes to the regridded DataArray
        regridded = regridded.assign_coords(dx=dx, dy=dy)
        vprint(f'  {item_name}: 🟢 - Area field regridded with grid ratio scaling')
        return regridded

    # Option A: Interpolate first, then clip to target bounds
    # This is simpler but more computationally expensive for large arrays
    try:
        regridded = DA.interp(x=x_CeCes, y=y_CeCes, method=method)

        # Attach dx and dy attributes to the regridded DataArray
        regridded = regridded.assign_coords(dx=dx, dy=dy)
        vprint(f'  {item_name}: 🟢 - {DA.sizes} -> {regridded.sizes}. Method: {method}.')
        return regridded
    except Exception as e:
        vprint(f'  {item_name}: 🔴 - Regridding failed ({e}) - keeping original')
        return DA


def mete_grid_Cvt_to_AbsPa(Pa_PRJ: str, PRJ: dict = None):
    """
    Converts mete_grid.inp paths to absolute paths in the PRJ file.
    This is necessary because imod doesn't handle relative paths in mete_grid.inp correctly.
    - Pa_PRJ is necessary cause it is used in the path conversion.
    - PRJ is optional, if not provided, it will be loaded from Pa_PRJ.
    Returns Pa of mete_grid.inp with absolute paths.
    """

    Dir_PRJ = PDN(Pa_PRJ)

    if not PRJ:  # If PRJ is not provided, load it from Pa_PRJ
        PRJ_, PRJ_OBS = o_PRJ_with_OBS(Pa_PRJ)
        PRJ, period_data = PRJ_[0], PRJ_[1]
        return None

    Pa_mete_grid = PRJ['extra']['paths'][2][0]  # 3rd file (index 2) (by default. immutable order)

    # Load mete_grid, edit and save it
    Pa_mete_grid_AbsPa = PJ(PDN(Pa_mete_grid), 'temp', 'mete_grid.inp')
    if not PE(PDN(Pa_mete_grid_AbsPa)):
        MDs(PDN(Pa_mete_grid_AbsPa))

    DF = pd.read_csv(Pa_mete_grid, header=None, names=['N', 'Y', 'P', 'PET'])
    DF.P = DF.P.apply(lambda x: os.path.abspath(PJ(Dir_PRJ, x)))
    DF.PET = DF.PET.apply(lambda x: os.path.abspath(PJ(Dir_PRJ, x)))  # Fixed: was DF.P instead of DF.PET

    # Write CSV with proper format to avoid imod parsing issues with newlines
    # imod doesn't strip newlines from paths, so we need to format carefully
    corrected_lines = []
    for index, row in DF.iterrows():
        # Add quotes around paths like the original format
        line = f'{row["N"]},{row["Y"]},"{row["P"]}","{row["PET"]}"'
        corrected_lines.append(line)

    # Write without newlines in path columns
    with open(Pa_mete_grid_AbsPa, 'w') as f:
        for i, line in enumerate(corrected_lines):
            if i == len(corrected_lines) - 1:  # Last line - no newline
                f.write(line)
            else:
                f.write(line + '\n')

    print(f'Created corrected mete_grid.inp: {Pa_mete_grid_AbsPa}')

    return Pa_mete_grid_AbsPa


# --------------------------------------------------------------------------------

# Mdl related --------------------------------------------------------------------


def Mdl_Prep(MdlN: str, Pa_MF6_DLL: str = None, Pa_MSW_DLL: str = None, verbose=False):
    """
    Prepares Sim Fis from In Fis.
    Ins need to be read and processed, then MF6 and MSW need to be coupled. Then Sim Ins can be written.
    """

    set_verbose(verbose)

    # Load paths and variables from PRJ & INI
    d_Pa = get_MdlN_Pa(MdlN)
    Pa_PRJ = d_Pa['PRJ']
    Dir_PRJ = PDN(Pa_PRJ)
    d_INI = INI_to_d(d_Pa['INI'])
    Xmin, Ymin, Xmax, Ymax = [float(i) for i in d_INI['WINDOW'].split(',')]
    SP_date_1st, SP_date_last = [
        DT.strftime(DT.strptime(d_INI[f'{i}'], '%Y%m%d'), '%Y-%m-%d') for i in ['SDATE', 'EDATE']
    ]
    dx = dy = float(d_INI['CELLSIZE'])

    if not Pa_MF6_DLL:  # If not specified, the default location will be used.
        Pa_MF6_DLL = d_Pa['MF6_DLL']
    if not Pa_MSW_DLL:
        Pa_MSW_DLL = d_Pa['MSW_DLL']

    # Load PRJ & regrid it to Mdl Aa
    PRJ_, PRJ_OBS = o_PRJ_with_OBS(Pa_PRJ)
    PRJ, period_data = PRJ_[0], PRJ_[1]
    PRJ_regrid = regrid_PRJ(
        PRJ, MdlN
    )  # Using original PRJ to load MF6 Mdl gives warnings (and it's very slow). Regridding works much better though.

    # Set outer boundaries to -1. Otherwise CHD won't be loaded properly.
    BND = PRJ_regrid['bnd']['ibound']
    BND.loc[:, [BND.y[0], BND.y[-1]], :] = -1  # Top and bottom rows
    BND.loc[:, :, [BND.x[0], BND.x[-1]]] = -1  # Left and right columns
    vprint('🟢 - Boundary conditions set successfully!')

    # Load MF6 Simulation
    times = pd.date_range(SP_date_1st, SP_date_last, freq='D')
    Sim_MF6 = mf6.Modflow6Simulation.from_imod5_data(
        PRJ_regrid, period_data, times
    )  # It can be further sped up by multi-processing, but this is not implemented yet.
    vprint('🟢 - MF6 Simulation loaded successfully!')
    # Sim_MF6[f'{MdlN}'] = Sim_MF6.pop('imported_model')  # Rename imported_model to MdlN.

    # Pass the Sim components to objects.
    MF6_Mdl = Sim_MF6['imported_model']
    MF6_Mdl['oc'] = mf6.OutputControl(save_head='last', save_budget='last')
    Sim_MF6['ims'] = mf6_solution_moderate_settings()  # Mimic iMOD5's "Moderate" settings.
    MF6_DIS = MF6_Mdl['dis']

    # Load MSW
    PRJ_MSW = {'cap': PRJ_regrid.copy()['cap'], 'extra': PRJ_regrid.copy()['extra']}  # Isolate MSW keys from PRJ.
    PRJ_MSW['extra']['paths'][2][0] = mete_grid_Cvt_to_AbsPa(
        Pa_PRJ, PRJ
    )  ## Fix mete_grid.inp relative paths. Replace the mete_grid.inp path in the PRJ_MSW dictionary
    MSW_Mdl = msw.MetaSwapModel.from_imod5_data(PRJ_MSW, MF6_DIS, times)  # Load MSW model from PRJ
    vprint('🟢 - MSW Simulation loaded successfully!')

    # Clip models
    Sim_MF6_AoI = Sim_MF6.clip_box(x_min=Xmin, x_max=Xmax, y_min=Ymin, y_max=Ymax)
    MF6_Mdl_AoI = Sim_MF6_AoI['imported_model']
    MSW_Mdl_AoI = MSW_Mdl.clip_box(
        x_min=Xmin, x_max=Xmax, y_min=Ymin, y_max=Ymax
    )  # clip_box doesn't clip the packages I clipped beforehand, but it clips non raster-like packages like WEL and removes packages that are not in the AoI.
    print(f'MF6 Model AoI DIS shape: {MF6_Mdl_AoI["dis"].dataset.sizes}')
    print(f'MSW Model AoI grid shape: {MSW_Mdl_AoI["grid"].dataset.sizes}')
    print('🟢 Both models successfully clipped to Area of Interest with compatible discretization!')

    ## I've sense checked that the AoI models are correct. Check imod_python_init_NBr32.ipynb for more info.

    # Load models into memory
    for pkg in MF6_Mdl_AoI.values():
        pkg.dataset.load()

    for pkg in MSW_Mdl_AoI.values():
        pkg.dataset.load()

    # Create mask from current regridded model (not the old one)
    mask = (
        MF6_Mdl_AoI.domain
    )  # 666 mask needs to be checked and potentially updated with -1 values at the edge of the Mdl Aa.
    Sim_MF6_AoI.mask_all_models(mask)
    DIS_AoI = MF6_Mdl_AoI['dis']

    ### MF6 cleanup
    try:
        for Pkg in [i for i in MF6_Mdl_AoI.keys() if ('riv' in i.lower()) or ('drn' in i.lower())]:
            MF6_Mdl_AoI[Pkg].cleanup(DIS_AoI)
    except:
        print('Failed to cleanup packages. Proceeding without cleanup. Fingers crossed!')

    # MetaSWAP cleanup
    MSW_Mdl_AoI['grid'].dataset['rootzone_depth'] = MSW_Mdl_AoI['grid'].dataset['rootzone_depth'].fillna(1.0)

    # Coupling
    metamod_coupling = primod.MetaModDriverCoupling(
        mf6_model='imported_model', mf6_recharge_package='msw-rch', mf6_wel_package='msw-sprinkling'
    )
    metamod = primod.MetaMod(MSW_Mdl_AoI, Sim_MF6_AoI, coupling_list=[metamod_coupling])
    os.makedirs(d_Pa['Pa_MdlN'], exist_ok=True)  # Create simulation directory if it doesn't exist

    # Write Mdl Files
    metamod.write(
        directory=d_Pa['Pa_MdlN'],
        modflow6_dll=Pa_MF6_DLL,
        metaswap_dll=Pa_MSW_DLL,
        metaswap_dll_dependency=PDN(Pa_MF6_DLL),
    )

    # # Review execution times per cell
    try:
        result = sp.run(
            [d_Pa['coupler_Exe'], d_Pa['TOML']], cwd=d_Pa['Pa_MdlN'], capture_output=True, text=True, timeout=3600
        )  # 1 hour timeout

        print(f'Return code: {result.returncode}')
        if result.stdout:
            print('STDOUT:')
            print(result.stdout)
        if result.stderr:
            print('STDERR:')
            print(result.stderr)

        if result.returncode == 0:
            print('✅ Model execution completed successfully!')
        else:
            print(f'❌ Model execution failed with return code {result.returncode}')

    except sp.TimeoutExpired:
        print('⏰ Model execution timed out after 1 hour')
    except Exception as e:
        print(f'❌ Error executing model: {e}')


# --------------------------------------------------------------------------------


# PrSimP related -----------------------------------------------------------------
def add_OBS(MdlN: str, Opt: str = 'BEGIN OPTIONS\nEND OPTIONS', iMOD5=False):
    """
    Adds OBS file(s) from PRJ file OBS block to Mdl Sim (which iMOD can't do). Thus the OBS file needs to be written, and then a link to the OBS file needs to be created within the NAM file.
    Assumes OBS IPF file contains the following parameters/columns: 'Id', 'L', 'X', 'Y'
    for iMOD5 option check WS_Mdl.utils.get_MdlN_Pa() description.
    """

    vprint(Pre_Sign)
    vprint('Running add_OBS ...')
    d_Pa = get_MdlN_Pa(MdlN, iMOD5=iMOD5)  # Get default directories
    Pa_MdlN, Pa_INI, Pa_PRJ = (
        d_Pa[k] for k in ['Pa_MdlN', 'INI', 'PRJ']
    )  # and pass them to objects that will be used in the function

    # Extract info from INI file.
    d_INI = INI_to_d(Pa_INI)
    Xmin, Ymin, Xmax, Ymax = [float(i) for i in d_INI['WINDOW'].split(',')]
    cellsize = float(d_INI['CELLSIZE'])
    # N_R, N_C = int( - (Ymin - Ymax) / cellsize ), int( (Xmax - Xmin) / cellsize ),

    # Read PRJ file to extract OBS block info - list of OBS files to be added.
    l_OBS_lines = r_PRJ_with_OBS(Pa_PRJ)[1]
    pattern = r"['\",]([^'\",]*?\.ipf)"  # Regex pattern to extract file paths ending in .ipf
    l_IPF = [
        match.group(1) for line in l_OBS_lines for match in re.finditer(pattern, line)
    ]  # Find all IPF files of the OBS block.

    # Iterate through OBS files of OBS blocks and add them to the Sim
    for i, path in enumerate(l_IPF):
        Pa_OBS_IPF = os.path.abspath(PJ(Pa_MdlN, path))  # path of IPF file. To be read.
        OBS_IPF_Fi = PBN(Pa_OBS_IPF)  # Filename of OBS file to be added to Sim (to be added without ending)
        if i == 0:
            Pa_OBS = PJ(Pa_MdlN, f'GWF_1/MODELINPUT/{MdlN}.OBS6')  # path of OBS file. To be written.
        else:
            Pa_OBS = PJ(Pa_MdlN, f'GWF_1/MODELINPUT/{MdlN}_N{i}.OBS6')  # path of OBS file. To be written.

        DF_OBS_IPF = r_IPF_Spa(
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
            # vprint(Pa_MdlN, path, Pa_OBS_IPF, sep='\n')
            f.write(f'# created from {Pa_OBS_IPF}\n')
            f.write(Opt.encode().decode('unicode_escape'))  # write optional block
            f.write(f'\n\nBEGIN CONTINUOUS FILEOUT OBS_{OBS_IPF_Fi.split(".")[0]}.csv\n')

            for _, row in DF_OBS_IPF_MdlAa.drop_duplicates(subset=['Id', 'L', 'R', 'C']).iterrows():
                f.write(f' {row["Id"]} HEAD {row["L"]} {row["R"]} {row["C"]}\n')

            f.write('END CONTINUOUS\n')

        # Open NAM file and add OBS file to it
        lock = FL(d_Pa['NAM_Mdl'] + '.lock')  # Create a file lock to prevent concurrent writes
        with lock, open(d_Pa['NAM_Mdl'], 'r+') as f:
            l_NAM = f.read().split('END PACKAGES')
            f.seek(0)
            f.truncate()  # overwrite in-place
            Pa_OBS_Rel = os.path.relpath(Pa_OBS, Pa_MdlN)

            f.write(l_NAM[0])
            f.write(rf' OBS6 .\{Pa_OBS_Rel} OBS_{OBS_IPF_Fi.split(".")[0]}')
            f.write('\nEND PACKAGES')

            f.flush()
            os.fsync(f.fileno())  # ensure it’s on disk
            # lock is released automatically when the with-block closes
        vprint(f'🟢 - {Pa_OBS} has been added successfully!')
    vprint(Sign)


# --------------------------------------------------------------------------------


# IDF processing -----------------------------------------------------------------
def IDFs_to_DF(S_Pa_IDF):
    """Reads all .IDF Fis listed in a S_Fi_IDF into DF['IDF']. Returns the DF containing Fi_names and the IDF contents.
    Pa_Fo is the path of the Fo where th files are stored in."""

    DF = pd.DataFrame({'path': S_Pa_IDF, 'IDF': None})

    for i, p in tqdm(DF['path'].items(), desc='Loading .IDF files', total=len(DF['path'])):
        if p.endswith('.IDF'):  # Ensure only .IDF files are processed
            try:  # Read the .IDF file into an xA DataA
                DF.at[i, 'IDF'] = imod.idf.read(p)
            except Exception as e:
                print(f'Error reading {p}: {e}')
    return DF


# --------------------------------------------------------------------------------


# xarray processing --------------------------------------------------------------
def xr_describe(data, name: str = None):
    """
    Generates descriptive statistics for an xarray DataArray or Dataset,
    including value statistics and coordinate ranges.

    If a Dataset is provided, it will describe each DataArray within it.
    """

    def _describe_da(da: xr.DataArray, da_name: str):
        """Helper function to describe a single DataArray."""
        print(f'--- Statistics for: {da_name} ---')

        # Explicitly load data into memory to work with NumPy arrays
        da = da.load()

        # --- Value Statistics ---
        # Check if the data is numeric before calculating statistics
        if np.issubdtype(da.dtype, np.number):
            if da.count().item() > 0:
                stats = {
                    'count': da.count().item(),
                    'mean': da.mean().item(),
                    'std': da.std().item(),
                    'min': da.min().item(),
                    '25%': da.quantile(0.25).item(),
                    '50%': da.median().item(),
                    '75%': da.quantile(0.75).item(),
                    'max': da.max().item(),
                }
                value_stats = pd.Series(stats, name=da_name)
                print(value_stats)
            else:
                print('Array has no valid data (all NaNs or empty).')
        else:
            # For non-numeric data, show a simpler summary
            print(f'Variable is non-numeric (dtype: {da.dtype}).')
            unique_vals, counts = np.unique(da.values, return_counts=True)
            print('Unique values and their counts:')
            for val, count in zip(unique_vals, counts):
                print(f'  - {val}: {count}')

        # --- Coordinate Ranges ---
        print('\n--- Coordinate Summary ---')
        for coord_name in da.coords:
            coord = da.coords[coord_name]
            if coord.ndim == 1:
                summary = {'count': coord.size}

                c_min = coord.min().values
                c_max = coord.max().values

                if np.issubdtype(coord.dtype, np.datetime64):
                    summary['min'] = pd.to_datetime(c_min).strftime('%Y-%m-%d')
                    summary['max'] = pd.to_datetime(c_max).strftime('%Y-%m-%d')
                else:
                    summary['min'] = c_min.item()
                    summary['max'] = c_max.item()

                if np.issubdtype(coord.dtype, np.number) and coord.size > 1:
                    diffs = np.diff(coord.values)
                    if np.allclose(diffs, diffs[0]):
                        summary['step'] = diffs[0].item()

                print(f'- {coord_name} ({coord.dtype}):')
                print(pd.Series(summary).to_string())
                print()
        print('-' * 30)

    if isinstance(data, xr.DataArray):
        _describe_da(data, name or data.name or 'DataArray')
    elif isinstance(data, xr.Dataset):
        if name:
            print(f'--- Describing Dataset: {name} ---')
        for var_name, da in data.data_vars.items():
            _describe_da(da, var_name)
    else:
        print(f'Input must be an xarray.DataArray or xarray.Dataset, but got {type(data)}')


def xr_clip_Mdl_Aa(
    xr_data: xr.DataArray | xr.Dataset,
    MdlN: str = None,
    Pa_INI: str = None,
    l_L=None,
    Lmin: int = None,
    Lmax: int = None,
    x_dim: str = 'x',
    y_dim: str = 'y',
    L_dim: str = 'layer',
) -> xr.DataArray | xr.Dataset:
    """
    Clips an xarray DataArray or Dataset to the model area defined in an INI file, with optional layer subsetting.

    Parameters:
    -----------
    xr_data : xr.DataArray or xr.Dataset
        The xarray data to clip
    MdlN : str, optional
        Model name to automatically get INI path via get_MdlN_Pa
    Pa_INI : str, optional
        Direct path to INI file (alternative to MdlN)
    l_L : list-like, optional
        List of layers to select (e.g., [1, 3, 5])
    Lmin : int, optional
        Minimum layer index (inclusive)
    Lmax : int, optional
        Maximum layer index (inclusive)
    x_dim : str, default 'x'
        Name of the x coordinate dimension
    y_dim : str, default 'y'
        Name of the y coordinate dimension
    L_dim : str, default 'layer'
        Name of the layer coordinate dimension

    Returns:
    --------
    xr.DataArray or xr.Dataset
        Clipped xarray data

    Raises:
    -------
    ValueError
        If neither MdlN nor Pa_INI is provided, or if required dimensions are missing
    """

    # Get INI file path
    if Pa_INI is None:
        if MdlN is None:
            raise ValueError('Either MdlN or Pa_INI must be provided')
        d_Pa = get_MdlN_Pa(MdlN)
        Pa_INI = d_Pa['INI']

    # Get model dimensions from INI file
    Xmin, Ymin, Xmax, Ymax, cellsize, N_R, N_C = Mdl_Dmns_from_INI(Pa_INI)

    # Check if required dimensions exist
    if x_dim not in xr_data.coords:
        raise ValueError(f"X dimension '{x_dim}' not found in data coordinates: {list(xr_data.coords.keys())}")
    if y_dim not in xr_data.coords:
        raise ValueError(f"Y dimension '{y_dim}' not found in data coordinates: {list(xr_data.coords.keys())}")

    vprint(f'Clipping xarray data to model area: X=[{Xmin}, {Xmax}], Y=[{Ymin}, {Ymax}]')

    # Check y-coordinate order and adjust slice accordingly
    y_coords = xr_data.coords[y_dim].values
    if len(y_coords) > 1 and y_coords[0] > y_coords[-1]:  # Descending order (big to small)
        y_slice = slice(Ymax, Ymin)
        vprint('Y coordinates in descending order, using slice(Ymax, Ymin)')
    else:  # Ascending order (small to big) or single value
        y_slice = slice(Ymin, Ymax)
        vprint('Y coordinates in ascending order, using slice(Ymin, Ymax)')

    # Clip to spatial extent
    clipped = xr_data.sel({x_dim: slice(Xmin, Xmax), y_dim: y_slice})

    # Handle layer subsetting if layer dimension exists
    if L_dim in xr_data.coords:
        if l_L is not None:
            vprint(f'Selecting specific layers: {l_L}')
            clipped = clipped.sel({L_dim: l_L})
        elif Lmin is not None or Lmax is not None:
            # Build slice for layer range
            layer_slice = slice(Lmin, Lmax)
            vprint(f'Selecting layer range: {Lmin} to {Lmax}')
            clipped = clipped.sel({L_dim: layer_slice})
    elif l_L is not None or Lmin is not None or Lmax is not None:
        vprint(f"Warning: Layer subsetting requested but dimension '{L_dim}' not found in data")

    vprint('🟢🟢 - Successfully clipped xarray data to model area')
    return clipped


def xr_compare_As(
    array1: xr.DataArray,
    array2: xr.DataArray,
    name1: str = 'Array 1',
    name2: str = 'Array 2',
    x_dim: str = 'x',
    y_dim: str = 'y',
    tolerance: float = 1e-10,
    title: str = None,
) -> dict:
    """
    Provides comprehensive diagnostics for comparing two xarray DataArrays.

    Parameters:
    -----------
    array1 : xr.DataArray
        First array to compare
    array2 : xr.DataArray
        Second array to compare
    name1 : str, default "Array 1"
        Name/description for the first array
    name2 : str, default "Array 2"
        Name/description for the second array
    x_dim : str, default 'x'
        Name of the x coordinate dimension
    y_dim : str, default 'y'
        Name of the y coordinate dimension
    tolerance : float, default 1e-10
        Tolerance for considering values as different
    title : str, optional
        Custom title for the diagnostic output

    Returns:
    --------
    dict
        Dictionary containing comparison results and statistics
    """

    if title is None:
        title = f'=== Diagnostic Analysis: {name1} vs {name2} ==='

    print(title)
    print(f'{name1} shape: {array1.shape}')
    print(f'{name2} shape: {array2.shape}')
    print(f'{name1} dtype: {array1.dtype}')
    print(f'{name2} dtype: {array2.dtype}')

    # Initialize results dictionary
    results = {
        'shapes_identical': array1.shape == array2.shape,
        'dtypes_identical': array1.dtype == array2.dtype,
        'arrays_identical': array1.identical(array2),
        'arrays_equal': array1.equals(array2),
    }

    # Check if shapes match
    print(f'\nShapes identical: {results["shapes_identical"]}')

    # Check coordinate differences
    coords_to_check = []
    if x_dim in array1.coords and x_dim in array2.coords:
        coords_to_check.append(x_dim)
    if y_dim in array1.coords and y_dim in array2.coords:
        coords_to_check.append(y_dim)

    for coord_name in coords_to_check:
        coord_identical = array1[coord_name].identical(array2[coord_name])
        results[f'{coord_name}_coords_identical'] = coord_identical
        print(f'{coord_name.upper()} coordinates identical: {coord_identical}')

        if not coord_identical:
            coord1, coord2 = array1.coords[coord_name], array2.coords[coord_name]
            print(f'  {coord_name.upper()} {name1} range: {coord1.min().values:.1f} to {coord1.max().values:.1f}')
            print(f'  {coord_name.upper()} {name2} range: {coord2.min().values:.1f} to {coord2.max().values:.1f}')

            # Check spacing if coordinates have more than one value
            if len(coord1) > 1:
                spacing1 = float(coord1.diff(coord_name)[0].values)
                print(f'  {coord_name.upper()} {name1} spacing: {spacing1:.1f}')
            if len(coord2) > 1:
                spacing2 = float(coord2.diff(coord_name)[0].values)
                print(f'  {coord_name.upper()} {name2} spacing: {spacing2:.1f}')

    # Check if data values are the same (ignoring coordinates/metadata)
    print(f'Data values equal (equals): {results["arrays_equal"]}')

    # Check value differences if possible
    try:
        # Check if we can align them for comparison
        if results['shapes_identical']:
            # Try interpolating array2 to array1 coordinates for comparison
            if x_dim in array2.coords and y_dim in array2.coords:
                array2_aligned = array2.interp({x_dim: array1[x_dim], y_dim: array1[y_dim]}, method='nearest')
            else:
                array2_aligned = array2

            diff = array1 - array2_aligned
            max_diff = abs(diff).max().values
            num_different = (abs(diff) > tolerance).sum().values

            results['max_absolute_difference'] = max_diff
            results['num_different_cells'] = num_different

            print(f'Maximum absolute difference (after alignment): {max_diff}')
            print(f'Number of different cells (tolerance={tolerance}): {num_different}')
        else:
            print('Cannot compare values directly due to different shapes')
            results['max_absolute_difference'] = None
            results['num_different_cells'] = None
    except Exception as e:
        print(f'Error comparing values: {e}')
        results['max_absolute_difference'] = None
        results['num_different_cells'] = None

    # Check data ranges
    try:
        min1, max1 = array1.min().values, array1.max().values
        min2, max2 = array2.min().values, array2.max().values
        results['array1_range'] = (min1, max1)
        results['array2_range'] = (min2, max2)

        print(f'\n{name1} min/max: {min1}/{max1}')
        print(f'{name2} min/max: {min2}/{max2}')
    except Exception as e:
        print(f'Error calculating min/max: {e}')
        results['array1_range'] = None
        results['array2_range'] = None

    # Check for any NaN differences
    try:
        has_nan1 = array1.isnull().any().values
        has_nan2 = array2.isnull().any().values
        results['array1_has_nan'] = has_nan1
        results['array2_has_nan'] = has_nan2

        print(f'{name1} has NaN: {has_nan1}')
        print(f'{name2} has NaN: {has_nan2}')
    except Exception as e:
        print(f'Error checking for NaN values: {e}')
        results['array1_has_nan'] = None
        results['array2_has_nan'] = None

    return results


# --------------------------------------------------------------------------------


# Standard options ---------------------------------------------------------------
def mf6_solution_moderate_settings(
    modelnames: list[str] = ['imported_model'],
    print_option: str = 'summary',
    outer_csvfile: str | None = None,
    inner_csvfile: str | None = None,
    no_ptc: bool | None = None,
    outer_dvclose: float = 0.001,
    outer_maximum: int = 150,
    under_relaxation: str = 'dbd',
    under_relaxation_theta: float = 0.9,
    under_relaxation_kappa: float = 0.0001,
    under_relaxation_gamma: float = 0.0,
    under_relaxation_momentum: float = 0.0,
    backtracking_number: int = 0,
    backtracking_tolerance: float = 0.0,
    backtracking_reduction_factor: float = 0.0,
    backtracking_residual_limit: float = 0.0,
    inner_maximum: int = 30,
    inner_dvclose: float = 0.001,
    inner_rclose: float = 100.0,
    rclose_option: str = 'strict',
    linear_acceleration: str = 'bicgstab',
    relaxation_factor: float = 0.97,
    preconditioner_levels: int = 0,
    preconditioner_drop_tolerance: float = 0.0,
    number_orthogonalizations: int = 0,
):
    """Returns a mf6.Solution object with moderate settings for model solution.
    These settings are suitable for most models, but can be adjusted as needed."""

    return mf6.Solution(
        modelnames=modelnames,
        print_option=print_option,
        outer_csvfile=outer_csvfile,
        inner_csvfile=inner_csvfile,
        no_ptc=no_ptc,
        outer_dvclose=outer_dvclose,
        outer_maximum=outer_maximum,
        under_relaxation=under_relaxation,
        under_relaxation_theta=under_relaxation_theta,
        under_relaxation_kappa=under_relaxation_kappa,
        under_relaxation_gamma=under_relaxation_gamma,
        under_relaxation_momentum=under_relaxation_momentum,
        backtracking_number=backtracking_number,
        backtracking_tolerance=backtracking_tolerance,
        backtracking_reduction_factor=backtracking_reduction_factor,
        backtracking_residual_limit=backtracking_residual_limit,
        inner_maximum=inner_maximum,
        inner_dvclose=inner_dvclose,
        inner_rclose=inner_rclose,
        rclose_option=rclose_option,
        linear_acceleration=linear_acceleration,
        relaxation_factor=relaxation_factor,
        preconditioner_levels=preconditioner_levels,
        preconditioner_drop_tolerance=preconditioner_drop_tolerance,
        number_orthogonalizations=number_orthogonalizations,
    )


# --------------------------------------------------------------------------------


# numpy --------------------------------------------------------------------------
def get_CeCes_from_INI(MdlN: str):
    """Get centroids of the model grid from the INI file of the model.
    Returns x_CeCes, y_CeCes."""

    Xmin, Ymin, Xmax, Ymax, cellsize, N_R, N_C = Mdl_Dmns_from_INI(get_MdlN_Pa(MdlN)['INI'])
    dx = float(cellsize)
    dy = -float(cellsize)

    return np.arange(Xmin + dx / 2, Xmax, dx), np.arange(Ymax + dy / 2, Ymin, dy)


# --------------------------------------------------------------------------------
