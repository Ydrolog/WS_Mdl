import imod


def r_with_OBS(
    Pa_PRJ, remove_SS=True, season_to_DT=True
):  # 666 Turn this into a class. And add other functions as methods.
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
    with tempfile.NamedTemporaryFile(delete=False, mode='w', dir=PDN(Pa_PRJ), suffix='.prj') as temp_file:
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


def o_with_OBS(Pa_PRJ):
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

    sprint(f'🟢🟢 - PRJ loaded from {Pa_PRJ}')
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

    else:
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

    item_name_lower = item_name.lower()

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
            vprint(f'  {item_name}: ⚫️ - Already on target grid')
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
