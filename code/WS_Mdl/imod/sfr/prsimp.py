import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from WS_Mdl.core.defaults import CRS
from WS_Mdl.core.mdl import Mdl_N
from WS_Mdl.core.style import sprint
from WS_Mdl.imod.mf6.bin import to_DF


def create_SFR_lines(Pa_GPkg: str | Path, verbose: bool, debug_sfr: bool = True):
    """Creates SFR_lines object (from SFRmaker Lib) from a GPkg with a specific structure. The GPkg is expected to have the following columns:"""  # 666 fill with more info
    import sfrmaker as sfr

    Pa_GPkg = Path(Pa_GPkg)

    # %% Load GPkg
    sprint(' -- Create SFR lines.', verbose_in=True, verbose_out=verbose)
    GDF = gpd.read_file(Pa_GPkg)

    # %% Ensure slope
    sprint('  - Ensure slope.', verbose_in=True, verbose_out=verbose, set_time=True)
    if debug_sfr:
        from tabulate import tabulate

        sprint(
            tabulate(
                GDF[['ID', 'Elv_UStr', 'Elv_DStr']].describe(include='all'),
                headers='keys',
                tablefmt='grid',
                showindex=False,
            ),
            verbose_in=True,
            verbose_out=verbose,
            indent=3,
        )

    # %% Check NaN counts
    NaN_count_UStr = GDF['Elv_UStr'].isna().sum()
    NaN_count_DStr = GDF['Elv_DStr'].isna().sum()
    if NaN_count_UStr > 0 or NaN_count_DStr > 0:
        sys.exit(
            f'ERROR: There are {NaN_count_UStr} NaN values in Elv_UStr and {NaN_count_DStr} NaN values in Elv_DStr. Please fill these values before proceeding.'
        )

    # %% Enumerate segments by slope
    GDF_Elv = GDF[['ID', 'Elv_UStr', 'Elv_DStr', 'DStr_code', 'DStr_ID']].copy()
    GDF_Elv['Diff'] = GDF_Elv['Elv_UStr'] - GDF_Elv['Elv_DStr']

    N_UStr_LT_DStr = int((GDF_Elv['Diff'] < 0).sum())
    N_UStr_Eq_DStr = int((GDF_Elv['Diff'] == 0).sum())
    N_UStr_GT_DStr = int((GDF_Elv['Diff'] > 0).sum())

    sprint(
        f'    There are:\n\t{N_UStr_LT_DStr:6} segments where Elv_UStr < Elv_DStr\n\t{N_UStr_Eq_DStr:6} segments where Elv_UStr = Elv_DStr\n\t{N_UStr_GT_DStr:6} segments where Elv_UStr > Elv_DStr\n\t{GDF.shape[0]:6} segments in total.',
        verbose_in=True,
        verbose_out=verbose,
    )

    # %% Identify problematic Elv segments
    sprint(
        "It's assumed SFRmaker works where Elv_UStr >= Elv_DStr. Only segments where Elv_UStr < Elv_DStr will be adjusted.",
        verbose_in=True,
        verbose_out=verbose,
        indent=3,
    )

    if debug_sfr:
        sprint('\tIncline segments:', indent=3, verbose_in=True, verbose_out=verbose)
        sprint(
            tabulate(GDF_Elv.loc[GDF_Elv['Diff'] < 0].head(), headers='keys', tablefmt='grid', showindex=False),
            verbose_in=True,
            verbose_out=verbose,
        )

        sprint('\tHorizontal segments:', indent=3, verbose_in=True, verbose_out=verbose)
        sprint(
            tabulate(GDF_Elv.loc[GDF_Elv['Diff'] == 0].head(), headers='keys', tablefmt='grid', showindex=False),
            verbose_in=True,
            verbose_out=verbose,
        )

    # %% Check if problematic segments have multiple UStr segments
    if debug_sfr:
        l_Incl = GDF_Elv.loc[GDF_Elv['Diff'] < 0, 'ID'].tolist()
        if len(l_Incl) > 0:
            l_Incl_multiple_UStr = []
            for S in l_Incl:
                sum = (GDF['DStr_ID'] == S).sum()
                if sum > 1:
                    l_Incl_multiple_UStr.append((S, sum))
            if len(l_Incl_multiple_UStr) > 0:
                sprint(
                    '\tSegments with Elv_UStr < Elv_DStr that have multiple UStr segments (complicates adjustment):',
                    indent=3,
                    verbose_in=True,
                    verbose_out=verbose,
                )
                sprint(
                    tabulate(
                        l_Incl_multiple_UStr,
                        headers=['ID', 'Number of UStr segments'],
                        tablefmt='grid',
                        showindex=False,
                    ),
                    verbose_in=True,
                    verbose_out=verbose,
                )

    # %%
    """
    Elv correction algorithm:
    To fix segments with Elv_UStr < Elv_DStr. Segments with Elv_UStr = Elv_DStr will be fixed by SFR itself (hopefully). The following abbreviations are useful for explaining the concept:
    - A: DStr Elv of DStr segment
    - B: UStr Elv of DStr segment
    - C: DStr Elv of current segment
    - D: UStr Elv of current segment
    - F: DStr Elv of UStr segment(s)

    Here is the idea behind the algorithm:
    1. If C > D & B <= D :-> Set C = D
    2. If C > D & B > D :-> Set C = D. Set B = D
    3. If C <= D :-> No action.

    Repeat till there are no segments with C < D.

    When there is no downstream segment, we apply the logic used in case 1.

    When there is only 1 UStr segment, the DStr Elv of the UStr segment can be modified to allow the UStr Elv of the current segmet to be increased as well, but if there are multiple, this becomes more complicated.
    """

    sprint(
        'Applying elevation correction algorithm to segments where Elv_UStr < Elv_DStr.',
        verbose_in=True,
        verbose_out=verbose,
        indent=3,
        set_time=True,
        end='',
    )
    GDF_Elv = GDF_Elv.merge(
        GDF[['ID', 'Elv_UStr', 'Elv_DStr']], left_on='DStr_ID', right_on='ID', suffixes=('', '_DStr'), how='left'
    )
    GDF_Elv[['A', 'B']] = GDF_Elv[['Elv_UStr_DStr', 'Elv_DStr_DStr']].copy()
    GDF_Elv[['C', 'D']] = GDF_Elv[['Elv_UStr', 'Elv_DStr']].copy()
    GDF_Elv[GDF_Elv['B'].isna()]

    def adjust_elevations(row):
        if row['C'] <= row['D']:  # If UStr Elv <= DStr Elv, no adjustment needed
            return row['B'], row['C']
        elif (row['C'] > row['D']) and (
            pd.isna(row['B'])
        ):  # If UStr Elv <= DStr Elv but DStr Elv is missing (OuFl segment)
            return pd.NA, row['D']
        elif (row['C'] > row['D']) and (row['B'] <= row['D']):
            return row['B'], row['D']
        elif (row['C'] > row['D']) and (row['B'] > row['D']):
            return row['D'], row['D']
        else:
            # Default case - should not happen, but ensures function always returns a tuple
            return row['B'], row['C']

    GDF_Elv[['B_', 'C_']] = GDF_Elv.apply(adjust_elevations, axis=1, result_type='expand')
    sprint('🟢', verbose_in=True, verbose_out=verbose, print_time=True)

    # %% Print problematic segments after adjustment
    # I'm worried consequtive segments might be problematic. Let's check if there are any.
    if debug_sfr:
        GDF_Elv_unfixed = GDF_Elv[(GDF_Elv['Diff'] < 0)]
        consequtive = GDF_Elv_unfixed.loc[GDF_Elv_unfixed['DStr_ID'].isin(GDF_Elv_unfixed['ID']), 'DStr_ID']
        sprint(
            "Segments where Elv_UStr < Elv_DStr that are consequtive (i.e. one segment's DStr_ID is another segment's ID). Those might cause problems later.:",
            verbose_in=True,
            verbose_out=verbose,
            indent=3,
        )
        sprint(
            tabulate(
                GDF_Elv_unfixed.loc[
                    (GDF_Elv_unfixed['ID'].isin(consequtive)) | (GDF_Elv_unfixed['DStr_ID'].isin(consequtive)),
                    ['ID', 'DStr_ID', 'A', 'B', 'B_', 'C', 'C_', 'D'],
                ]
                .sort_values(by='D')
                .reset_index(drop=True),
                headers='keys',
                tablefmt='grid',
                showindex=False,
            ),
            verbose_in=True,
            verbose_out=verbose,
        )

        GDF_Elv['segment_drop'] = GDF_Elv['D'] - GDF_Elv['C_']
        GDF_Elv['DStr_drop'] = GDF_Elv['C_'] - GDF_Elv['B']
        sprint(
            'Segments where the adjusted DStr Elv (C_) is still greater than the UStr Elv of the DStr segment (B). Those might cause problems later:',
            verbose_in=True,
            verbose_out=verbose,
            indent=3,
        )
        sprint(
            tabulate(
                GDF_Elv.loc[
                    GDF_Elv['C_'] - GDF_Elv['B_'] < 0,
                    ['ID', 'DStr_ID', 'A', 'B', 'B_', 'C', 'C_', 'D', 'segment_drop', 'DStr_drop'],
                ]
                .sort_values(by='DStr_drop')
                .reset_index(drop=True),
                headers='keys',
                tablefmt='grid',
                showindex=False,
            ),
            verbose_in=True,
            verbose_out=verbose,
        )

    # %%
    # print(GDF_Elv.loc[ GDF_Elv['D'] - GDF_Elv['C_'] < 0 ])

    # %% Merge adjusted elevations back to GDF
    # GDF2 = GDF.copy()
    GDF = GDF.merge(GDF_Elv[['ID', 'C_', 'D']], on='ID', how='left')

    # %% Generate SFRmaker lines

    GDF['width2'] = GDF['width'].copy()
    lines = sfr.Lines.from_dataframe(
        df=GDF.copy(),  # .copy() to avoid GDF columns being renamed by function (this feels like a bug to me)
        id_column='ID',
        routing_column='DStr_ID',
        width1_column='width',
        width2_column='width2',
        dn_elevation_column='C_',
        up_elevation_column='D',
        name_column='CODE',
        width_units='m',
        height_units='m',
        crs=GDF.crs,
        #    shapefile=Pa_GPkg_1ry_SHP_SFR,
    )
    return lines


def connect_SFR_lines_to_MF6(M: Mdl_N, debug_sfr: bool = True):
    """
    Creates SFR file for MF6 Sim from SFR lines and connects it to the MF6 NAM file.
    Takes in Mdl_N instance as an argument. It needs to have following extra attributes (extra: attributes not created on __init__):
    - lines: sfrmaker.lines object. (Can be created by create_SFR_lines function in this module.)
    - Sim_MF6: imod.mf6.Modflow6Simulation instance. (Can be created by WS_Mdl.imod.prj.PrSimP function.)
    - verbose: bool, for printing progress and info.
    - Pa_Cond_A: Path to primary conductance IDF file.
    - Pa_Cond_B: Path to secondary conductance IDF file. Used wherever primary Cond file has not values.
    - Xmin, Xmax, Ymin, Ymax: model domain extents, used for subsetting the Cond IDF files.
    - Pa_SFR_OBS_In: Path to SFR OBS csv file, used for adding SFR OBS to the model.
    -
    """  # 666 fill with more info

    import shutil as sh

    import imod
    import sfrmaker as sfr
    from shapely import box

    # %% Connect SFR to MF6 model
    sprint(' -- SFRmaker - Connecting SFR lines to MF6 model.', verbose_in=True, verbose_out=M.verbose)
    sprint('  - Creating SFR_grid item.', verbose_in=True, verbose_out=M.verbose, set_time=True, end='')
    # Create sfr.StructuredGrid directly from MF6_DIS (DataFrame approach) #666 This cell and the cells below it can be combined into a function to read in a MF6_DIS (imod) object, and return a DF (GDF_grid) with the grid and geometry.
    DS = M.Sim_MF6['imported_model']['dis'].dataset
    N_L, N_R, N_C = DS.dims['layer'], DS.dims['y'], DS.dims['x']
    dx, dy = abs(float(DS.coords['dx'].values)), abs(float(DS.coords['dy'].values))
    Xs, Ys = DS.coords['x'].values, DS.coords['y'].values
    X_Ogn, Y_Ogn = Xs[0] - dx / 2, Ys[0] + dy / 2  # Upper-left corner
    # Construct TOP, BOT. TOP array: 1st layer from DS['top'], rest from DS['bottom'][::-1] with layer+1
    TOPs = np.zeros((N_L, N_R, N_C))
    TOPs[0] = DS['top'].values
    TOPs[1:] = DS['bottom'].sel(layer=range(1, N_L))
    BOTs = DS['bottom'].values  # Shape: (N_L, N_R, N_C)
    # Create full 3D grid indices
    k, i, j = np.meshgrid(range(N_L), range(N_R), range(N_C), indexing='ij')
    k, i, j = k.ravel(), i.ravel(), j.ravel()
    ### 3.0.1 Prepare GDF
    GDF_grid = gpd.GeoDataFrame(
        {
            'k': k,
            'i': i,
            'j': j,
            'node': range(N_L * N_R * N_C),
            'isfr': 1,  # All cells can potentially have SFR # if function is made out of this, this needs to be removed and added to the DF after the function has run.
            'top': TOPs.ravel(),
            'bottom': BOTs.ravel(),
        }
    )

    mask = GDF_grid['k'].eq(0)
    i_L0 = GDF_grid.loc[mask, 'i'].to_numpy()
    j_L0 = GDF_grid.loc[mask, 'j'].to_numpy()
    xmin = X_Ogn + j_L0 * dx
    xmax = X_Ogn + (j_L0 + 1) * dx
    ymin = Y_Ogn - (i_L0 + 1) * dy
    ymax = Y_Ogn - i_L0 * dy
    L0_geom = [box(x0, y0, x1, y1) for x0, y0, x1, y1 in zip(xmin, ymin, xmax, ymax)]
    for k in GDF_grid['k'].unique():
        GDF_grid.loc[GDF_grid['k'] == k, 'geometry'] = L0_geom
    GDF_grid = GDF_grid.set_geometry('geometry', crs=DS.rio.crs)

    sprint('🟢', verbose_in=True, verbose_out=M.verbose, print_time=True)

    # %% Identify deepest SFR layer
    sprint('  - Identifying deepest SFR layer.', verbose_in=True, verbose_out=M.verbose, set_time=True, end='')
    """The reason we're doing this is that the model has too many Ls and it takes a very long time to run the SFR functions with all of them. So we'll find the deepest L that has any part of the stream network in it, and **we'll only use up to that layer for the SFR grid.**"""
    for L in range(BOTs.shape[0]):
        L_BOT_min = BOTs[L].min()
        L_BOT_max = BOTs[L].max()
        sprint(L + 1, f'|{L_BOT_min:8.2f} |', f'{L_BOT_max:8.2f} |', indent=3)
        if L_BOT_min > M.lines.df['elevdn'].min():
            SFR_deepest_L = L + 1
    ### 3.0.3 Create SFR grid(s)
    SFR_grid = sfr.StructuredGrid(
        GDF_grid.loc[GDF_grid['k'] <= SFR_deepest_L - 1], crs=CRS
    )  # -1 cause grid k starts at 0, L at 1
    SFR_grid_L1 = sfr.StructuredGrid(GDF_grid.loc[GDF_grid['k'] == 0], crs=CRS)  # Extract layer 1 (k=0)
    # Check what type of object and its basic info without triggering full repr
    # print(f"Type: {type(SFR_grid)}")
    # print(f"SFR_grid object created: {SFR_grid is not None}")

    # Check if it has expensive methods for representation
    # print(f"Available methods: {[method for method in dir(SFR_grid) if not method.startswith('_')][:10]}")

    # Try to get basic info without full representation
    if M.verbose:
        print(f'Grid shape info: {hasattr(SFR_grid, "shape")}')
        if hasattr(SFR_grid, 'nlay'):
            print(f'Number of layers: {SFR_grid.nlay}')
        if hasattr(SFR_grid, 'nrow'):
            print(f'Number of rows: {SFR_grid.nrow}')
        if hasattr(SFR_grid, 'ncol'):
            print(f'Number of cols: {SFR_grid.ncol}')
    sprint('🟢', verbose_in=True, verbose_out=M.verbose, print_time=True)

    # %% SFRdata
    # paths = M.lines.paths
    SFR_data = M.lines.to_sfr(grid=SFR_grid_L1, one_reach_per_cell=True)

    # %% Explore DF_reach
    SFR_data.reach_data.sort_values(by=['i', 'j'])
    DF_reach = SFR_data.reach_data.copy()
    DF_reach[['k', 'i', 'j']] = DF_reach[['k', 'i', 'j']] + 1  # convert to 1-based indexing for reviewing
    DF_reach.describe()  # include='all')

    # %% Set default streambed thickness and hydraulic conductivity
    DF_reach['strthick'] = 0.1  # Set a default streambed thickness of 0.1 m
    DF_reach['strhc1'] = 0.1  # Set a default streambed hydraulic conductivity of 0.1 m/d

    # %% Explore width
    # DF_reach.loc[:, ['rno', 'outreach', 'iseg', 'outseg', 'node', 'k', 'i', 'j', 'name', 'rchlen', 'width', 'strtop', 'strthick', 'asum']].sort_values(by=['width', 'i', 'j'], ascending=[False, True, True])
    # I'll set all widths > 100 m to 1 m for now. #666
    # DF_reach.loc[ DF_reach['width']>100, 'width'] = 1

    # %% Assign correct layers - k.
    DF_reach[['k', 'i', 'j']] = (
        DF_reach[['k', 'i', 'j']] - 1
    )  # convert to 0-based indexing for utils_assign_layers function
    reach_Ls, strtps = sfr.utils.assign_layers(reach_data=DF_reach, botm_array=BOTs, pad=0)
    DF_reach['k'] = reach_Ls

    # %% Examples to check if segments were connected to the right cells #666 needs improvement.
    if debug_sfr:
        for i, seg in enumerate(DF_reach['name'].unique()[:10]):
            print(i + 1, seg, DF_reach.loc[DF_reach['name'] == seg, 'name'].count())
        DF_reach[['k', 'i', 'j']] = DF_reach[['k', 'i', 'j']] + 1  # convert to 1-based indexing for reviewing
        DF_reach.loc[
            DF_reach['name'] == 'OVK01451',
            [
                'rno',
                'outreach',
                'iseg',
                'outseg',
                'node',
                'k',
                'i',
                'j',
                'name',
                'rchlen',
                'width',
                'strtop',
                'strthick',
                'asum',
            ],
        ].sort_values(by=['i', 'j'])
        DF_reach.loc[
            DF_reach['name'] == 'OVK02048',
            [
                'rno',
                'outreach',
                'iseg',
                'outseg',
                'node',
                'k',
                'i',
                'j',
                'name',
                'rchlen',
                'width',
                'strtop',
                'strthick',
                'asum',
            ],
        ].sort_values(by=['name', 'j', 'i'])
        DF_reach.loc[
            DF_reach['name'] == 'OVK20466',
            [
                'rno',
                'outreach',
                'iseg',
                'outseg',
                'node',
                'k',
                'i',
                'j',
                'name',
                'rchlen',
                'width',
                'strtop',
                'strthick',
                'asum',
            ],
        ].sort_values(by=['name', 'j', 'i'])
        DF_reach[['k', 'i', 'j']] = DF_reach[['k', 'i', 'j']] - 1  # convert to 0-based indexing for SFRmaker operations

    # %% Apply RIV conductance to DF_reach
    # %% Calculate Default Conductance
    DF_RC = DF_reach.copy()[
        ['rno', 'name', 'k', 'i', 'j', 'iseg', 'outseg', 'rchlen', 'width', 'strtop', 'strthick', 'strhc1', 'asum']
    ]
    DF_RC
    DF_RC['Cond'] = DF_RC['width'] * DF_RC['rchlen'] * DF_RC['strhc1'] / DF_RC['strthick']
    # DF_RC.describe()

    # %% Import RIV Cond shapefile.

    A = imod.idf.open(M.Pa_Cond_A).sel(x=slice(M.Xmin, M.Xmax), y=slice(M.Ymax, M.Ymin))
    B = imod.idf.open(M.Pa_Cond_B).sel(x=slice(M.Xmin, M.Xmax), y=slice(M.Ymax, M.Ymin))
    # A.plot.imshow()
    # B.plot.imshow()
    # # (A>0).plot(), (B>0).plot()
    # print(f"A values >0: {(A > 1).sum().compute().values.item()} / {A.size} ({(A > 1).sum().compute().values.item() / A.size:.2%}),\nB values >0: {(B > 1).sum().compute().values.item()} / {B.size} ({(B > 1).sum().compute().values.item() / B.size:.2%})")
    C = B.where(B > 0, A)
    # D = A.where(A > 0, B)
    DF_RC['RIV_Cond'] = DF_RC[
        'Cond'
    ].copy()  # Apply conductance matching to DF_RC using array A. Start with copy of existing Cond values as fallback

    C_DF_RC = C.values[
        DF_RC['i'].values, DF_RC['j'].values
    ]  # Get array values for all i,j coordinates at once (vectorized)
    # Replace only where array has valid (non-NaN) values
    valid_mask_RC = ~np.isnan(C_DF_RC)
    DF_RC.loc[valid_mask_RC, 'RIV_Cond'] = C_DF_RC[valid_mask_RC]
    sprint(
        f'Replaced {valid_mask_RC.sum()} values out of {len(DF_RC)} total rows ({valid_mask_RC.sum() / len(DF_RC) * 100:.1f}%)',
        verbose_in=True,
        verbose_out=M.verbose,
        indent=3,
    )
    sprint(
        f'Original Cond: min={DF_RC["Cond"].min():.3f}, max={DF_RC["Cond"].max():.3f}',
        verbose_in=True,
        verbose_out=M.verbose,
        indent=3,
    )
    sprint(
        f'New RIV_Cond: min={DF_RC["RIV_Cond"].min():.3f}, max={DF_RC["RIV_Cond"].max():.3f}',
        verbose_in=True,
        verbose_out=M.verbose,
        indent=3,
    )

    # %% Check how many values actually changed
    changed_values_RC = DF_RC['Cond'] != DF_RC['RIV_Cond']
    print(f'Values that changed: {changed_values_RC.sum()} out of {len(DF_RC)}')
    DF_RC['K_RIV'] = DF_RC['RIV_Cond'] * DF_RC['strthick'] / (DF_RC['width'] * DF_RC['rchlen'])
    DF_RC['Cond_Diff'] = DF_RC['RIV_Cond'] - DF_RC['Cond']
    # DF_RC.describe()
    DF_reach['strhc1'] = DF_RC['K_RIV']  # Set it back to DF_reach
    # # %% 3.2.5 Explore segments
    # DF_Sgm = SFR_data.segment_data.copy()
    # DF_Sgm.iloc[:].describe()
    # # Most columns aren't interesting. Let's plot the interesting ones.
    # DF_Sgm[["nseg", "outseg", "roughch", "elevup", "elevdn", "width1", "width2", ]]
    # (DF_Sgm['width1'] == DF_Sgm['width1']).all()
    # (DF_Sgm['elevup'] >= DF_Sgm['elevdn']).all()
    # We can see:
    # - the roughness values are all the same (default) - **OK**
    # - downstream elevation is always lower than (or equal to) upstream - **OK**
    # - the widths seem to be the ones read from the shapefile - **OK**
    # %% Add SFR OBS
    # %% Calibration points
    DF_SFR_OBS = pd.read_csv(M.Pa_SFR_OBS_In)
    for (
        i,
        row,
    ) in DF_SFR_OBS.iterrows():  # Have to add them one by one, otherwise it groups them by reach and only keeps the 1st one. This is an SFRmaker bug, I can fix that later and make a pull request. #666 it worked for stage though, so maybe I should trty again.
        SFR_data.add_observations(
            pd.DataFrame(row).T,
            x_location_column='x',
            y_location_column='y',
            obstype_column='obstype',
            obsname_column='site_no',
        )
    # %% Stage
    DF_stage_OBS = pd.DataFrame({'rno': DF_reach['rno']})
    DF_stage_OBS['obs_name'] = (
        'Stg_L'
        + (DF_reach['k'] + 1).astype(str)
        + '_R'
        + (DF_reach['i'] + 1).astype(str)
        + '_C'
        + (DF_reach['j'] + 1).astype(str)
    )
    DF_stage_OBS['obstype'] = 'stage'
    SFR_data.add_observations(DF_stage_OBS, rno_column='rno', obstype_column='obstype', obsname_column='obs_name')
    SFR_data.observations
    # %% Run diagnostics
    SFR_data.run_diagnostics(verbose=True)
    # GDF_Elv.loc[ GDF_Elv['D'] - GDF_Elv['B_'] < 0]
    # There are fewer entries in the GDF_Elv where the DStr Elv > UStr Elv, but this DF contains segments, not reaches. So this is expected.
    # %% Write file and add to NAM
    SFR_data.reach_data = DF_reach
    SFR_data.write_package(str(M.Pa.SFR), version='mf6')
    # Try to find an inteernal SFRmaker way to fix this later. This is just a temporary patch.
    with open(M.Pa.SFR, 'r+', encoding='cp1252') as f:
        content = f.read()
        content = content.replace(f'FILEIN {M.MdlN}.SFR6.obs', f'FILEIN imported_model/{M.MdlN}.SFR6.obs')
        content = content.replace('BUDGET FILEOUT', '#BUDGET FILEOUT')
        f.seek(0)
        f.truncate()
        f.write(content)
    sh.copy2('model_SFR.chk', M.Pa.MF6 / 'imported_model/model_SFR.chk')
    with open(M.Pa.NAM_Mdl, 'r') as f1:
        l_Lns_NAM = f1.readlines()
    l_Lns_NAM.insert(-1, f'  sfr6 imported_model/{M.Pa.SFR.name} sfr\n')
    with open(M.Pa.NAM_Mdl, 'w') as f2:
        f2.writelines(l_Lns_NAM)

    return DF_reach


def Pkgs_to_SFR_via_MVR(M: Mdl_N, Pkgs: list | str, Pa_Shp: str | Path):  # 666 needs a lot of cleanup and streamlinging
    """Connects Pkg elements to the nearest SFR reach using MVR package.
    - M: Mdl_N instance.
    - Pkgs: List of package names to connect (e.g. ['DRN']). If str is provided, it will be converted to a list with one element.
    - Pa_Shp: shapefile containing the outer boundaries to clip the Pkg elements within, e.g. if only elements in a catchment need to be connected.
    """
    import re

    import WS_Mdl.core.df  # noqa: F401
    from scipy.spatial.distance import cdist
    from shapely.geometry import LineString

    # %% Load shapefile
    if Pa_Shp is not None:
        GDF_Shp = gpd.read_file(Pa_Shp)
        GDF_Shp.crs = CRS
        print(f'Loaded shapefile with {len(GDF_Shp)} features')
        print(f'Bounds: {GDF_Shp.bounds}')

    # %% Load Ins as DFs and combine them into one GDF
    d_DF = {}
    for Pkg in Pkgs:
        l_Pa = [i for i in M.Pa.Sim_In.rglob(f'*{Pkg.lower()}*.bin')]

        for Pa in l_Pa:
            PkgN = re.search(r'^.*?\d+', Pa.parent.name).group()

            DF = to_DF(Pa, Pkg=Pkg)  # Load
            DF = DF.loc[~DF['i'].isin([1, M.N_R]) & ~DF['j'].isin([1, M.N_C]), ['k', 'i', 'j']]  # Remove boundary cells
            DF = DF.ws.Calc_XY(M.Xmin, M.Ymax, M.cellsize)
            DF['Pkg1'] = PkgN
            DF['Pvd_ID'] = DF.index + 1  # 1-based index
            d_DF[PkgN] = DF

    DF_all = pd.concat(d_DF.values(), ignore_index=True)
    GDF_all = gpd.GeoDataFrame(DF_all, geometry=gpd.points_from_xy(DF_all.x, DF_all.y), crs=CRS)
    GDF = gpd.sjoin(GDF_all, GDF_Shp, how='inner', predicate='within')

    # %% Calculate distances and find closest reach for each DRN point
    DF_reach = M.DF_reach[['rno', 'i', 'j']].ws.Calc_XY(M.Xmin, M.Ymax, M.cellsize)  # Calc DF_reach XY

    coords = GDF[['x', 'y']].values
    reach_coords = DF_reach[['x', 'y']].values
    distances = cdist(coords, reach_coords, metric='euclidean')
    min_indices = np.argmin(distances, axis=1)

    # %% Add matched reach data to DRN DataFrame
    matched_reach_data = DF_reach.iloc[min_indices].reset_index(drop=True)
    DF_match = GDF.drop(columns='geometry').copy()
    DF_match['Rcv_ID'] = matched_reach_data['rno'].values
    DF_match['distance_to_match'] = distances[np.arange(len(coords)), min_indices]

    sprint(
        f'Combined {len(GDF):,} points ({Pkgs}) from {len(d_DF)} DataFrames',
        indent=2,
        verbose_in=True,
        verbose_out=M.verbose,
    )
    sprint(
        f'Matched to {DF_match["Rcv_ID"].nunique()} unique reaches', indent=2, verbose_in=True, verbose_out=M.verbose
    )
    sprint(
        f'Mean distance: {DF_match["distance_to_match"].mean():.0f}m', indent=2, verbose_in=True, verbose_out=M.verbose
    )
    sprint(
        f'Perfect matches (same cell): {(DF_match["distance_to_match"] == 0).sum():,}',
        indent=2,
        verbose_in=True,
        verbose_out=M.verbose,
    )

    # %% Plot connections
    if M.verbose:
        DF_match_plot = DF_match.merge(DF_reach, left_on='Rcv_ID', right_on='rno', suffixes=('', '_r'))
        GDF_match = gpd.GeoDataFrame(
            DF_match,
            geometry=[LineString([(r.x, r.y), (r.x_r, r.y_r)]) for _, r in DF_match_plot.iterrows()],
            crs='EPSG:28992',
        )  # Create GDF with LineString Geom

        GDF_match = GDF_match.drop(columns=['index_right', 'CATCHMENT_', 'SUM_OPPERV', 'x', 'y']).rename(
            columns={'distance_to_match': 'distance', 'Pkg1': 'Pkg'}
        )  # Trim extra Cols + rename

        for Pkg in GDF_match['Pkg'].unique():
            GDF = GDF_match.loc[GDF_match['Pkg'] == Pkg]

            Pa = M.Pa.PoP / f'In/MVR/{M.MdlN}/{Pkg}_to_SFR_{M.MdlN}.gpkg'
            Pa.parent.mkdir(parents=True, exist_ok=True)
            GDF.to_file(Pa)

    # %% Prepare DF_w for MVR
    DF_match['Pkd2'] = 'sfr'
    DF_w = DF_match[['Pkg1', 'Pvd_ID', 'Pkd2', 'Rcv_ID']]
    DF_w['MVR_TYPE'] = 'FACTOR'
    DF_w['value'] = 1

    # %% Write MVR file
    Pa_MVR = M.Pa.Sim_In / f'{M.MdlN}.MVR6'
    with open(Pa_MVR, 'w') as f:
        f.write(f"""BEGIN OPTIONS
    END OPTIONS

    BEGIN DIMENSIONS
    MAXMVR {DF_w.shape[0]}
    MAXPACKAGES {len(DF_match['Pkg1'].unique()) + 1}
    END DIMENSIONS

    BEGIN PACKAGES
    {'\n  '.join([k for k in DF_match['Pkg1'].unique()])}
    sfr
    END PACKAGES

    BEGIN PERIOD 1
    """)
        f.write(DF_w.ws.to_MF_block(indent=1))
        f.write('END PERIOD')

    # %% Insert MVR line to NAM
    with open(M.Pa.NAM_Mdl, 'r') as f1:
        l_Lns_NAM = f1.readlines()

    l_Lns_NAM.insert(-1, f'  MVR6 imported_model/{Pa_MVR.name} MVR\n')

    with open(M.Pa.NAM_Mdl, 'w') as f2:
        f2.writelines(l_Lns_NAM)

    # %% Add MOVER option to SFR
    with open(M.Pa.SFR, 'r') as f1:
        l_Lns_SFR = f1.readlines()

    l_Lns_SFR.insert(3, '  MOVER\n')

    with open(M.Pa.SFR, 'w') as f2:
        f2.writelines(l_Lns_SFR)

    # %% Add MOVER option to DRN files
    for i in DF_match['Pkg1'].unique():
        Pa = list(M.Pa.Sim_In.rglob(f'{i}*.{i[:3].lower()}'))[0]
        with open((Pa), 'r') as f1:
            l_Lns = f1.readlines()

        l_Lns.insert(3, '  MOVER\n')

        with open((Pa), 'w') as f2:
            f2.writelines(l_Lns)
