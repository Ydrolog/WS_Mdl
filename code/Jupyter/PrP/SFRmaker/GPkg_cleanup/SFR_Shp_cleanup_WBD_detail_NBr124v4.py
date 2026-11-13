# %% [markdown] 0. Imports
# ## 0. Imports
import geopandas as gpd
import imod
import numpy as np
import pandas as pd
import shapely
import WS_Mdl.core.df  # noqa: F401
import xarray as xr
from shapely.geometry import Point, box
from shapely.ops import split, unary_union
from WS_Mdl.core.mdl import Mdl_N
from WS_Mdl.core.spatial import c_Dist
from WS_Mdl.core.style import bold, style_reset

# %% [markdown] 1. Options
# ## 1. Options
MdlN = 'NBr124'
M = Mdl_N(MdlN)

detailed = 'hydroobject'
primary = 'LEGGER_VASTGESTELD_WATERLOOP_CATEGORIE_A'
Pa_Gpkg_In = M.Pa.WS / r'models\NBr\other\BrabantseDelta\acceptatiedatabase.gdb'
Pa_Out = M.Pa.WS / rf'models\NBr\In\SFR\{MdlN}\WBD_detail_SW_NW_cleaned_{MdlN}.gpkg'
l_X_Y_Cols = ['Xa', 'Ya', 'Xz', 'Yz']

# %% --------------------------------------------------
# %% [markdown] 2. Load GPkg
# ## 2. Load GPkg
DF0 = gpd.read_file(Pa_Gpkg_In, layer=detailed)  # DF is the primary Gpkg layer #111
DF0.geometry = DF0.geometry.set_precision(0.01)  # Set geometry precision to 1cm to avoid closing errors

# %% [markdown] 3. Calculate X and Y Cols.
# ## 3. Calculate X and Y Cols. Discard extra Cols. Rename.
DF1 = DF0.ws.Calc_XY_start_end_from_Geom()  # Create X & Y columns

# Keep only the Cols needed for SFRmaker and rename them
DF2 = (
    DF1[
        [
            'CODE',
            'WS_BODEMBREEDTE_L',
            'WS_BH_BOVENSTROOMS_L',
            'WS_BH_BENEDENSTROOMS_L',
            'SHAPE_Length',
            *l_X_Y_Cols,
            'geometry',
        ]
    ]
    .rename(
        columns={
            'WS_BODEMBREEDTE_L': 'width',
            'WS_BH_BOVENSTROOMS_L': 'Elv_UStr',
            'WS_BH_BENEDENSTROOMS_L': 'Elv_DStr',
            'SHAPE_Length': 'length',
        },
    )
    .copy()
)

# %% [markdown] 4. Assign unique CODEs
# ## 4. Assign unique CODEs to any rows with missing CODE values
DF2.loc[DF2['CODE'].isnull(), 'CODE'] = [f'dummy_{i}' for i in range(1, DF2['CODE'].isnull().sum() + 1)]


# %% --------------------------------------------------
# %% [markdown] 5. Split Polylines
# ## 5. Split Polylines
def split_lines_by_neighbors(GDF: gpd.GeoDataFrame, Tol: float = 0.000001) -> gpd.GeoDataFrame:
    """
    Split each (Multi)LineString where it meets OTHER features in gdf.
    Uses a spatial index so only intersecting neighbors are used as splitters.
    Returns one row per split segment, attributes copied from the original row.
    Tol: tolerance in CRS units (e.g. meters).
    """

    def flatten_Geom(geom):
        """Recursively flatten GeometryCollections and Multi-geometries into simple geometries."""
        if hasattr(geom, 'geoms'):
            for g in geom.geoms:
                yield from flatten_Geom(g)
        else:
            yield geom

    GDF = GDF.copy()
    other_cols = [c for c in GDF.columns if c != 'geometry']
    GDF['Init_Geom'] = GDF['geometry'].copy()  # Store original geometries

    # Snap Coordinates so near-coincident nodes match (for spatial index and splitting)
    if hasattr(shapely, 'set_precision'):
        GDF['geometry'] = shapely.set_precision(GDF['geometry'].array, Tol)

    new_rows = []

    for i, R in GDF.iterrows():
        Geom = R['geometry']
        Init_Geom = R['Init_Geom']
        Attrs = R.to_dict()

        Ngb_i = list(GDF.sindex.query(Geom, predicate='intersects'))  # Neighbors potentially intersectng Geom
        Ngb_i = [j for j in Ngb_i if j != i]  # remove itself

        if not Ngb_i:  # No neighbors - use original unsnapped geometry
            # Flatten in case original was MultiLineString
            for sub_seg in flatten_Geom(Init_Geom):
                new_rows.append({**Attrs, 'geometry': sub_seg})
            continue

        Ngb_Geom = GDF.loc[Ngb_i, 'geometry'].values

        splitter = unary_union(Ngb_Geom)  # build splitter from neighbors

        # NEW: remove overlapping segments so split() only sees crossing/touching parts
        splitter = splitter.difference(Geom)

        if splitter.is_empty or not Geom.intersects(splitter):
            # Has neighbors but no actual split - use original unsnapped geometry
            # Flatten in case original was MultiLineString
            for sub_seg in flatten_Geom(Init_Geom):
                new_rows.append({**Attrs, 'geometry': sub_seg})
            print(R['CODE'], 'no split')
            continue

        parts = split(Geom, splitter)

        if len(parts.geoms) == 1:  # Those that weren't split
            for sub_seg in flatten_Geom(
                Init_Geom
            ):  # Use original unsnapped geometry, but flatten in case original was MultiLineString
                new_rows.append({**Attrs, 'geometry': sub_seg})
        else:  # Actually split
            for seg in parts.geoms:
                for sub_seg in flatten_Geom(seg):
                    new_rows.append({**Attrs, 'geometry': sub_seg})

    # Return without _original_geom column
    return gpd.GeoDataFrame(new_rows, columns=other_cols + ['geometry'], crs=GDF.crs)


DF3 = split_lines_by_neighbors(DF2, Tol=0.01)

# %% [markdown] 6. QA Split
# ## 6. QA Split
# %% 6.1 Ensure coordinates are preserved for non-split features
DF3_codes = DF3['CODE'].value_counts()
non_split_codes = DF3_codes[DF3_codes == 1].index  # Codes that weren't split

DF3_non_split = DF3[DF3['CODE'].isin(non_split_codes)].copy()
DF2_matching = DF2[DF2['CODE'].isin(non_split_codes)].copy()

DF2_matching = DF2_matching.set_index('CODE')
DF3_non_split = DF3_non_split.set_index('CODE')

# Check coordinate differences
Coo_diff = 0
for code in non_split_codes:
    if code in DF2_matching.index and code in DF3_non_split.index:
        g3 = DF2_matching.loc[code, 'geometry']
        g4 = DF3_non_split.loc[code, 'geometry']

        # Extract start and end coordinates properly for both LineString and MultiLineString
        if g3.geom_type == 'LineString':
            c3 = (g3.coords[0], g3.coords[-1])
        else:  # MultiLineString
            c3 = (g3.geoms[0].coords[0], g3.geoms[-1].coords[-1])

        if g4.geom_type == 'LineString':
            c4 = (g4.coords[0], g4.coords[-1])
        else:  # MultiLineString
            c4 = (g4.geoms[0].coords[0], g4.geoms[-1].coords[-1])

        if c3 != c4:
            Coo_diff += 1

print(f'Total non-split features: {len(non_split_codes)}')
print(f'Non-split features with changed coordinates: {Coo_diff}')
print(f'Total features in DF2: {len(DF2)}')
print(f'Total features in DF3: {len(DF3)}')
print(f'DF3 features that were split: {len(DF3) - len(non_split_codes)}')

# %% 6.2 Create Col to identify split segments
l_X_Y_Cols_ = [i + '_' for i in l_X_Y_Cols]
l_Cols1 = sorted(['CODE', 'length', *l_X_Y_Cols, *l_X_Y_Cols_, 'geometry'])
DF3[l_X_Y_Cols_] = DF3.geometry.apply(
    lambda g: pd.Series(
        g.geoms[0].coords[0] + g.geoms[-1].coords[-1]
        if g.geom_type.startswith('Multi')
        else g.coords[0] + g.coords[-1],
        index=l_X_Y_Cols_,
    )
)
DF3['split'] = 0  # Set blank split column
DF3_split = DF3.loc[DF3['CODE'].isin(DF3['CODE'].value_counts()[DF3['CODE'].value_counts() > 1].index)]
DF3.loc[
    DF3['CODE'].isin(DF3['CODE'].value_counts()[DF3['CODE'].value_counts() > 1].index),
    'split',
] = 1  # Set to 1 for split segments
DF3_Coo_change = DF3.loc[
    (DF3['Xa'].round(3) != DF3['Xa_'].round(3))  # Only contains rows where coordinates have changed
    | (DF3['Ya'].round(3) != DF3['Ya_'].round(3))
    | (DF3['Xz'].round(3) != DF3['Xz_'].round(3))
    | (DF3['Yz'].round(3) != DF3['Yz_'].round(3))
]
DF3_split.shape, DF3_Coo_change.shape, DF3.shape

# %% 6.3 Ensure that no coordinates were changed for non-split features
shift = DF3_Coo_change.loc[~DF3_Coo_change['CODE'].isin(DF3_split['CODE']), l_Cols1]
shift.shape

# %% 6.4 Identify rows in DF3_split but not in DF3_Coo_change
DF3_split_no_Coo_change = DF3_split.loc[~DF3_split.index.isin(DF3_Coo_change.index)].copy()
print(f'Rows in DF3_split but not in DF3_Coo_change: {len(DF3_split_no_Coo_change)}')
print(f'\nUnique CODEs: {DF3_split_no_Coo_change["CODE"].unique()}')
DF3_split_no_Coo_change[l_Cols1]


# %% [markdown]
# A few entries seem to have had their coordinates changed for one of the split sub-segments but not for the other. In a lot of cases, there are multiple lines in a polyline with the exact same coordinates. SFRmaker should be able to handle this, so there will be no futher check at this point.

# %% --------------------------------------------------
# %% [markdown] 7. Recalc/assign some columns
# ## 7. Recalc/assign some columns

# %% [markdown] 7.1 Length
# ### 7.1 Length
DF3.loc[DF3['CODE'] == 'OVK03536']  # Plot a split segment to inspect length

# %% [markdown]
# Length needs to be recalculated after the split, based on the new geometries. As you can see above, all lengths for split segments are currently the same as the original segment length.

# %% Recalculte length based on the new geometries
DF4 = DF3.copy()
DF4['length'] = DF4.geometry.length

# %% Check recalculated lengths for a split segment
(
    DF4.loc[DF3['CODE'] == 'OVK03536', 'length'].sum().round(3),
    DF3.loc[DF3['CODE'] == 'OVK03536', 'length'].iloc[0].round(3),
)

# %% [markdown]
# The length sum isn't exactly identical, but I'll move forward for now. #666 can come back to this later.

# %% --------------------------------------------------
# %% [markdown] 7.2 Coordinates
# ### 7.2 Coordinates
Bool_Coo_changed = DF4[l_X_Y_Cols].sum(axis=1) != DF4[l_X_Y_Cols_].sum(axis=1)
DF4.loc[Bool_Coo_changed, l_X_Y_Cols] = DF4.loc[Bool_Coo_changed, l_X_Y_Cols_].values
(DF4.loc[:, l_X_Y_Cols].values == DF4.loc[:, l_X_Y_Cols_].values).all()

# %% [markdown]
# All values from the newly calculated coordinates have been assigned to the main GDF.

# %% Drop extra coordinate columns
DF4.drop(columns=l_X_Y_Cols_, inplace=True)

# %% --------------------------------------------------
# %% [markdown] 7.3 Width
# ### 7.3 Width
# %% Investigate
DF4.describe(include='all')

# %% Set width to 1m where missing
# Width will be set to 1m where missing. This is very simplistic, but a good start. #666
DF4.loc[DF4['width'].isnull(), 'width'] = 1.0

# %% --------------------------------------------------
# %% [markdown] 7.4 Elv: Fill NaNs from TOP/BOT
# ### 7.4 Elv: Fill NaNs from TOP/BOT
# %% [markdown]
# Elevations will be assigned based on the thickest layer from 0.75-1.5m from the surface - standard drainage (ditch) depth in the Netherlands.

# %% Load TOP and BOT IDFs
l_TOP = [i for i in (M.Pa.WS / 'models/NBr/In/TOP').glob('*.idf')]
l_BOT = [i for i in (M.Pa.WS / 'models/NBr/In/BOT').glob('*.idf')]

TOP = imod.formats.idf.open(l_TOP, pattern='{name}_L{layer}_').interp(x=M.Xs, y=M.Ys, method='linear')
BOT = imod.formats.idf.open(l_BOT, pattern='{name}_L{layer}_').interp(x=M.Xs, y=M.Ys, method='linear')
Thk = TOP - BOT

# %% Isolate rows with missing Elv_UStr or Elv_DStr. Then print those where one is missing but the other is not.
DF4_UStr_NA = DF4.loc[DF4['Elv_UStr'].isna()]
DF4_DStr_NA = DF4.loc[DF4['Elv_DStr'].isna()]

print(DF4.shape, DF4_DStr_NA.shape, DF4_UStr_NA.shape)
DF4.loc[(DF4['Elv_UStr'].isna() & DF4['Elv_DStr'].notna()) | (DF4['Elv_UStr'].notna() & DF4['Elv_DStr'].isna())]

# %% [markdown]
# DF4_DStr_NA & DF4_UStr_NA have different number of entries. There are some rows where one of Elv_UStr or Elv_DStr are available, but the other one is missing. We can use those as an indicator of how well the filling worked.


# %% Def function to fill missing
def rule_based_3ry_Elv(A_Xas, A_Yas, TOP, BOT, Drn_min_depth=0.75, Drn_max_depth=1.5):
    """
    Selects the layer where the drainage elevation (0.75m to 1.5m below surface) falls within the layer.
    If multiple layers are valid, selects the one with the maximum thickness within the valid range.
    A_Xas, A_Yas: xarray DataArrays of point coordinates (dims='points')
    TOP, BOT: xarray DataArrays of model TOP and BOT elevations (dims='layer', 'x', 'y')
    Returns: numpy array of selected elevations for each point
    """
    # Ensure inputs are xarray DataArrays with a common dimension 'points'
    # This prevents errors when passing pandas Series directly to .sel()
    if not isinstance(A_Xas, xr.DataArray):
        A_Xas = xr.DataArray(A_Xas, dims='points')
    if not isinstance(A_Yas, xr.DataArray):
        A_Yas = xr.DataArray(A_Yas, dims='points')

    # Slice DF coordinates for points where
    TOP_Pts = TOP.sel(x=A_Xas, y=A_Yas, method='nearest')
    BOT_Pts = BOT.sel(x=A_Xas, y=A_Yas, method='nearest')
    Surface = TOP_Pts.sel(layer=1)

    # Get valid Ls and valid Thk - within drainage elevation range
    Drn_Elv_max = Surface - Drn_min_depth
    Drn_Elv_min = Surface - Drn_max_depth
    valid_Ls = (
        ((Drn_Elv_max >= TOP_Pts) & (TOP_Pts > Drn_Elv_min))
        | ((Drn_Elv_max > BOT_Pts) & (BOT_Pts >= Drn_Elv_min))
        | ((TOP_Pts >= Drn_Elv_max) & (BOT_Pts <= Drn_Elv_min))
    )

    Thk_Pts = TOP_Pts - BOT_Pts
    valid_Thk = Thk_Pts.where(valid_Ls, -1.0)  # Mask invalid layers (set thickness to -1 so they are not picked)

    # Find index of max thickness
    best_layer_idx = valid_Thk.argmax(dim='layer')

    # Compute the index if it's lazy (dask) to avoid "vindex does not support indexing with dask objects"
    if hasattr(best_layer_idx, 'compute'):
        best_layer_idx = best_layer_idx.compute()

    # Select best layer TOP and BOT (based on Thk)
    TOP_best = TOP_Pts.isel(layer=best_layer_idx)
    BOT_best = BOT_Pts.isel(layer=best_layer_idx)

    # Result: Middle of the layer
    Elv = (TOP_best + BOT_best) / 2

    return Elv.values


# %% Apply to Elv_UStr
mask_U = DF4['Elv_UStr'].isna()
if mask_U.any():
    print(f'Filling {mask_U.sum()} NaNs in Elv_UStr')
    DF4.loc[mask_U, 'Elv_UStr'] = rule_based_3ry_Elv(DF4.loc[mask_U, 'Xa'], DF4.loc[mask_U, 'Ya'], TOP, BOT)

# %% Apply to Elv_DStr
mask_D = DF4['Elv_DStr'].isna()
if mask_D.any():
    print(f'Filling {mask_D.sum()} NaNs in Elv_DStr')
    DF4.loc[mask_D, 'Elv_DStr'] = rule_based_3ry_Elv(DF4.loc[mask_D, 'Xz'], DF4.loc[mask_D, 'Yz'], TOP, BOT)

# %% Confirm no NaNs remain
DF4.loc[DF4['Elv_UStr'].isna() | DF4['Elv_DStr'].isna()].ws.clip_Mdl_area(M=M)

# %% [markdown]
# Checking in QGIS confirms that the Elvs were assigned properly.

# %% --------------------------------------------------
# %% [markdown] 7.5. Calculate routing
# ### 7.5. Calculate routing
# %% Create ID Col & Identify downstream
DF4['ID'] = range(1, len(DF4) + 1)
COORD_PRECISION = 0.05


def snap_coordinate(value):
    return round(value / COORD_PRECISION) * COORD_PRECISION


start_Coo_to_rows = {}
for R in DF4.itertuples():
    key = (snap_coordinate(R.Xa), snap_coordinate(R.Ya))
    start_Coo_to_rows.setdefault(key, []).append((R.CODE, R.ID))
print(f'🟢 Lookup dictionary created with {bold}{len(start_Coo_to_rows)}{style_reset} entries.')


# %% Define function to get DStr_code and DStr_ID based on end coordinates
def get_DStr(row):
    end_Coo = (snap_coordinate(row.Xz), snap_coordinate(row.Yz))
    candidates = start_Coo_to_rows.get(end_Coo, [])
    same_code = [candidate for candidate in candidates if candidate[0] == row.CODE and candidate[1] != row.ID]
    other_code = [candidate for candidate in candidates if candidate[1] != row.ID]
    return (same_code or other_code or [(0, 0)])[0]


# %% Apply the function to create the 'DStr' column
DF4[['DStr_code', 'DStr_ID']] = DF4.apply(get_DStr, axis=1, result_type='expand')
print("🟢 'DStr' columns calculated.")
print(
    f'{round(DF4["DStr_code"].value_counts().max() / DF4.shape[0] * 100, 2)} % of DStrs are 0 (i.e. no start Coos match the end Coos of the current node).'
)

# %% Calculate UStr_IDs for each ID
UStr = DF4.groupby('DStr_ID')['ID'].agg(list)
DF4['UStr_IDs'] = DF4['ID'].map(UStr).apply(lambda ids: ids if isinstance(ids, list) else [])
print("🟢 'UStr_IDs' column calculated.")
print(f'{round((DF4["UStr_IDs"].str.len() > 0).sum() / DF4.shape[0] * 100, 2)} % of segments have upstream IDs.')

# %% [markdown]
# We'll drop short segments only when they have neither an upstream nor a downstream segment.

# %% Drop short segments with no upstream or downstream segment.
has_UStr = DF4['UStr_IDs'].str.len().gt(0)
has_DStr = DF4['DStr_ID'].ne(0)
DF4 = DF4.loc[~((DF4['length'] < 1) & ~has_UStr & ~has_DStr)].copy()

print("🟢 'DStr' columns calculated.")
print(
    f'{round(DF4["DStr_code"].value_counts().max() / DF4.shape[0] * 100, 2)} % of DStrs are 0 (i.e. no start Coos match the end Coos of the current node).'
)


# %% [markdown]
# #### Connect to closest end, including intermediate segments
def connect_intermediary_same_cell(DF4):
    """
    Connect segments without a DStr match to the nearest line crossing the
    model cell containing their downstream endpoint.
    """
    DF4 = DF4.copy()
    unresolved = DF4['DStr_ID'].fillna(0).eq(0)  # Filter for unconnected
    if not unresolved.any():
        return DF4

    for i, row in DF4.loc[unresolved].iterrows():
        C = int(np.floor((row['Xz'] - M.Xmin) / M.cellsize))
        R = int(np.floor((M.Ymax - row['Yz']) / M.cellsize))
        cell = box(
            M.Xmin + C * M.cellsize,
            M.Ymax - (R + 1) * M.cellsize,
            M.Xmin + (C + 1) * M.cellsize,
            M.Ymax - R * M.cellsize,
        )  # Define cell geometry

        current_position = DF4.index.get_loc(i)
        candidate_indices = list(DF4.sindex.query(cell, predicate='intersects'))  # make list of intersecting indices
        candidate_indices = [j for j in candidate_indices if j != current_position]  # exclude itself
        if not candidate_indices:
            continue

        candidates = DF4.iloc[candidate_indices]
        same_code_candidates = candidates.loc[candidates['CODE'] == row['CODE']]
        if not same_code_candidates.empty:
            candidates = same_code_candidates

        candidate_distances = candidates['geometry'].distance(
            Point(row['Xz'], row['Yz'])
        )  # Calculate distances to preferred candidates
        candidate_index = candidate_distances.sort_values(kind='stable').index[
            0
        ]  # Select closest. Stable makes sorting deterministic according to existing order
        candidate = DF4.loc[candidate_index]

        DF4.loc[i, 'DStr_code'] = candidate['CODE']
        DF4.loc[i, 'DStr_ID'] = candidate['ID']

    UStr = DF4.groupby('DStr_ID')['ID'].agg(list)
    DF4['UStr_IDs'] = DF4['ID'].map(UStr).apply(lambda ids: ids if isinstance(ids, list) else [])
    return DF4


DF5 = connect_intermediary_same_cell(DF4)

print("🟢 'DStr' columns calculated.")
print(
    f'{round(DF5["DStr_code"].value_counts().max() / DF5.shape[0] * 100, 2)} % of DStrs are 0 (i.e. no start Coos match the end Coos of the current node).'
)

# %% [markdown]
# This reduced the DStr=0 to 1.35%. Let's review them.

# %% First check how many are connected to the same ID after fix
Pa_Out_DF5 = M.Pa.In / 'SFR' / M.MdlN / f'WBD_detail_SW_NW_cleaned_DF5_{M.MdlN}.gpkg'
Pa_Out_DF5.parent.mkdir(parents=True, exist_ok=True)
DF5.to_file(Pa_Out_DF5, layer='WBD_detail_SW_NW_cleaned_v4', driver='GPKG')
DF5['DStr_ID'].value_counts()

# %% [markdown]
# No more than 5 segments are connected to the same DStr_ID - reasonable.

# %% Calculate min distance from start to any reach's end and investigate no matches.
DF5['min_Dist'] = np.nan
DF5['closest_ID'] = pd.Series(pd.NA, index=DF5.index, dtype='Int64')


def get_closest_end(row):
    candidates = DF5.loc[DF5['ID'] != row['ID']]
    distances = c_Dist(row['Xz'], row['Yz'], candidates['Xa'], candidates['Ya'])
    closest_index = distances.idxmin()
    return distances.loc[closest_index], candidates.loc[closest_index, 'ID']


unresolved = DF5['DStr_code'] == 0
closest_results = DF5.loc[unresolved].apply(
    get_closest_end,
    axis=1,
    result_type='expand',
)
DF5.loc[unresolved, 'min_Dist'] = closest_results.iloc[:, 0].to_numpy(dtype=float)
DF5.loc[unresolved, 'closest_ID'] = closest_results.iloc[:, 1].to_numpy(dtype='int64')
DF5['closest_ID'] = DF5['closest_ID'].astype('Int64')

l_Vals = [0.001, 0.1, 1, 10, 100, 1000, 10000]


def print_segments_per_min_Dist(DF, l_Vals):
    d_Vals = {}  # to store the counts for each range
    N_no_match = (DF['DStr_code'] == 0).sum()
    DF.loc[DF['DStr_code'] == 0, 'min_Dist'].describe()

    print(f'Out of the {N_no_match} segments that do not match:')

    N_below_Prv, Val_Prv, P_below_Prv = 0, 0, 0
    for v in l_Vals:
        N_below = (DF.loc[DF['DStr_code'] == 0, 'min_Dist'] <= v).sum()
        P_below = round(N_below / N_no_match * 100, 2)

        code_min_Dist = (
            DF.loc[
                (DF['DStr_code'] == 0) & (DF['min_Dist'] > Val_Prv) & (DF['min_Dist'] <= v),
                ['ID', 'min_Dist'],
            ]
            .sort_values(by='min_Dist')
            .apply(lambda R: f'{R["ID"].astype(int)}: {R["min_Dist"]:8.4f}', axis=1)
            .tolist()
        )

        print(
            f'-{Val_Prv:6} < min_Dist <= {v:5} |N: {N_below:6} (+ {(N_below - N_below_Prv):4}) ({round(P_below, 1):5} %) | Codes: {code_min_Dist}\n'  # [:10] + code_min_Dist[-10:]}\n'
        )

        d_Vals[v] = {
            'range': f'{Val_Prv} - {v}',
            'count': N_below - N_below_Prv,
            'count_cumulative': N_below,
            'percentage': P_below - P_below_Prv,
            'P_cumulative': P_below,
            'example codes': code_min_Dist[:5] + code_min_Dist[-5:],
        }

        N_below_Prv, Val_Prv = N_below, v

    return d_Vals


d_min_Dist = print_segments_per_min_Dist(DF5, l_Vals)

# %% [markdown]
# The total number of non matching segments has decreased a lot. They're still too many to manually connect. But if the splitting algortithm is improved, so it splits lines when they cross, it may drop even more.
# - The ones <1m can be attributed to closing errors, and we can connect them to the closest one via an algorithm.


# %% Review segments in model area
DF5_w = DF5.loc[DF5.within(box(M.Xmin, M.Ymin, M.Xmax, M.Ymax))]

d_min_Dist_within = print_segments_per_min_Dist(DF5_w, l_Vals)

# %% [markdown] Conclusion
# A QGIS review of segments follows:<br>
# ||: connecting to closest cell end within 25m (outside of cell) fixes it.<br>
# ~||: same as above, but with caveats<br>
# NP: No problem if we lose it. Maily for UStr-most segments with smallish width.<br>
#
# ---
# 10 - 100 m:
# -  6296 (21.8 m): barely outside the cell boundary of another segment's end. ||
# -  6167 (23.4 m): ||
# - 48451 (26.4 m): ~||: Closest end 26 m away (25m cutoff wouldn't work). Honestly, the ditch direction seems reversed.
# - 35958 (26.6 m): " " "
# - 24103 (28.9 m): ~||: Closest not the right one, but no big Diff, especially since it is the UStr-most segment.
# - 59124 (31.2 m): ~||
# - 35068 (31.2 m): ~||
# - 18900 (31.7 m): NP
# - 54121 (44.7 m): ~||
# - 13213 (45.1 m): NP
# -  6065 (48.2 m): ~||
# - 16493 (58.2 m): ~||: Not small, not UStr-most. Would lose decent chunch of water if not connected.
# - 45225 (60.8 m): NP: UStr-most. Connectin to closest end would be incorrect, but no big deal.
# - 18221 (79.5 m): NP: Not small, but UStr-most, and looks like some weird ditch into urban chanel configuration.
# - 42857 (85.7 m): NP
# - 16690 (96.9 m): Lake
# ---
# 100 - 1000 m:
# -  5298 (139.8 m): NP: Reversed. Min_Dist~=length
# - 16978 (145.6 m): Lake (tiny, pond)
# -  4500 (147.0 m): Lake
# - 16691 (157.6 m): Lake
# - 16979 (163.7 m): Lake
# - 20521 (171.2 m): NP: Reverse. Long, but UStr-most.
# - 19829 (184.2 m): NP: Should be connected as it's not UStr-most, but it's not clear from QGIS which should be it's DStr. And UStr area is not that big.
# - 17255 (206.7 m): Lake
# -  6037 (245.3 m): Lake
# - 12349 (257.9 m): Lake
# - 20521 (171.2 m): NP: Reverse. Long, but UStr-most.
# - 35129 (264.5 m): Lake
# -  1563 (302.9 m): NP: Cant' determine DStr from QGIS.
# - 17365 (328.3 m): Lake
# - 45190 (427.9 m): Lake
# - 16390 (745.8 m): Lake

# Conclusion: UStr-most segments don't need to be connected - they're either lakes, or we don't lose much water.
# But, those that have UStr should be connected to the closest within 100m even IMO.


# %% [markdown] 7.6. Assign unconnected with UStr_IDs to closest end within 100m
# ### 7.6. Assign unconnected with UStr_IDs to closest end within 100m
DF6 = DF5.copy()

DF6.loc[(DF6['DStr_ID'] == 0) & (DF6['UStr_IDs']) & (DF6['min_Dist'] < 100), 'DStr_ID'] = DF6.loc[
    (DF6['DStr_ID'] == 0) & (DF6['UStr_IDs']) & (DF6['min_Dist'] < 100), 'closest_ID'
]
print(
    'Percentage of segments with DStr_ID=0 after connecting to closest end within 100m:',
    round((DF6['DStr_ID'] == 0).sum() / len(DF6) * 100, 2),
    '%',
)

# %% [markdown]
# The percentage is now down to 1.05%. Based on the applied rules, we've connected all non UStr-most segments. The rule is not extended further than 100m or segments without UStr_IDs, because then segments that aren't really connected (e.g. lates, or ditces that aren't connected) might be connected incorrectly. The error is most likey less than 1% in this case, which we consider acceptable for the use of this method.<br><br>
# Outlets will be further reviewed in QGIS when an SFR packae will be created from this GPkg.

# %% --------------------------------------------------

# %% [markdown] 7.7 Correct Elvs for split segments (1ry SW Nw)
# ### 7.7 Correct Elvs for split segments (1ry SW Nw)
# %% Prep & review
l_DF7_Cols = [
    'CODE',
    'DStr_code',
    'ID',
    'DStr_ID',
    'UStr_IDs',
    'width',
    'length',
    'Elv_UStr',
    'Elv_DStr',
    'Xa',
    'Ya',
    'Xz',
    'Yz',
    'geometry',
    'split',
]  # Order columns logically.

l_success, l_infinite_loop, l_closing_error = [], [], []
DF7 = DF6.copy()[l_DF7_Cols]

# %% [markdown]
# Elvs were inherited from the original segments when they were split. This is incorrect, except for the UStr_Elv of the UStr-most segment and the DStr_Elv of the DStr-most segment, of every CODE.
# But, in DF6 (therefore DF7 as well), NA Elvs were filled for split segments. So those don't need to be corrected.

# %% [markdown] Create original-segment elevation lookup
# DF6 contains the final split segments. Reconstruct missing original endpoint
# elevations from the filled split-segment elevations without modifying DF2.
DF2_corr = DF2.copy()
DF2_corr_Elv_NaN_codes = DF2_corr.loc[
    DF2_corr['Elv_UStr'].isna() | DF2_corr['Elv_DStr'].isna(), 'CODE'
].unique()  # Get codes where Elv is missing.

DF6_corr = DF6.loc[
    DF6['CODE'].isin(DF2_corr_Elv_NaN_codes),
    ['CODE', 'Elv_UStr', 'Elv_DStr'],
]  # Slice DF
DF6_Elv_UStr_max = DF6_corr.groupby('CODE')['Elv_UStr'].max()  # Crete mapping for max UStr_Elv
DF6_Elv_DStr_min = DF6_corr.groupby('CODE')['Elv_DStr'].min()

DF2_corr.loc[DF2_corr['CODE'].isin(DF2_corr_Elv_NaN_codes), 'Elv_UStr'] = DF2_corr.loc[
    DF2_corr['CODE'].isin(DF2_corr_Elv_NaN_codes), 'CODE'
].map(DF6_Elv_UStr_max)  # Use Elv_UStr from the split segments
DF2_corr.loc[DF2_corr['CODE'].isin(DF2_corr_Elv_NaN_codes), 'Elv_DStr'] = DF2_corr.loc[
    DF2_corr['CODE'].isin(DF2_corr_Elv_NaN_codes), 'CODE'
].map(DF6_Elv_DStr_min)

# %% [markdown] Correct Elvs
n = 1
Elv_NA_and_split_codes = (~DF6['CODE'].isin(DF3.loc[DF3['Elv_UStr'].isna() & DF3['Elv_DStr'].isna(), 'CODE'])) & (
    DF6['split'] == 1
)


for i in DF6.loc[Elv_NA_and_split_codes, 'CODE'].unique():  # [::-1]: # [26:27]: #[n:n+1]:
    print(
        f'{i} - {n:4}/{len(DF6.loc[Elv_NA_and_split_codes, "CODE"].unique()):4}',
        end=' - ',
    )
    n += 1

    # Prep GDF and Se for CODE
    GDF = DF6.loc[DF6['CODE'] == i, l_DF7_Cols]  # Create GDF for the current CODE
    Se = DF2_corr.loc[DF2_corr['CODE'] == i]
    Coo_a = (snap_coordinate(Se['Xa'].values[0]), snap_coordinate(Se['Ya'].values[0]))

    # Prepare for Elv_DStr adjustment
    drop = Se['Elv_UStr'].values[0] - Se['Elv_DStr'].values[0]
    length = GDF['length'].sum()
    GDF['C'] = GDF['length'] / length
    GDF_IDs = set(GDF['ID'])

    # Iterate UStr to DStr and edit Elvs
    dx = GDF['Xa'].map(snap_coordinate) - Coo_a[0]
    dy = GDF['Ya'].map(snap_coordinate) - Coo_a[1]
    idx = np.hypot(dx, dy).idxmin()
    ID_UStr = GDF.loc[[idx], 'ID'].values[0]  # or GDF.loc[idx] for a Series

    # Initialize Elv_UStr for the first segment
    GDF.loc[GDF['ID'] == ID_UStr, 'Elv_UStr'] = Se['Elv_UStr'].values[0]

    ID = ID_UStr
    visited_IDs = set()
    while True:
        if ID in visited_IDs:
            print(f'Stuck at ID {ID}. Breaking loop to avoid infinite iteration.')
            l_infinite_loop.append(i)
            break
        visited_IDs.add(ID)

        GDF.loc[GDF['ID'] == ID, 'Elv_DStr'] = (
            GDF.loc[GDF['ID'] == ID, 'Elv_UStr'].values[0] - drop * GDF.loc[GDF['ID'] == ID, 'C'].values[0]
        )

        next_ID = GDF.loc[GDF['ID'] == ID, 'DStr_ID'].iloc[0]
        if pd.isna(next_ID) or next_ID == 0 or next_ID not in GDF_IDs:
            break

        ID_prev = ID
        ID = next_ID
        GDF.loc[GDF['ID'] == ID, 'Elv_UStr'] = GDF.loc[GDF['ID'] == ID_prev, 'Elv_DStr'].values[0]

    # Check
    expected_DStr = DF2_corr.loc[DF2_corr['CODE'] == i, 'Elv_DStr'].values[0]
    calculated_DStr = GDF.loc[GDF['ID'] == ID, 'Elv_UStr'].values[0] - drop * GDF.loc[GDF['ID'] == ID, 'C'].values[0]
    check = np.isclose(round(expected_DStr, 3), round(calculated_DStr, 3), atol=0.0)
    if not check:
        print(f'Discrepancy in Elv_DStr for CODE {i}: calculated {calculated_DStr}, expected {expected_DStr}')
        l_closing_error.append(i)
    else:
        print('Elvs calculated correctly.')
        l_success.append(i)
        DF7.loc[DF7['CODE'] == i, 'Elv_UStr'] = GDF['Elv_UStr'].values
        DF7.loc[DF7['CODE'] == i, 'Elv_DStr'] = GDF['Elv_DStr'].values

Tot = len(l_success) + len(l_infinite_loop) + len(l_closing_error)
print(f'Successful: {len(l_success)} ({round(len(l_success) * 100 / Tot, 2)} %)')
print(f'Infinite loop: {len(l_infinite_loop)} ({round(len(l_infinite_loop) * 100 / Tot, 2)} %)')
print(f'Closing error: {len(l_closing_error)} ({round(len(l_closing_error) * 100 / Tot, 2)} %)')

# %% [markdown]
# 99% are successful. Acceptable.

# %% [markdown]
# Inspect Out path
Pa_Out

# %% [markdown] Save
Pa_Out.parent.mkdir(parents=True, exist_ok=True)
DF7_out = DF7.copy()
DF7_out['DStr_code'] = DF7_out['DStr_code'].astype('string')
DF7_out['DStr_ID'] = DF7_out['DStr_ID'].astype('Int64')
DF7_out['split'] = DF7_out['split'].astype('Int64')
DF7_out = DF7_out[
    [
        'ID',
        'DStr_ID',
        'width',
        'length',
        'Elv_UStr',
        'Elv_DStr',
        'Xa',
        'Ya',
        'Xz',
        'Yz',
        'geometry',
        'CODE',
        'DStr_code',
        'split',
    ]
].copy()
if Pa_Out.exists():
    Pa_Out.unlink()
DF7_out.to_file(Pa_Out, driver='GPKG', engine='fiona', index=False)


# %% [markdown] DESCRIPTION OF DF FORMS:
# DF0: As loaded from OG GPkg
# DF1: X & Y calculated
# DF2: Selected & renamed Cols.
# DF3: Split segments
# DF4: Recalc lengh & coordinates. Fill Elv from TOP/BOT. Calculate DStr_ID & UStr_IDs.
# DF5: Connect segments without DStr_ID to closest end within same cell.
# DF6: Connect segments without DStr_ID to closest end within 100m if they have UStr_IDs.
# DF7: Correct Elvs for split segments.
