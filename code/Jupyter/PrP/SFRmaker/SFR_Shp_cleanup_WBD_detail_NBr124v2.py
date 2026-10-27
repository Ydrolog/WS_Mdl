# region Imports
# %%
import geopandas as gpd
import imod
import numpy as np
import pandas as pd
import shapely
import WS_Mdl.core.df  # noqa: F401
import xarray as xr
from shapely.geometry import box
from shapely.ops import split, unary_union
from WS_Mdl.core.mdl import Mdl_N
from WS_Mdl.core.spatial import c_Dist
from WS_Mdl.core.style import bold, style_reset

# endregion

# region 1. Options
# %%
MdlN = 'NBr124'
M = Mdl_N(MdlN)

detailed = 'hydroobject'
primary = 'LEGGER_VASTGESTELD_WATERLOOP_CATEGORIE_A'
Pa_Gpkg_In = M.Pa.WS / r'models\NBr\other\BrabantseDelta\acceptatiedatabase.gdb'
Pa_Out = M.Pa.WS / rf'models\NBr\In\SFR\{MdlN}\WBD_detail_SW_NW_cleaned_{MdlN}.gpkg'
l_X_Y_Cols = ['Xa', 'Ya', 'Xz', 'Yz']
# endregion

# region 2. Load GPkg
# %%
A0 = gpd.read_file(Pa_Gpkg_In, layer=detailed)  # A is the primary Gpkg layer #111
A0.geometry = A0.geometry.set_precision(0.01)  # Set geometry precision to 1cm to avoid closing errors
# endregion

# region 3. Calculate X and Y Cols, & discard extra Cols
# %%
A1 = A0.ws.Calc_XY_start_end_from_Geom()  # Create X & Y columns

# Keep only the Cols needed for SFRmaker and rename them
A2 = (
    A1[
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

# endregion

# region 4. Assign unique CODEs to any rows with missing CODE values
# %%
A2['CODE']
A2.loc[A2['CODE'].isnull(), 'CODE'] = [f'dummy_{i}' for i in range(1, A2['CODE'].isnull().sum() + 1)]
# endregion


# region 5. Split Polylines
# %%
def split_lines_by_neighbors(gdf: gpd.GeoDataFrame, tol: float = 0.000001) -> gpd.GeoDataFrame:
    """
    Split each (Multi)LineString where it meets OTHER features in gdf.
    Uses a spatial index so only intersecting neighbors are used as splitters.
    Returns one row per split segment, attributes copied from the original row.
    tol in CRS units (e.g. meters).
    """

    def flatten_geom(geom):
        """Recursively flatten GeometryCollections and Multi-geometries into simple geometries."""
        if hasattr(geom, 'geoms'):
            for g in geom.geoms:
                yield from flatten_geom(g)
        else:
            yield geom

    gdf = gdf.copy()
    geom_col = gdf.geometry.name
    other_cols = [c for c in gdf.columns if c != geom_col]
    gdf['_original_geom'] = gdf[geom_col].copy()  # Store original geometries

    # Snap Coordinates so near-coincident nodes match (for spatial index and splitting)
    if hasattr(shapely, 'set_precision'):
        gdf[geom_col] = shapely.set_precision(gdf[geom_col].array, tol)

    sindex = gdf.sindex

    new_rows = []

    for idx, row in gdf.iterrows():
        geom = row[geom_col]
        original_geom = row['_original_geom']
        attrs = row.to_dict()

        # candidate neighbors that intersect this geom's bbox
        cand_idx = list(sindex.query(geom, predicate='intersects'))
        cand_idx = [i for i in cand_idx if i != idx]

        if not cand_idx:
            # No neighbors - use original unsnapped geometry
            # Flatten in case original was MultiLineString
            for sub_seg in flatten_geom(original_geom):
                new_rows.append({**attrs, geom_col: sub_seg})
            continue

        neigh_geoms = gdf.loc[cand_idx, geom_col].values

        # build splitter from neighbors
        splitter = unary_union(neigh_geoms)

        # NEW: remove overlapping segments so split() only sees crossing/touching parts
        splitter = splitter.difference(geom)

        if splitter.is_empty or not geom.intersects(splitter):
            # Has neighbors but no actual split - use original unsnapped geometry
            # Flatten in case original was MultiLineString
            for sub_seg in flatten_geom(original_geom):
                new_rows.append({**attrs, geom_col: sub_seg})
            print(row['CODE'], 'no split')
            continue

        parts = split(geom, splitter)

        # Check if geometry was actually split (more than 1 part)
        if len(parts.geoms) == 1:
            # Not actually split - use original unsnapped geometry
            # Flatten in case original was MultiLineString
            for sub_seg in flatten_geom(original_geom):
                new_rows.append({**attrs, geom_col: sub_seg})
        else:
            # Geometry was actually split - use snapped version
            # Flatten any MultiLineStrings or GeometryCollections in the result
            for seg in parts.geoms:
                for sub_seg in flatten_geom(seg):
                    new_rows.append({**attrs, geom_col: sub_seg})

    # Return without _original_geom column
    return gpd.GeoDataFrame(new_rows, columns=other_cols + [geom_col], crs=gdf.crs)


A3 = split_lines_by_neighbors(A2, tol=0.01)
# endregion

# region 6. QA Split
# %% 6.1 Ensure coordinates are preserved for non-split features
A3_codes = A3['CODE'].value_counts()
non_split_codes = A3_codes[A3_codes == 1].index  # Codes that weren't split

A3_non_split = A3[A3['CODE'].isin(non_split_codes)].copy()
A2_matching = A2[A2['CODE'].isin(non_split_codes)].copy()

A2_matching = A2_matching.set_index('CODE')
A3_non_split = A3_non_split.set_index('CODE')

# Check coordinate differences
Coo_diff = 0
for code in non_split_codes:
    if code in A2_matching.index and code in A3_non_split.index:
        g3 = A2_matching.loc[code, 'geometry']
        g4 = A3_non_split.loc[code, 'geometry']

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
print(f'Total features in A2: {len(A2)}')
print(f'Total features in A3: {len(A3)}')
print(f'A3 features that were split: {len(A3) - len(non_split_codes)}')

# %% 6.2 Create Col to identify split segments
l_X_Y_Cols_ = [i + '_' for i in l_X_Y_Cols]
l_Cols1 = sorted(['CODE', 'length', *l_X_Y_Cols, *l_X_Y_Cols_, 'geometry'])
A3[l_X_Y_Cols_] = A3.geometry.apply(
    lambda g: pd.Series(
        g.geoms[0].coords[0] + g.geoms[-1].coords[-1]
        if g.geom_type.startswith('Multi')
        else g.coords[0] + g.coords[-1],
        index=l_X_Y_Cols_,
    )
)
A3['split'] = 0  # Set blank split column
A3_split = A3.loc[A3['CODE'].isin(A3['CODE'].value_counts()[A3['CODE'].value_counts() > 1].index)]
A3.loc[
    A3['CODE'].isin(A3['CODE'].value_counts()[A3['CODE'].value_counts() > 1].index),
    'split',
] = 1  # Set to 1 for split segments
A3_Coo_change = A3.loc[
    (A3['Xa'].round(3) != A3['Xa_'].round(3))  # Only contains rows where coordinates have changed
    | (A3['Ya'].round(3) != A3['Ya_'].round(3))
    | (A3['Xz'].round(3) != A3['Xz_'].round(3))
    | (A3['Yz'].round(3) != A3['Yz_'].round(3))
]
A3_split.shape, A3_Coo_change.shape, A3.shape

# %% 6.3 Ensure that no coordinates were changed for non-split features
shift = A3_Coo_change.loc[~A3_Coo_change['CODE'].isin(A3_split['CODE']), l_Cols1]
shift.shape

# %% 6.4 Identify rows in A3_split but not in A3_Coo_change
A3_split_no_Coo_change = A3_split.loc[~A3_split.index.isin(A3_Coo_change.index)].copy()
print(f'Rows in A3_split but not in A3_Coo_change: {len(A3_split_no_Coo_change)}')
print(f'\nUnique CODEs: {A3_split_no_Coo_change["CODE"].unique()}')
A3_split_no_Coo_change[l_Cols1]


# %% [markdown]
# A few entries seem to have had their coordinates changed for one of the split sub-segments but not for the other. In a lot of cases, there are multiple lines in a polyline with the exact same coordinates. SFRmaker should be able to handle this, so there will be no futher check at this point.
# %%
# endregion

# region 7. Recalc/assign some columns

# region 7.1 Length
# %% Plot a split segment to inspect length
A3.loc[A3['CODE'] == 'OVK03536']

# %% [markdown]
# Length needs to be recalculated after the split, based on the new geometries. As you can see above, all lengths for split segments are currently the same as the original segment length.

# %% Recalculte length based on the new geometries
A4 = A3.copy()
A4['length'] = A4.geometry.length

# %% Check recalculated lengths for a split segment
(
    A4.loc[A3['CODE'] == 'OVK03536', 'length'].sum().round(3),
    A3.loc[A3['CODE'] == 'OVK03536', 'length'].iloc[0].round(3),
)

# %% [markdown]
# The length sum isn't exactly identical, but I'll move forward for now. #666 can come back to this later.

# %%
# endregion

# region 7.2 Coordinates
# %%
Bool_Coo_changed = A4[l_X_Y_Cols].sum(axis=1) != A4[l_X_Y_Cols_].sum(axis=1)
A4.loc[Bool_Coo_changed, l_X_Y_Cols] = A4.loc[Bool_Coo_changed, l_X_Y_Cols_].values
(A4.loc[:, l_X_Y_Cols].values == A4.loc[:, l_X_Y_Cols_].values).all()

# %% [markdown]
# All values from the newly calculated coordinates have been assigned to the main GDF.

# %%
# %% Drop extra coordinate columns
A4.drop(columns=l_X_Y_Cols_, inplace=True)
# endregion

# region 7.3 Width
# %% Investigate
A4.describe(include='all')

# %% Set width to 1m where missing
# Width will be set to 1m where missing. This is very simplistic, but a good start. #666
A4.loc[A4['width'].isnull(), 'width'] = 1.0
# endregion

# region 7.4 Elv
# %% [markdown]
# Elevations will be assigned based on the thickest layer from 0.75-1.5m from the surface - standard drainage (ditch) depth in the Netherlands.

# %% Load TOP and BOT IDFs
l_TOP = [i for i in (M.Pa.WS / 'models/NBr/In/TOP').glob('*.idf')]
l_BOT = [i for i in (M.Pa.WS / 'models/NBr/In/BOT').glob('*.idf')]

TOP = imod.formats.idf.open(l_TOP, pattern='{name}_L{layer}_').interp(x=M.Xs, y=M.Ys, method='linear')
BOT = imod.formats.idf.open(l_BOT, pattern='{name}_L{layer}_').interp(x=M.Xs, y=M.Ys, method='linear')
Thk = TOP - BOT

# %% Isolate rows with missing Elv_UStr or Elv_DStr. Then print those where one is missing but the other is not.
A4_UStr_NA = A4.loc[A4['Elv_UStr'].isna()]
A4_DStr_NA = A4.loc[A4['Elv_DStr'].isna()]

print(A4.shape, A4_DStr_NA.shape, A4_UStr_NA.shape)
A4.loc[(A4['Elv_UStr'].isna() & A4['Elv_DStr'].notna()) | (A4['Elv_UStr'].notna() & A4['Elv_DStr'].isna())]

# %% [markdown]
# A4_DStr_NA & A4_UStr_NA have different number of entries. There are some rows where one of Elv_UStr or Elv_DStr are available, but the other one is missing. We can use those as an indicator of how well the filling worked.


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

    # Slice A for points where
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
mask_U = A4['Elv_UStr'].isna()
if mask_U.any():
    print(f'Filling {mask_U.sum()} NaNs in Elv_UStr')
    A4.loc[mask_U, 'Elv_UStr'] = rule_based_3ry_Elv(A4.loc[mask_U, 'Xa'], A4.loc[mask_U, 'Ya'], TOP, BOT)

# %% Apply to Elv_DStr
mask_D = A4['Elv_DStr'].isna()
if mask_D.any():
    print(f'Filling {mask_D.sum()} NaNs in Elv_DStr')
    A4.loc[mask_D, 'Elv_DStr'] = rule_based_3ry_Elv(A4.loc[mask_D, 'Xz'], A4.loc[mask_D, 'Yz'], TOP, BOT)

# %% Confirm no NaNs remain
A4.loc[A4['Elv_UStr'].isna() | A4['Elv_DStr'].isna()].ws.clip_Mdl_area(M=M)

# %% [markdown]
# Checking in QGIS confirms that the Elvs were assigned properly.

# endregion

# region 7.5. Calculate routing
# %% Create ID Col & Identify downstream
A4['ID'] = range(1, len(A4) + 1)
Coo_to_id = {(R.Xa, R.Ya): (R.CODE, R.ID) for R in A4.itertuples()}  # lookup d_ from coordinates to CODE
print(f'🟢 Lookup dictionary created with {bold}{len(Coo_to_id)}{style_reset} entries.')


# %% Define function to get DStr_code and DStr_ID based on end coordinates
def get_DStr(row):
    end_Coos = (row.Xz, row.Yz)
    result = Coo_to_id.get(end_Coos, (0, 0))
    return result


# %% Apply the function to create the 'DStr' column
A4[['DStr_code', 'DStr_ID']] = A4.apply(get_DStr, axis=1, result_type='expand')
print("🟢 'DStr' columns calculated.")
print(
    f'{round(A4["DStr_code"].value_counts().max() / A4.shape[0] * 100, 2)} % of DStrs are 0 (i.e. no start Coos match the end Coos of the current node).'
)

# %% Calculate UStr_IDs for each ID
upstream = A4.groupby('DStr_ID')['ID'].agg(list)
A4['UStr_IDs'] = A4['ID'].map(upstream).apply(lambda ids: ids if isinstance(ids, list) else [])
print("🟢 'UStr_IDs' column calculated.")
print(f'{round((A4["UStr_IDs"].str.len() > 0).sum() / A4.shape[0] * 100, 2)} % of segments have upstream IDs.')

# %% [markdown]
# We'll drop the segments that are smaller than 1m and don't have a DStr_ID. Those usually are protrusions, and they won't matter anyway as segments with no DStr ID will be connected to the closest end within 1m.

# %% Drop segments with length < 1m and no DStr_ID. Then plot segments with no connection.
A4 = A4.loc[~(A4['length'] < 1) | (A4['DStr_ID'] != 0)].copy()

print("🟢 'DStr' columns calculated.")
print(
    f'{round(A4["DStr_code"].value_counts().max() / A4.shape[0] * 100, 2)} % of DStrs are 0 (i.e. no start Coos match the end Coos of the current node).'
)

# %% [markdown]
# The percentage is still significant, even though the geometry has been precision-adjusted this time.
# Let's investigate

# %%
Pa_Out_A4 = M.Pa.In / 'SFR' / M.MdlN / f'WBD_detail_SW_NW_cleaned_A4_{M.MdlN}.gpkg'
Pa_Out_A4.parent.mkdir(parents=True, exist_ok=True)
A4.to_file(Pa_Out_A4, layer='WBD_detail_SW_NW_cleaned', driver='GPKG')
A4['DStr_ID'].value_counts()

# %% [markdown]
# No more than 5 segments are connected to the same DStr_ID - reasonable.

# %% Calculate min distance from start to any reach's end and investigate no matches.
A4['min_Dist'] = 0.0
A4.loc[A4['DStr_code'] == 0, 'min_Dist'] = A4.loc[A4['DStr_code'] == 0].apply(
    lambda row: c_Dist(row['Xz'], row['Yz'], A4['Xa'], A4['Ya']).min(), axis=1
)  # c_ min Dist for segments with no match

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
            DF.loc[(DF['min_Dist'] > Val_Prv) & (DF['min_Dist'] <= v), ['ID', 'min_Dist']]
            .sort_values(by='min_Dist')
            .apply(lambda R: f'{R["ID"].astype(int)}: {R["min_Dist"]:8.4f}', axis=1)
            .tolist()
        )

        print(
            f'-{Val_Prv:6} < min_Dist <= {v:5} |N: {N_below:6} (+ {(N_below - N_below_Prv):4}) ({round(P_below, 1):5} %) | Codes: {code_min_Dist[:5] + code_min_Dist[-5:]}\n'
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


d_min_Dist = print_segments_per_min_Dist(A4, l_Vals)

# %% [markdown]
# The total number of non matching segments has decreased a lot. They're still too many to manually connect. But if the splitting algortithm is improved, so it splits lines when they cross, it may drop even more.
# - The ones <1m can be attributed to closing errors, and we can connect them to the closest one via an algorithm.


# %% Review segments in model area
A4_w = A4.loc[A4.within(box(M.Xmin, M.Ymin, M.Xmax, M.Ymax))]

d_min_Dist_within = print_segments_per_min_Dist(A4_w, l_Vals)

# %% [markdown] Conclusion
# A QGIS review of segments follows:
# Segments that can get fixed by connecting them to the closest end are marked with "-|""
# CE: clipping error
#
# ---
# 0.001 - 0.1 m:
# - 7246 | 18780 | 41621 : 3 segments arriving at confluence with no outlet. -|
# - 34500: CE. -|
# - 8528:  CE. -|
# - 32400: CE. -|
#
# ---
# 0.1 - 1 m:
# - 18189: CE. -|
# - 22508: CE. -|
# - 21358: CE. -|
# - 42528: CE. -|
# - 18384: CE. -|
#
# ---
# 1 - 10 m:
# - 16581: reverse (length=min_Dist). -|
# - 19440: UStr of 19365. -|
# - 18685: Dstr segment didn't break properly. One segment's line's end is close to the middle of the other segment's line, not the end. But -|
# - 42780: -|
# - 18986: Didn't breat as end of segment meets lines of DStr segment in the middle and even then, there is a small gap. This is an upstreamost segment. I'd say it's not a big deal,but it covers a lot (~10 cells). Connecting it to the closest end might still work though. Let's see, then review later.
# - 21991: -|
# ---
# 10 - 100 m:
# - 5642: Didn't break properly. -|
# - 43413: Didn't break properly. -|
# - 58223: DStr segment has opposite direction and isn't connected to anything. -| (both it and the DStr segment)
# - 18023: This one is problematic, as connecting to the closest end might mess up the routing completely. Let's try connecting to the closest end, and review how bad it is.
# - 14492: Problematic... But it gave me the idea that they need to be connected to the segmnents where one of it's lines' (not the whole polyline) ends is closest. "-|" means that (line end instead of polline end)
# - 16874: Lake! -|
# - 19754: CE "-|"
# ---
# 100 - 1000 m:
# - 7314: CE "-|"
# - 43337: CE "-|"
# - 16875: Lake. Really unconnected -> Algorithm should be limited to some distance, e.g. a cell length.
# - 4601: CE "-|"
# - 12490: Lake. Confirsms limit idea.
# - 16571: Lake. " " "

# endregion

# region 7.6 Edit connections
# %%
A4_ = A4.copy()


# endregion

# region
# %%
A4_w = A4.loc[A4.within(box(M.Xmin, M.Ymin, M.Xmax, M.Ymax))]
A4_w.loc[A4_w['multiple_close'] != '', ['CODE', 'multiple_close']].sort_values(by='multiple_close', ascending=False)
A4.loc[
    (A4['multiple_close'] == '') & (A4['ID'] == A4['DStr_ID']),
    ['CODE', 'DStr_code', 'ID', 'DStr_ID', 'min_Dist', 'multiple_close'],
]
# endregion

# region
# %%
A4.columns
# endregion

# region
# %%
A4.loc[A4['CODE'].isin(A4_w['CODE']) & (A4['multiple_close'] != ''), 'multiple_close']
# endregion

# region
# %%
A4.loc[A4['CODE'].isin(A4_w['CODE']), ['DStr_code', 'DStr_ID']].shape
# endregion

# region
# %%
within_and_multiple = A4['CODE'].isin(A4_w['CODE']) & (A4['multiple_close'] != '')
# A4.loc[ within_and_multiple.index, 'DStr_code']= A4.loc[ within_and_multiple , 'multiple_close'].str.split(', ').str[0]
# endregion

# region  Check for any missing DStr_ID values after assignment
# %%
missing_dstr_id = A4.loc[within_and_multiple, 'DStr_ID'].isna().sum()
print(f'Number of segments with missing DStr_ID after assignment: {missing_dstr_id}')

if missing_dstr_id > 0:
    print('Segments with missing DStr_ID:')
    print(A4.loc[within_and_multiple & A4['DStr_ID'].isna(), ['CODE', 'DStr_code', 'DStr_ID']])
# endregion

# region 2.6 Correct Elv_UStr and Elv_DStr
# %%
#### Prep
l_A4_Cols = [
    'ID',
    'CODE',
    'width',
    'length',
    'Elv_UStr',
    'Elv_DStr',
    'DStr_code',
    'DStr_ID',
    'Xa',
    'Ya',
    'Xz',
    'Yz',
    'geometry',
    'split',
]
l_success, l_infinite_loop, l_closing_error = [], [], []
A5 = A4.copy()[
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
]
A5.describe()
A2.describe()
A2.shape, A4.shape, A5.shape
A5.sort_values('CODE')
A5.loc[
    A3['Elv_UStr'].isna() | A3['Elv_DStr'].isna(),
    ['CODE', 'ID', 'DStr_ID', 'Elv_UStr', 'Elv_DStr'],
].sort_values('CODE')
# endregion

# region [markdown]
# %%
# Elvs were inherited from the original segments when they were split. This is incorrect, except for the UStr_Elv of the UStr-most segment and the DStr_Elv of the DStr-most segment.
# But, in A4 (therefore A5 as well), NA Elvs were filled for split sements. So those don't need to be corrected.
# endregion

# region Fill A3_ Elv NaNs from A5_ values
# %%
# We filled all NA values of A4
# A2_ Elv columns need to be filled
# endregion

# region NaN Elvs need to be filled in A3 too, if it is to be used for Elv corrections
# %%
A2_ = A2.copy()
A2_Elv_NaN_codes = A2_.loc[
    A2_['Elv_UStr'].isnull() | A2_['Elv_DStr'].isnull(), 'CODE'
].unique()  # list of codes where Elv is NaN
A4_Elv_NaN_in_A2 = A4.loc[A4['CODE'].isin(A2_Elv_NaN_codes), ['CODE', 'Elv_UStr', 'Elv_DStr']]  # DF for those codes
A4_Elv_UStr_max = A4_Elv_NaN_in_A2.groupby(['CODE'])['Elv_UStr'].max()
A4_Elv_DStr_min = A4_Elv_NaN_in_A2.groupby(['CODE'])['Elv_DStr'].min()
A2_.loc[A2_['CODE'].isin(A2_Elv_NaN_codes), 'Elv_UStr'] = A2_.loc[A2_['CODE'].isin(A2_Elv_NaN_codes), 'CODE'].map(
    A4_Elv_UStr_max
)
A2_.loc[A2_['CODE'].isin(A2_Elv_NaN_codes), 'Elv_DStr'] = A2_.loc[A2_['CODE'].isin(A2_Elv_NaN_codes), 'CODE'].map(
    A4_Elv_DStr_min
)
# endregion

# region Fill A3 Elv - Alt method
# %%
A2__ = A2.copy()
A2___UStr_NA = A2__.loc[A2__['Elv_UStr'].isna()]  # Isolate upstream missing elevations
A2___DStr_NA = A2__.loc[A2__['Elv_DStr'].isna()]  # Isolate downstream missing elevations
A_Xas = xr.DataArray(A2___UStr_NA.Xa, dims='points')
A_Yas = xr.DataArray(A2___UStr_NA.Ya, dims='points')
A2___UStr_NA.shape
# endregion

# region Apply to Elv_UStr
# %%
mask_U = A2__['Elv_UStr'].isna()
if mask_U.any():
    print(f'Filling {mask_U.sum()} NaNs in Elv_UStr')
    A2__.loc[mask_U, 'Elv_UStr'] = rule_based_3ry_Elv(A2__.loc[mask_U, 'Xa'], A2__.loc[mask_U, 'Ya'], TOP, BOT)
# endregion

# region Apply to Elv_DStr
# %%
mask_D = A2__['Elv_DStr'].isna()
if mask_D.any():
    print(f'Filling {mask_D.sum()} NaNs in Elv_DStr')
    A2__.loc[mask_D, 'Elv_DStr'] = rule_based_3ry_Elv(A2__.loc[mask_D, 'Xz'], A2__.loc[mask_D, 'Yz'], TOP, BOT)
A2__[['CODE', 'Elv_UStr', 'Elv_DStr']].describe()
A2_[['CODE', 'Elv_UStr', 'Elv_DStr']].describe()
# endregion

# region [markdown]
# %%
# We'll procede with A2__. It should be more accurate, since it's calculated from TOP and BOT directly.<br>
# A2_ might miscalculate when a segment was split into multiple and an intermediate point has highe/r/lower elevation than the start/end points.
# endregion

# region Correct Elvs
# %%
n = 1
for i in A4.loc[
    (~A4['CODE'].isin(A3.loc[A3['Elv_UStr'].isna() & A3['Elv_DStr'].isna(), 'CODE'])) & (A4['split'] == 1),
    'CODE',
].unique():  # [::-1]: # [26:27]: #[n:n+1]:
    print(
        f'{i} - {n:4}/{
            len(
                A4.loc[
                    (~A4["CODE"].isin(A3.loc[A3["Elv_UStr"].isna() & A3["Elv_DStr"].isna(), "CODE"]))
                    & (A4["split"] == 1),
                    "CODE",
                ].unique()
            ):4}',
        end=' - ',
    )
    n += 1

    # Prep GDF and Se for CODE
    GDF = A4.loc[A4['CODE'] == i, l_A4_Cols]  # Create GDF for the current CODE
    Se = A2_.loc[A2_['CODE'] == i]
    Coo_a = (Se['Xa'].values[0], Se['Ya'].values[0])
    Coo_z = (Se['Xz'].values[0], Se['Yz'].values[0])

    # Prepare for Elv_DStr adjustment
    drop = Se['Elv_UStr'].values[0] - Se['Elv_DStr'].values[0]
    length = GDF['length'].sum()
    GDF['C'] = GDF['length'] / length

    # Iterate UStr to DStr and edit Elvs
    dx = GDF['Xa'] - Coo_a[0]
    dy = GDF['Ya'] - Coo_a[1]
    idx = np.hypot(dx, dy).idxmin()
    ID_UStr = GDF.loc[[idx], 'ID'].values[0]  # or GDF.loc[idx] for a Series

    # Initialize Elv_UStr for the first segment
    GDF.loc[GDF['ID'] == ID_UStr, 'Elv_UStr'] = Se['Elv_UStr'].values[0]

    ID = ID_UStr
    while True:
        GDF.loc[GDF['ID'] == ID, 'Elv_DStr'] = (
            GDF.loc[GDF['ID'] == ID, 'Elv_UStr'].values[0] - drop * GDF.loc[GDF['ID'] == ID, 'C'].values[0]
        )

        # Replace the problematic section with:
        distances_to_end = GDF.apply(lambda R: np.hypot(R.Xz - Coo_z[0], R.Yz - Coo_z[1]), axis=1)
        closest_to_end_idx = distances_to_end.idxmin()
        end_ID = GDF.loc[closest_to_end_idx, 'ID']

        if ID == end_ID:
            break

        Coo_prev = (
            GDF.loc[GDF['ID'] == ID, 'Xz'].values[0],
            GDF.loc[GDF['ID'] == ID, 'Yz'].values[0],
        )
        ID_prev = ID
        try:
            ID = GDF.loc[
                GDF.apply(
                    lambda R: np.isclose((R.Xa, R.Ya), Coo_prev, atol=0.0001).all(),
                    axis=1,
                ),
                'ID',
            ].values[0]
            if ID == ID_prev:
                print(f'Stuck at ID {ID}. Breaking loop to avoid infinite iteration.')
                l_infinite_loop.append(i)
                break
        except IndexError:
            print(f'No matching ID found for coordinates {Coo_prev} in CODE {i}. Breaking loop.')
            l_closing_error.append(i)
            break

        # print(ID_prev, ID)
        GDF.loc[GDF['ID'] == ID, 'Elv_UStr'] = GDF.loc[GDF['ID'] == ID_prev, 'Elv_DStr'].values[0]

    # Check
    check = np.isclose(
        A2_.loc[A2_['CODE'] == i, 'Elv_DStr'],
        GDF.loc[GDF['ID'] == ID, 'Elv_UStr'].values[0] - drop * GDF.loc[GDF['ID'] == ID, 'C'].values[0],
        atol=0.1,
    ).any()
    if not check:
        print(
            f'Discrepancy in Elv_DStr for CODE {i}: calculated {GDF.loc[GDF["ID"] == ID, "Elv_DStr"].values[0]}, expected {A2_.loc[A2_["CODE"] == i, "Elv_DStr"].values[0]}'
        )
        l_closing_error.append(i)
    else:
        print('Elvs calculated correctly.')
        l_success.append(i)
        A5.loc[A5['CODE'] == i, 'Elv_UStr'] = GDF['Elv_UStr'].values
        A5.loc[A5['CODE'] == i, 'Elv_DStr'] = GDF['Elv_DStr'].values

Tot = len(l_success) + len(l_infinite_loop) + len(l_closing_error)
print(f'Successful: {len(l_success)} ({round(len(l_success) * 100 / Tot, 2)} %)')
print(f'Infinite loop: {len(l_infinite_loop)} ({round(len(l_infinite_loop) * 100 / Tot, 2)} %)')
print(f'Closing error: {len(l_closing_error)} ({round(len(l_closing_error) * 100 / Tot, 2)} %)')
# endregion

# region [markdown]
# %%
# Most are successful. We'll proceed for now, and hope SFRmaker will handle the rest. Otherwise we can come back.
# endregion

# region Inspect Out path
# %%
Pa_Out
# endregion

# region Save
# %%
Pa_Out.parent.mkdir(parents=True, exist_ok=True)
A5_out = A5.copy()
A5_out['DStr_code'] = A5_out['DStr_code'].astype('string')
A5_out['DStr_ID'] = A5_out['DStr_ID'].astype('Int64')
A5_out['split'] = A5_out['split'].astype('Int64')
A5_out = A5_out[
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
A5_out.to_file(Pa_Out, driver='GPKG', engine='fiona', index=False)
# endregion

# region
# %%
# endregion
