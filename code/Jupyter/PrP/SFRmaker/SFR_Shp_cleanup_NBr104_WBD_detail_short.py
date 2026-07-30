# %% Imports

# Import sfrmaker and other necessary packages for SFR network creation
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

# %% 1. Options
MdlN = 'NBr104'
M = Mdl_N(MdlN)

detailed = 'hydroobject'
primary = 'LEGGER_VASTGESTELD_WATERLOOP_CATEGORIE_A'
Pa_Gpkg_In = M.Pa.WS / r'models\NBr\other\BrabantseDelta\acceptatiedatabase.gdb'
Pa_Out = M.Pa.WS / rf'models\NBr\In\SFR\{MdlN}\WBD_detail_SW_NW_cleaned_{MdlN}.gpkg'
l_X_Y_Cols = ['Xa', 'Ya', 'Xz', 'Yz']

# %% 2.1.Load GPkg
A0 = gpd.read_file(Pa_Gpkg_In, layer=detailed)  # A is the primary Gpkg layer #111
A0_ = A0.copy()

# %% 2.2. Review GPkg and clean up columns
# 2.2.0 Create X & Y columns and inspect
A1 = A0.ws.Calc_XY_start_end_from_Geom()
A1.describe()

# %% [markdown]
# It's clear that a lot of the columns have very little data. So I will remove those columns as there isn't much we can do with them, but it would take time to analyze them.

# %% 2.2.2 Check number of values and remove columns with little data.
A1.ws.Col_value_counts_grouped()

# %% [markdown]
# This confirms that most columns have very few values. We'll drop all columns that have fewer valid values than 10% of the length of the DataFrame.

# %% Keep columns with >10% valid values
l_GDF_Cols_to_keep = [col for col in A1.columns if A1[col].notnull().sum() >= (0.1 * len(A1))]
print(f'{len(l_GDF_Cols_to_keep)}/{A1.shape[1]} columns kept in A1.')
A2 = A1[l_GDF_Cols_to_keep].copy()

# %% [markdown]
# A lot of columns remain, but their number has reduced a lot. Let's check them, and decide which to keep. We'll proceed with a joint GDF called GDF.
A2.shape

# %% 2.2.3 Narrow down to SFRmaker columns.
# Truth is we only need a few columns for SFRmaker to work. We'll proceed with just those columns, but the rest could potentially be reviewed and saved as a .shp/.gpkg file later on.
A2.describe(include='all')

# %% [markdown]
# 1. **CODE**:                        OK           -          All unique, as we want them to be.
# 2. **WS_STATUS_L**:                 OK           -          All filled. Seems ok.
# 3. **WS_LEGGERCATEGORIE_L**:        OK           -          "   "   "
# 4. **WS_LEGGERBRON_L**:             OK           -
# 5. **DATUM_VASTGESTELD**:           OK           -
# 6. **LEGGER_KENMERK**:              OK           -
# 7. **WS_BODEMBREEDTE_L**:           OK           -
# 8. **WS_BH_BOVENSTROOMS_L**:        OK           -
# 9. **WS_BH_BENEDENSTROOMS_L**:      OK           -
# 10. **WS_TALUD_LINKS_L**:           OK           -
# 11. **WS_TALUD_RECHTS_L**:          OK           -
# 12. **WS_BIJZ_FUNCTIE_L**:          OK           -
# 13. **WIJZIGING**:                  OK           -
# 14. **CREATED_USER**:               OK           -
# 15. **CREATED_DATE**:               OK           -
# 16. **LAST_EDITED_USER**:           OK           -
# 17. **LAST_EDITED_DATE**:           OK           -
# 18. **WS_LEGGERVERWIJZING_L**:      OK           -
# 19. **SHAPE_Length**:               OK           -
# 20. **geometry**:                   OK           -
# 21. **Xstart**:                     OK           -
# 22. **Ystart**:                     OK           -
# 23. **Xend**:                       OK           -
# 24. **Yend**:                       OK           -

# %% The only columns we need
A3 = A2[
    [
        'CODE',
        'WS_BODEMBREEDTE_L',
        'WS_BH_BOVENSTROOMS_L',
        'WS_BH_BENEDENSTROOMS_L',
        'SHAPE_Length',
        *l_X_Y_Cols,
        'geometry',
    ]
].copy()
A3.rename(
    columns={
        'WS_BODEMBREEDTE_L': 'width',
        'WS_BH_BOVENSTROOMS_L': 'Elv_UStr',
        'WS_BH_BENEDENSTROOMS_L': 'Elv_DStr',
        'SHAPE_Length': 'length',
    },
    inplace=True,
)

# %% [markdown]
# The CODE column is not full. We'll add dummy codes where missing. (although they should have been there in the first place)

# %% 2.2.4 Fill CODE
A3['CODE']
A3.loc[A3['CODE'].isnull(), 'CODE'] = [f'dummy_{i}' for i in range(1, A3['CODE'].isnull().sum() + 1)]


# %% 2.3. Split
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


# %% 2.3.1 Split and QA
A4 = split_lines_by_neighbors(A3, tol=0.001)
# Diagnostic: Check coordinates preservation
# Compare A3 (before split) with non-split rows in A4
A4_codes = A4['CODE'].value_counts()
non_split_codes = A4_codes[A4_codes == 1].index  # Codes that weren't split

# Get non-split rows from A4
A4_non_split = A4[A4['CODE'].isin(non_split_codes)].copy()
A3_matching = A3[A3['CODE'].isin(non_split_codes)].copy()

# Extract coordinates
A3_matching = A3_matching.set_index('CODE')
A4_non_split = A4_non_split.set_index('CODE')

# Check coordinate differences
Coo_diff = 0
for code in non_split_codes:
    if code in A3_matching.index and code in A4_non_split.index:
        g3 = A3_matching.loc[code, 'geometry']
        g4 = A4_non_split.loc[code, 'geometry']

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
print(f'Total features in A3: {len(A3)}')
print(f'Total features in A4: {len(A4)}')
print(f'A4 features that were split: {len(A4) - len(non_split_codes)}')

# %%
l_X_Y_Cols_ = [i + '_' for i in l_X_Y_Cols]
l_Cols1 = sorted(['CODE', 'length', *l_X_Y_Cols, *l_X_Y_Cols_, 'geometry'])
A4[l_X_Y_Cols_] = A4.geometry.apply(
    lambda g: pd.Series(
        g.geoms[0].coords[0] + g.geoms[-1].coords[-1]
        if g.geom_type.startswith('Multi')
        else g.coords[0] + g.coords[-1],
        index=l_X_Y_Cols_,
    )
)
A4['split'] = 0  # Set blank split column
A4_split = A4.loc[A4['CODE'].isin(A4['CODE'].value_counts()[A4['CODE'].value_counts() > 1].index)]
A4.loc[A4['CODE'].isin(A4['CODE'].value_counts()[A4['CODE'].value_counts() > 1].index), 'split'] = (
    1  # Set to 1 for split segments
)
A4_Coo_change = A4.loc[
    (A4['Xa'].round(3) != A4['Xa_'].round(3))  # Only contains rows where coordinates have changed
    | (A4['Ya'].round(3) != A4['Ya_'].round(3))
    | (A4['Xz'].round(3) != A4['Xz_'].round(3))
    | (A4['Yz'].round(3) != A4['Yz_'].round(3))
]
A4_split.shape, A4_Coo_change.shape, A4.shape

# %%
# pd.DataFrame( {'CODE': A4_split['CODE'].value_counts().index, 'counts': A4_split['CODE'].value_counts().values} ).groupby('counts').size()
shift = A4_Coo_change.loc[~A4_Coo_change['CODE'].isin(A4_split['CODE']), l_Cols1]
shift.shape

# %%
# - Shift being empty shows that no coordinates were changed, except for the CODEs that were split! As was desired.
# Identify rows in A4_split but not in A4_Coo_change
A4_split_no_Coo_change = A4_split.loc[~A4_split.index.isin(A4_Coo_change.index)].copy()
print(f'Rows in A4_split but not in A4_Coo_change: {len(A4_split_no_Coo_change)}')
print(f'\nUnique CODEs: {A4_split_no_Coo_change["CODE"].unique()}')
A4_split_no_Coo_change[l_Cols1]

# %% [markdown]
# A few entries seem to have had their coordinates changed for one of the split sub-segments but not for the other. This shows some sort of false start and going back and forth. Let's check those that are in the Mdl_Aa.

# %%
A4_split_no_Coo_change.ws.clip_Mdl_area(M=M)

# %% [markdown]
# Fortunately, there are only 3 CODEs.

# %%
l_Cols1_ = l_Cols1.copy()
l_Cols1_.remove('geometry')
l_Cols1_.insert(2, 'geometry')
for i, code in A4_split_no_Coo_change.ws.clip_Mdl_area(M=M)['CODE'].items():
    GDF_section = A4.loc[A4['CODE'] == code, l_Cols1_]  # .loc['CODE'] # .iloc[i-1]
    print(f'Coordinates have changed for {code}')
    print('-' * 50)

# %% [markdown]
# After checking those CODEs in QGIS, I understand what happened:
# - When a MULTILINESTRING is split, sometimes the LINESTRINGS composing it are not in the correct order. So it is possible for Xa=Xa_ etc. for one of the sub-segments. This is not a problem in those cases, although it's also clear that there are redundant elements in the shapefile (which is understandable, as there are just too many to check manually).
# - Below are some comments I made before I came to the conclusion above. There is no reason to make any changes to those segments for now. SFRmaker will handle them fine, and they don't make much of a difference in NBr40 anyway.
#     - OWL41232 is redundant, we can get rid of the long segment, as it may cause problems later.
#     - OWL17672 is strange. I'll remove it altogether. It's close to the edge of the Mdl_Aa and outside the Mdl anyway.
#     - OVK02257 is supposed to be 1 long segment. The split was redundant. The small resulting segment can be removed, and the other 2 can be merged.

# %%
A4_split_no_Coo_change.ws.clip_Mdl_area(M=M)

# %% 2.3.3 Recalc/assign some columns
A4_split['CODE'].value_counts()

# %% Length
A4.loc[A4['CODE'] == 'OVK03536']

# %% [markdown]
# Length needs to be recalculated after the split, based on the new geometries. As you can see above, all lengths for split segments are currently the same as the original segment length.

# %%
A5 = A4.copy()
A5['length'] = A5.geometry.length
A5.loc[A4['CODE'] == 'OVK03536', 'length'].sum().round(3), A4.loc[A4['CODE'] == 'OVK03536', 'length'].iloc[0].round(3)
# The length sum isn't exactly identical, but I'll move forward for now. #666 can come back to this later

# %% Coordinates
Bool_Coo_changed = A5[l_X_Y_Cols].sum(axis=1) != A5[l_X_Y_Cols_].sum(axis=1)
A5.loc[Bool_Coo_changed, l_X_Y_Cols] = A5.loc[Bool_Coo_changed, l_X_Y_Cols_].values
(A5.loc[:, l_X_Y_Cols].values == A5.loc[:, l_X_Y_Cols_].values).all()

# %% [markdown]
# All values from the newly calculated coordinates have been assigned to the main GDF.

# %%
A5.drop(columns=l_X_Y_Cols_, inplace=True)

# %% 2.4 Fill NaN values for width, Elv_UStr, Elv_DStr
### 2.4.0 Investigate
A5.describe(include='all')

# %% 2.4.1 Width
# Width will be set to 1m where missing. This is very simplistic, but a good start. #666
A5.loc[A5['width'].isnull(), 'width'] = 1.0

# %% 2.4.2 Elv
# %% [markdown]
# ~~We'll assign Elv_UStr and Elv_DStr based on layer 1 top and bottom elevations.~~<br>
# This didn't workout well because L1 is not very thick in many places, leading to unrealistic/low GW discharge to the streams.<br>
# For this reason, we'll assign the the elevations based on the thickest layer from 0.75-1.5m from the surface.

# %% Prep
l_T = [i for i in (M.Pa.WS / 'models/NBr/In/TOP').glob('*.idf')]
l_BOT = [i for i in (M.Pa.WS / 'models/NBr/In/BOT').glob('*.idf')]

TOP_init = imod.formats.idf.open(l_T, pattern='{name}_L{layer}_')
BOT_init = imod.formats.idf.open(l_BOT, pattern='{name}_L{layer}_')

# %%
# Regrid to dx, dy using xarray's interp, preserving extent
# Determine target x and y coordinates based on TOP
# Use .item() to ensure we get scalar floats, not 0-d DataArrays
xmin, xmax = TOP_init.x.min().item(), TOP_init.x.max().item()
ymin, ymax = TOP_init.y.min().item(), TOP_init.y.max().item()

new_x = np.arange(xmin, xmax + M.cellsize, M.cellsize)

# %% Check if y is descending (standard for IDFs)
if TOP_init.y.values[1] < TOP_init.y.values[0]:
    new_y = np.arange(ymax, ymin - M.cellsize, -M.cellsize)
else:
    new_y = np.arange(ymin, ymax + M.cellsize, M.cellsize)

TOP = TOP_init.interp(x=new_x, y=new_y, method='linear')
BOT = BOT_init.interp(x=new_x, y=new_y, method='linear')
Thk = TOP - BOT

# %%
A5_UStr_NA = A5.loc[A5['Elv_UStr'].isna()]  # Isolate upstream missing elevations
A5_DStr_NA = A5.loc[A5['Elv_DStr'].isna()]  # Isolate downstream missing elevations
A5.shape, A5_DStr_NA.shape, A5_UStr_NA.shape

A5.loc[(A5['Elv_UStr'].isna() & A5['Elv_DStr'].notna()) | (A5['Elv_UStr'].notna() & A5['Elv_DStr'].isna())]

# %% [markdown]
# A5_DStr_NA & A5_UStr_NA have different number of entries. There are some rows where one of Elv_UStr or Elv_DStr are available, but the other one is missing. We can use those as an indicator of how well the filling worked.


# %% Fill
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


# %%
# ## For extra plotting in function above.
#     # for i in range(len((valid_Ls.sum(axis=0)/len(A5_UStr_NA)*100).round(2).data)):
#     #     print(f"L {i+1:2}:{valid_Ls[:,i].sum().item():6} valid points out of {len(A5_UStr_NA)} ({round(valid_Ls[:,i].sum().item()/len(A5_UStr_NA)*100, 2):5}%)")

#     N = 10000
#     np.set_printoptions(suppress=True)
#     display(Drn_Elv_max.isel(points=N).values, Drn_Elv_min.isel(points=N).values)
#     display(TOP_Pts.isel(points=N).data)
#     display(BOT_Pts.isel(points=N).data)
#     display(valid_Ls.isel(points=N).data)
#     display(valid_Ls.layer[valid_Ls.isel(points=N)].data)

# # plot some valid thicknesses to see check the function has worked well
# for i in range(0, 10000, 1000):
#     display(valid_Ls.layer[valid_Ls.isel(points=i)].data)
#     display(valid_Thk.where(valid_Thk.isel(points=i)>=0).isel(points=i).data)
#     print(best_layer_idx.isel(points=i).data)

# %%
A_Xas = xr.DataArray(A5_UStr_NA.Xa, dims='points')
A_Yas = xr.DataArray(A5_UStr_NA.Ya, dims='points')

# %% Apply to Elv_UStr
mask_U = A5['Elv_UStr'].isna()
if mask_U.any():
    print(f'Filling {mask_U.sum()} NaNs in Elv_UStr')
    A5.loc[mask_U, 'Elv_UStr'] = rule_based_3ry_Elv(A5.loc[mask_U, 'Xa'], A5.loc[mask_U, 'Ya'], TOP, BOT)

# %% Apply to Elv_DStr
mask_D = A5['Elv_DStr'].isna()
if mask_D.any():
    print(f'Filling {mask_D.sum()} NaNs in Elv_DStr')
    A5.loc[mask_D, 'Elv_DStr'] = rule_based_3ry_Elv(A5.loc[mask_D, 'Xz'], A5.loc[mask_D, 'Yz'], TOP, BOT)

# %% Check
A5.loc[A4['Elv_UStr'].isna() | A4['Elv_DStr'].isna()].ws.clip_Mdl_area(M=M)

# %% [markdown] Checking in QGIS confirms that the Elvs were assigned properly.

# %% 2.5. Calculate routing
### 2.5.0 Create ID Col & Identify downstream
A5.columns
A5['ID'] = range(1, len(A5) + 1)
# Create a lookup dictionary from start coordinates to CODE
Coo_to_id = {(R.Xa, R.Ya): (R.CODE, R.ID) for R in A5.itertuples()}

print(f'✓ Lookup dictionary created with {bold}{len(Coo_to_id)}{style_reset} entries.')


# %% Function to find the downstream ID
def get_DStr(row):
    end_Coos = (row.Xz, row.Yz)
    result = Coo_to_id.get(end_Coos, (0, 0))
    return result


# %% Apply the function to create the 'DStr' column
A5[['DStr_code', 'DStr_ID']] = A5.apply(get_DStr, axis=1, result_type='expand')

print("✓ 'DStr' columns calculated.")
print(
    f'{round(A5["DStr_code"].value_counts().max() / A5.shape[0] * 100, 2)} % of DStrs are 0 (i.e. no start Coos match the end Coos of the current node).'
)

# %% [markdown]
# The percentage is much bigger than expected. Let's investigate.

# %% 2.5.1 Investigate segments that failed to connect
#### Check out number of matches/no matches
A5['DStr_match'] = A5['DStr_code'].isin(A5['CODE'])
A5['DStr_code'].value_counts()

# %% [markdown]
# 24 features being UStr of the same feature is a lot, but those were split. If we count for each id, the number will be much smaller.

# %%
A5['DStr_ID'].value_counts()

# %%
A5['DStr_match'].value_counts()

# %% Calculate min distance from start to any reach's end and investigate no matches.
A5['min_Dist'] = 0.0
A5.loc[A5['DStr_code'] == 0, 'min_Dist'] = A5.loc[A5['DStr_code'] == 0].apply(
    lambda row: c_Dist(row['Xz'], row['Yz'], A5['Xa'], A5['Ya']).min(), axis=1
)
N_total_no_match = (A5['DStr_code'] == 0).sum()
A5.loc[A5['DStr_match'] == False, 'min_Dist'].describe()
l_Vals = [0.001, 0.1, 1, 10, 100, 1000, 10000]

print(f'Out of the {N_total_no_match} segments that do not match:')

N_below_Prv, Val_Prv = 0, 0
for v in l_Vals:
    N_below = (A5.loc[A5['DStr_match'] == False, 'min_Dist'] <= v).sum()
    P_below = round(N_below / N_total_no_match * 100, 2)

    sample_A5 = A5.loc[(A5['min_Dist'] > Val_Prv) & (A5['min_Dist'] <= v), ['ID', 'min_Dist']].sort_values(
        by='min_Dist'
    )
    sample_A5['Code:min_Dist'] = sample_A5.apply(lambda row: f'{row["ID"].astype(int)}: {row["min_Dist"]:8.4f}', axis=1)
    sample_A5 = sample_A5['Code:min_Dist']
    example_nodes = sample_A5.iloc[:].tolist()

    print(
        f'-{Val_Prv:6} < min_Dist <= {v:5} |N: {N_below:6} (+ {(N_below - N_below_Prv):4}) ({round(P_below, 1):5} %) | Codes: {example_nodes}\n'
    )

    N_below_Prv, Val_Prv = N_below, v

# %% [markdown]
# The total number of segments that do not match is too high to check them all, but we'll check some cases from each group.
# - The following information hasn't been checked. Proceeding for now #666.
# - The ones <1m can be attributed to closing errors, and we can connect them to the closest one via an algorithm.
# - They take up 82.9% of the unmatched, which means there aren't that many remaining.

# %%
A5.loc[A5['min_Dist'].between(1, 10, inclusive='right'), ['CODE', 'ID', 'min_Dist']].sort_values(
    by='min_Dist'
).reset_index(drop=True)

# %% [markdown]
# - Some of the smaller min_Dist values, e.g. OWL38849, OWL20317, are clearly closing errors.
# - Some of the bigger min_Dist values, e.g. OWL16354, OWL36507, OWL31824, OWL00338, are missing connections, but it's still reasonable to connect them to the closest segment.
# - The shapefile is flawed but we don't have access to the process used to make it, so it's hard to automatically fix those errors. We'll proceed with connecting them to the closest segment for now.

# %%
A5_within = A5.loc[A5.within(box(M.Xmin, M.Ymin, M.Xmax, M.Ymax))]

# %%
A5_within.loc[A5_within['min_Dist'].between(0, 1, inclusive='right'), [i for i in A5_within if i != 'geometry']]

# %%
A5_within.loc[A5_within['min_Dist'].between(1, 10, inclusive='right'), [i for i in A5_within if i != 'geometry']]

# %%
A5_within.loc[A5_within['min_Dist'].between(10, 10000, inclusive='right'), [i for i in A5_within if i != 'geometry']]

# %% 2.5.2 Edit connections
A5_ = A5.copy()

# %% Initialize the 'multiple_close' column with empty strings
A5['multiple_close'] = ''

# %% Select rows to correct
A5_correct_DStr = A5[(A5['DStr_code'] == 0) & (A5['min_Dist'] < 10)].copy()
print(f'Found {len(A5_correct_DStr)} segments with no downstream connection and a potential connection within 10m.')

N_single_match, N_multiple_match = 0, 0

# %% Loop through the rows that need correction
for i, R in A5_correct_DStr.iterrows():
    # Calculate distances from the current row's end point to all other rows' start points
    distances = c_Dist(R['Xz'], R['Yz'], A5['Xa'], A5['Ya'])

    # Find segments where the distance is less than 10m
    close_mask = (distances < 10) & (A5.index != i)  # Exclude self

    if close_mask.any():
        # Get the subset of distances and indices
        d_close = distances[close_mask]
        idx_close = A5.index[close_mask]

        # Find position of minimum distance
        if hasattr(d_close, 'idxmin'):
            best_idx = d_close.idxmin()
        else:
            min_pos = d_close.argmin()
            best_idx = idx_close[min_pos]

        # Update 'DStr_code' and 'DStr_ID' with the closest one
        A5.loc[i, 'DStr_code'] = A5.loc[best_idx, 'CODE']
        A5.loc[i, 'DStr_ID'] = A5.loc[best_idx, 'ID']

        if len(idx_close) == 1:
            N_single_match += 1
        else:
            N_multiple_match += 1
            # Store all IDs
            ids = A5.loc[idx_close, 'ID'].tolist()
            A5.loc[i, 'multiple_close'] = ', '.join(map(str, ids))

print(f'✓ Corrected {N_single_match} segments with a single match.')
print(f"✓ Corrected {N_multiple_match} segments with multiple matches (closest selected, others in 'multiple_close').")

# %%
A5_within = A5.loc[A5.within(box(M.Xmin, M.Ymin, M.Xmax, M.Ymax))]
A5_within.loc[A5_within['multiple_close'] != '', ['CODE', 'multiple_close']].sort_values(
    by='multiple_close', ascending=False
)
A5.loc[
    (A5['multiple_close'] == '') & (A5['ID'] == A5['DStr_ID']),
    ['CODE', 'DStr_code', 'ID', 'DStr_ID', 'min_Dist', 'multiple_close'],
]

# %%
A5.columns

# %%
A5.loc[A5['CODE'].isin(A5_within['CODE']) & (A5['multiple_close'] != ''), 'multiple_close']

# %%
A5.loc[A5['CODE'].isin(A5_within['CODE']), ['DStr_code', 'DStr_ID']].shape

# %%
within_and_multiple = A5['CODE'].isin(A5_within['CODE']) & (A5['multiple_close'] != '')
# A5.loc[ within_and_multiple.index, 'DStr_code']= A5.loc[ within_and_multiple , 'multiple_close'].str.split(', ').str[0]

# %%  Check for any missing DStr_ID values after assignment
missing_dstr_id = A5.loc[within_and_multiple, 'DStr_ID'].isna().sum()
print(f'Number of segments with missing DStr_ID after assignment: {missing_dstr_id}')

if missing_dstr_id > 0:
    print('Segments with missing DStr_ID:')
    print(A5.loc[within_and_multiple & A5['DStr_ID'].isna(), ['CODE', 'DStr_code', 'DStr_ID']])

# %% 2.6 Correct Elv_UStr and Elv_DStr
#### Prep
l_A5_Cols = [
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
A6 = A5.copy()[
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
A6.describe()
A3.describe()
A3.shape, A5.shape, A6.shape
A6.sort_values('CODE')
A6.loc[A4['Elv_UStr'].isna() | A4['Elv_DStr'].isna(), ['CODE', 'ID', 'DStr_ID', 'Elv_UStr', 'Elv_DStr']].sort_values(
    'CODE'
)

# %% [markdown]
# Elvs were inherited from the original segments when they were split. This is incorrect, except for the UStr_Elv of the UStr-most segment and the DStr_Elv of the DStr-most segment.
# But, in A5 (therefore A6 as well), NA Elvs were filled for split sements. So those don't need to be corrected.

# %% Fill A3_ Elv NaNs from A5_ values
# We filled all NA values of A5
# A3_ Elv columns need to be filled

# %% NaN Elvs need to be filled in A3 too, if it is to be used for Elv corrections
A3_ = A3.copy()
A3_Elv_NaN_codes = A3_.loc[
    A3_['Elv_UStr'].isnull() | A3_['Elv_DStr'].isnull(), 'CODE'
].unique()  # list of codes where Elv is NaN
A5_Elv_NaN_in_A3 = A5.loc[A5['CODE'].isin(A3_Elv_NaN_codes), ['CODE', 'Elv_UStr', 'Elv_DStr']]  # DF for those codes
A5_Elv_UStr_max = A5_Elv_NaN_in_A3.groupby(['CODE'])['Elv_UStr'].max()
A5_Elv_DStr_min = A5_Elv_NaN_in_A3.groupby(['CODE'])['Elv_DStr'].min()
A3_.loc[A3_['CODE'].isin(A3_Elv_NaN_codes), 'Elv_UStr'] = A3_.loc[A3_['CODE'].isin(A3_Elv_NaN_codes), 'CODE'].map(
    A5_Elv_UStr_max
)
A3_.loc[A3_['CODE'].isin(A3_Elv_NaN_codes), 'Elv_DStr'] = A3_.loc[A3_['CODE'].isin(A3_Elv_NaN_codes), 'CODE'].map(
    A5_Elv_DStr_min
)

# %% Fill A3 Elv - Alt method
A3__ = A3.copy()
A3___UStr_NA = A3__.loc[A3__['Elv_UStr'].isna()]  # Isolate upstream missing elevations
A3___DStr_NA = A3__.loc[A3__['Elv_DStr'].isna()]  # Isolate downstream missing elevations
A_Xas = xr.DataArray(A3___UStr_NA.Xa, dims='points')
A_Yas = xr.DataArray(A3___UStr_NA.Ya, dims='points')
A3___UStr_NA.shape

# %% Apply to Elv_UStr
mask_U = A3__['Elv_UStr'].isna()
if mask_U.any():
    print(f'Filling {mask_U.sum()} NaNs in Elv_UStr')
    A3__.loc[mask_U, 'Elv_UStr'] = rule_based_3ry_Elv(A3__.loc[mask_U, 'Xa'], A3__.loc[mask_U, 'Ya'], TOP, BOT)

# %% Apply to Elv_DStr
mask_D = A3__['Elv_DStr'].isna()
if mask_D.any():
    print(f'Filling {mask_D.sum()} NaNs in Elv_DStr')
    A3__.loc[mask_D, 'Elv_DStr'] = rule_based_3ry_Elv(A3__.loc[mask_D, 'Xz'], A3__.loc[mask_D, 'Yz'], TOP, BOT)
A3__[['CODE', 'Elv_UStr', 'Elv_DStr']].describe()
A3_[['CODE', 'Elv_UStr', 'Elv_DStr']].describe()

# %% [markdown]
# We'll procede with A3__. It should be more accurate, since it's calculated from TOP and BOT directly.<br>
# A3_ might miscalculate when a segment was split into multiple and an intermediate point has highe/r/lower elevation than the start/end points.

# %% Correct Elvs
n = 1
for i in A5.loc[
    (~A5['CODE'].isin(A4.loc[A4['Elv_UStr'].isna() & A4['Elv_DStr'].isna(), 'CODE'])) & (A5['split'] == 1), 'CODE'
].unique():  # [::-1]: # [26:27]: #[n:n+1]:
    print(
        f'{i} - {n:4}/{
            len(
                A5.loc[
                    (~A5["CODE"].isin(A4.loc[A4["Elv_UStr"].isna() & A4["Elv_DStr"].isna(), "CODE"]))
                    & (A5["split"] == 1),
                    "CODE",
                ].unique()
            ):4}',
        end=' - ',
    )
    n += 1

    # Prep GDF and Se for CODE
    GDF = A5.loc[A5['CODE'] == i, l_A5_Cols]  # Create GDF for the current CODE
    Se = A3_.loc[A3_['CODE'] == i]
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

        Coo_prev = (GDF.loc[GDF['ID'] == ID, 'Xz'].values[0], GDF.loc[GDF['ID'] == ID, 'Yz'].values[0])
        ID_prev = ID
        try:
            ID = GDF.loc[
                GDF.apply(lambda R: np.isclose((R.Xa, R.Ya), Coo_prev, atol=0.0001).all(), axis=1), 'ID'
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
        A3_.loc[A3_['CODE'] == i, 'Elv_DStr'],
        GDF.loc[GDF['ID'] == ID, 'Elv_UStr'].values[0] - drop * GDF.loc[GDF['ID'] == ID, 'C'].values[0],
        atol=0.1,
    ).any()
    if not check:
        print(
            f'Discrepancy in Elv_DStr for CODE {i}: calculated {GDF.loc[GDF["ID"] == ID, "Elv_DStr"].values[0]}, expected {A3_.loc[A3_["CODE"] == i, "Elv_DStr"].values[0]}'
        )
        l_closing_error.append(i)
    else:
        print('Elvs calculated correctly.')
        l_success.append(i)
        A6.loc[A6['CODE'] == i, 'Elv_UStr'] = GDF['Elv_UStr'].values
        A6.loc[A6['CODE'] == i, 'Elv_DStr'] = GDF['Elv_DStr'].values

Tot = len(l_success) + len(l_infinite_loop) + len(l_closing_error)
print(f'Successful: {len(l_success)} ({round(len(l_success) * 100 / Tot, 2)} %)')
print(f'Infinite loop: {len(l_infinite_loop)} ({round(len(l_infinite_loop) * 100 / Tot, 2)} %)')
print(f'Closing error: {len(l_closing_error)} ({round(len(l_closing_error) * 100 / Tot, 2)} %)')

# %% [markdown]
# Most are successful. We'll proceed for now, and hope SFRmaker will handle the rest. Otherwise we can come back.

# %% Inspect Out path
Pa_Out

# %% Save
Pa_Out.parent.mkdir(parents=True, exist_ok=True)
A6_out = A6.copy()
A6_out['DStr_code'] = A6_out['DStr_code'].astype('string')
A6_out['DStr_ID'] = A6_out['DStr_ID'].astype('Int64')
A6_out['split'] = A6_out['split'].astype('Int64')
A6_out = A6_out[
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
A6_out.to_file(Pa_Out, driver='GPKG', engine='fiona', index=False)

# %%
