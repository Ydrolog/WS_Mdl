import os
from os import makedirs as MDs
from os.path import dirname as PDN
from os.path import join as PJ

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import LineString, Point
from WS_Mdl.core.df import round_Cols
from WS_Mdl.core.mdl import Mdl_N
from WS_Mdl.core.style import Sep, set_verbose, sprint
from WS_Mdl.imod.sfr.info import SFR_ConnD_to_DF, SFR_PkgD_to_DF


def Par_to_Rst(
    MdlN: str, Par: str, crs: str = 28992, Pa_SFR=None, radius: float = None, iMOD5=False, verbose: bool = True
):
    """
    Creates a raster out of a parameter of an SFR file. Parameter needs to be typed exactly as in the PACKAGEDATA DF header (1st commented out line in PACKAGEDATA)
    """

    # --- Prep ---
    set_verbose(verbose)
    M = Mdl_N(MdlN)
    d_Pa = M.Pa

    if Pa_SFR is None:
        Pa_SFR = d_Pa['SFR']

    if not os.path.exists(Pa_SFR):
        sprint(f'🔴 ERROR: SFR file not found at {Pa_SFR}. Cannot proceed.')

    d_INI = M.INI
    Xmin, Ymin, Xmax, Ymax = [float(i) for i in d_INI['WINDOW'].split(',')]
    cellsize = float(d_INI['CELLSIZE'])
    N_R, N_C = int(-(Ymin - Ymax) / cellsize), int((Xmax - Xmin) / cellsize)

    # --- Load PACKAGEDATA DF ---
    DF_PkgDt = SFR_PkgD_to_DF(MdlN, Pa_SFR=Pa_SFR, iMOD5=iMOD5)

    Pa_Out = PJ(d_Pa['PoP'], f'In/SFR/{MdlN}/SFR_{Par}_{MdlN}.tif')

    # --- Create & Fill Array ---
    Arr = np.full((N_R, N_C), np.nan)  # Create empty array
    Arr[DF_PkgDt['i'].astype(int) - 1, DF_PkgDt['j'].astype(int) - 1] = DF_PkgDt[Par]  # Populate array using i, j

    # --- Save ---
    MDs(PDN(Pa_Out), exist_ok=True)
    transform = from_bounds(Xmin, Ymin, Xmax, Ymax, N_C, N_R)

    with rasterio.open(
        Pa_Out,
        'w',
        driver='GTiff',
        height=N_R,
        width=N_C,
        count=1,
        dtype=Arr.dtype,
        crs=crs,
        transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(Arr, 1)

    sprint(f'🟢🟢🟢 - Saved to {Pa_Out}')
    print(Sep)


def SFR_to_GPkg(MdlN: str, crs: str = 28992, Pa_SFR=None, radius: float = None, iMOD5=False, verbose: bool = True):
    """
    Reads SFR package file and converts it to a GeoDataFrame, then saves it as a GPkg file.
    ATM assumes that line right after 'BEGIN PACKAGEDATA' is the header line. This could be improved in the future.
    """

    # --- Prep ---
    set_verbose(verbose)
    M = Mdl_N(MdlN)
    d_Pa = M.Pa

    if Pa_SFR is None:
        Pa_SFR = d_Pa['SFR']

    if not os.path.exists(Pa_SFR):
        sprint(f'🔴 ERROR: SFR file not found at {Pa_SFR}. Cannot proceed.')

    if radius is None:  # If radius s not provided, use CELLSIZE from INI file
        set_verbose(False)
        d_INI = M.INI
        radius = float(d_INI['CELLSIZE'])
        set_verbose(verbose)

    # --- Load PACKAGEDATA DF ---
    DF_PkgDt = SFR_PkgD_to_DF(MdlN, Pa_SFR=Pa_SFR, iMOD5=iMOD5)

    # --- Load CONNECTIONDATA ---
    DF_Conn = SFR_ConnD_to_DF(MdlN, Pa_SFR=Pa_SFR, iMOD5=iMOD5)

    ## --- Merge ---
    if 'rno' in DF_PkgDt.columns:
        left_merge = 'rno'
    elif 'ifno' in DF_PkgDt.columns:
        left_merge = 'ifno'
    else:
        sprint('🔴 ERROR: Neither "rno" nor "ifno" columns found in PACKAGEDATA. Cannot merge with CONNECTIONDATA.')

    DF = pd.merge(DF_PkgDt, DF_Conn[['reach_N', 'downstream']], left_on=left_merge, right_on='reach_N', how='left')

    DF.insert(0, 'reach_N', DF.pop('reach_N'))
    DF.drop(left_merge, axis=1, inplace=True)
    DF = pd.merge(
        DF,
        DF[['reach_N', 'X', 'Y']].rename(columns={'reach_N': 'downstream', 'X': 'DStr_X', 'Y': 'DStr_Y'}),
        on='downstream',
        how='left',
    )

    # --- Identify Outlets ---
    # Build upstream map: downstream_id -> list of upstream_ids
    upstream_map = DF[DF['downstream'].notna()].groupby('downstream')['reach_N'].apply(list).to_dict()

    # Identify outlets (reaches with no downstream reach)
    outlets = DF[DF['downstream'].isna()]['reach_N'].tolist()

    # Dictionary to store reach -> outlet mapping
    reach_to_outlet = {outlet: outlet for outlet in outlets}

    # Propagate outlet IDs upstream
    current_layer = outlets
    while current_layer:
        next_layer = []
        for current_r in current_layer:
            current_outlet = reach_to_outlet[current_r]
            upstream_reaches = upstream_map.get(current_r, [])
            for up_r in upstream_reaches:
                if up_r not in reach_to_outlet:
                    reach_to_outlet[up_r] = current_outlet
                    next_layer.append(up_r)
        current_layer = next_layer

    # Map back to DF
    DF['outlet_id'] = DF['reach_N'].map(reach_to_outlet)

    # --- Create geometry for all reaches ---
    # Add reach type column to identify routing vs outlets
    DF['reach_type'] = DF.apply(
        lambda row: 'routing' if pd.notnull(row['DStr_X']) and pd.notnull(row['DStr_Y']) else 'outlet', axis=1
    )

    # Create LineString geometries for all reaches
    DF['geometry'] = DF.apply(
        lambda row: (
            LineString([(row['X'], row['Y']), (row['DStr_X'], row['DStr_Y'])])
            if pd.notnull(row['DStr_X']) and pd.notnull(row['DStr_Y'])
            else Point(row['X'], row['Y']).buffer(radius)
        ),
        axis=1,
    )

    DF = round_Cols(DF)

    # Create duplicates of outlet reaches for the routing layer (as LineStrings)
    DF_outlet_duplicates = DF[DF['reach_type'] == 'outlet'].copy()
    DF_outlet_duplicates['geometry'] = DF_outlet_duplicates.apply(
        lambda row: LineString([(row['X'], row['Y']), (row['X'], row['Y'])]),
        axis=1,
    )

    # --- Save to GPKG as separate layers ---
    Pa_SHP = PJ(d_Pa['PoP'], f'In/SFR/{MdlN}/SFR_{MdlN}.gpkg')
    os.makedirs(PDN(Pa_SHP), exist_ok=True)

    # Prepare routing layer: regular routing reaches + outlet duplicates (as LineStrings)
    DF_routing = DF[DF['reach_type'] == 'routing'].copy()
    DF_routing_combined = pd.concat([DF_routing, DF_outlet_duplicates], ignore_index=True)

    # Prepare outlets layer: outlet reaches with buffered polygon geometry
    DF_outlets = DF[DF['reach_type'] == 'outlet'].copy()

    # Save routing layer (including outlet duplicates as LineStrings)
    if not DF_routing_combined.empty:
        GDF_routing = gpd.GeoDataFrame(DF_routing_combined, geometry='geometry', crs=crs)
        GDF_routing.to_file(Pa_SHP, driver='GPKG', layer=f'SFR_{MdlN}_routing')
        routing_count = len(DF_routing)
        outlet_dup_count = len(DF_outlet_duplicates)
        sprint(
            f'🟢 - SFR routing layer saved with {routing_count} routing + {outlet_dup_count} outlet LineStrings = {len(GDF_routing)} total features'
        )

    # Save outlets layer if it has data
    if not DF_outlets.empty:
        GDF_outlets = gpd.GeoDataFrame(DF_outlets, geometry='geometry', crs=crs)
        GDF_outlets.to_file(Pa_SHP, driver='GPKG', layer=f'SFR_{MdlN}_Outlets')
        sprint(f'🟢 - SFR outlets layer saved with {len(GDF_outlets)} LineString features (outlets)')

    sprint(f'🟢🟢 - SFR for {MdlN} has been converted to Gpkg and saved at:\n\t{Pa_SHP}\n\t')
