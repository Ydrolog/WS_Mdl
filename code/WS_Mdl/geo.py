from .utils import Sign, Pre_Sign
from . import utils as U
from . import utils_imod as UIM
import os
from os import listdir as LD, makedirs as MDs
from os.path import join as PJ, basename as PBN, dirname as PDN, exists as PE
import shutil as sh
import re
from datetime import datetime as DT
from typing import Optional, Dict
import rasterio
from rasterio.transform import from_bounds
import imod
import numpy as np
import xarray as xr
from pathlib import Path
import pandas as pd
import geopandas as gpd
from concurrent.futures import ProcessPoolExecutor as PPE
import zipfile as ZF
import xml.etree.ElementTree as ET
import sys

crs = "EPSG:28992"

# TIF ------------------------------------------------------------------------------------------------------------------------------
def DA_stats(DA, verbose:bool=False):
    d_stats =  {'mean'      :   DA.mean().values,
                'sum'       :   DA.sum().values,
                'max'       :   DA.max().values,
                'min'       :   DA.min().values,
                'std'       :   DA.std().values}
    if verbose:
        print(d_stats)

    return d_stats

def IDF_to_TIF(Pa_IDF: str, Pa_TIF: Optional[str] = None, MtDt: Optional[Dict] = None, crs=crs):
    """ Converts IDF file to TIF file.
        If Pa_TIF is not provided, it'll be the same as Pa_IDF, except for the file type ending.
        crs (coordinate reference system) is set to the Amerfoot crs by default, but can be changed for other projects."""
    print(Pre_Sign)
    try:
        A, MtDt = imod.idf.read(Pa_IDF)

        Ogn_DT = DT.fromtimestamp(os.path.getctime(Pa_IDF)).strftime('%Y-%m-%d %H:%M:%S') # Get OG (IDF) file's date modified.
        Cvt_DT = DT.now().strftime('%Y-%m-%d %H:%M:%S') # Get current time, to write time of convertion to comment

        N_R, N_C = A.shape

        transform = from_bounds(west    =   MtDt['xmin'],
                                south   =   MtDt['ymin'],
                                east    =   MtDt['xmax'],
                                north   =   MtDt['ymax'],
                                width   =   N_C,
                                height  =   N_R)
        meta = {"driver"    :   "GTiff",
                "height"    :   N_R,
                "width"     :   N_C,
                "count"     :   1,
                "dtype"     :   str(A.dtype),
                "crs"       :   crs,
                "transform" :   transform}

        if not Pa_TIF:
            Pa_TIF = os.path.splitext(Pa_IDF)[0] + '.tif'

        with rasterio.open(Pa_TIF, "w", **meta) as Dst:
            Dst.write(A, 1)  # Write band 1

            Cvt_MtDt = {'COMMENT':( f"Converted from IDF on {Cvt_DT}."
                                    f"Original file created on {Ogn_DT}."
                                    f"Original IDF file location: {Pa_IDF}")}
                    
            if MtDt: # If project metadata exists, store it separately
                project_metadata = {f"USER_{k}": str(v) for k, v in MtDt.items()}
                Cvt_MtDt.update(project_metadata)

            Dst.update_tags(**Cvt_MtDt)
        print(f"🟢 {Pa_TIF} has been saved (GeoTIFF) with conversion and project metadata.")
    except Exception as e:
        print(f"\u274C \n{e}")
    print(Sign)

    # def l_IDF_to_TIF(l_IDF, Dir_Out):
    #     """#666 under construction. The aim of this is to make a multi-band tif file instead of multiple single-band tif files, for each parameter."""
    #     DA = imod.formats.idf.open(l_IDF)#.sel(x=slice(Xmin, Xmax), y=slice(Ymax, Ymin)) # Read IDF files to am xarray.DataArray and slice it to model area (read from INI file)
    #     DA = DA.rio.write_crs(crs)  # Set Dutch RD New projection
    #     DA.rio.to_raster(Dir_Out)

def DA_to_TIF(DA, Pa_Out, d_MtDt, crs=crs, _print=False):
    """ Write a 2D xarray.DataArray (shape = [y, x]) to a single-band GeoTIFF.
    - DA: 2D xarray.DataArray with shape [y, x]
    - Pa_Out: Path to the output GeoTIFF file.
    - d_MtDt: Dictionary with metadata for this single band.
      Must contain exactly 1 item: {band_description: band_metadata_dict}
    - crs: Coordinate Reference System (optional)."""    
    
    if len(d_MtDt) != 1: # We expect exactly one band, so parse the single (key, value) from d_MtDt
        raise ValueError("DA_to_TIF expects exactly 1 item in d_MtDt for a 2D DataArray.")

    (band_key, band_meta) = list(d_MtDt.items())[0]

    transform = DA.rio.transform() # Build transform from DA

    with rasterio.open(Pa_Out,
                       "w",
                       driver="GTiff",
                       height=DA.shape[0],
                       width=DA.shape[1],
                       count=1,                   # single band
                       dtype=str(DA.dtype),
                       crs=crs,
                       transform=transform) as Dst:
        Dst.write(DA.values, 1) # Write the 2D data as band 1
        Dst.set_band_description(1, band_key) # Give the band a useful name
        Dst.update_tags(1, **band_meta) # Write each row field as a separate metadata tag on this band
    if _print:
        print(f"DA_to_TIF finished successfully for: {Pa_Out}")

def DA_to_MBTIF(DA, Pa_Out, d_MtDt, crs=crs, _print=False):
    """ Write a 3D xarray.DataArray (shape = [n_bands, y, x]) to a GeoTIFF. This bypasses rioxarray.to_raster() entirely, letting us set per-band descriptions and metadata in a single pass.
    - DA: 3D xarray.DataArray with shape [n_bands, y, x]
    - Pa_Out: Path to the output GeoTIFF file.
    - d_MtDt: Dictionary with metadata to be written to the GeoTIFF file. Each key is a band index (1-based) and each value is a dictionary of metadata tags.
    - crs: Coordinate Reference System (optional)."""

    band_keys, band_MtDt = zip(*d_MtDt.items())

    transform = DA.rio.transform()

    with rasterio.open(Pa_Out, #666 add ask-to-overwrite function (preferably to any function/command in this Lib that writes a file.)
                       "w",
                       driver="GTiff",
                       height=DA.shape[1],
                       width=DA.shape[2],
                       count=DA.shape[0],
                       dtype=str(DA.dtype),
                       crs=crs,
                       transform=transform,
                       photometric="MINISBLACK") as Dst:
        for i in range(DA.shape[0]): # Write each band.
            Dst.write(DA[i].values, i + 1) # Write the actual pixels for this band (i+1 is the band index in Rasterio)
            Dst.set_band_description(i + 1, band_keys[i]) # Set a band description that QGIS will show as "Band 01: <description>"
            Dst.update_tags(i + 1, **band_MtDt[i]) # Write each row field as a separate metadata tag on this band

        if "all" in d_MtDt: # If "all" exists, write dataset-wide metadata (NOT tied to a band)
            Dst.update_tags(**d_MtDt["all"])  # Set global metadata for the whole dataset
            
    if _print:
        print(f"DA_to_MBTIF finished successfully for: {Pa_Out}")

def PRJ_to_TIF(MdlN):
    """ Converts PRJ file to TIF (multiband if necessary) files by package (only time independent packages).
    The function uses a DF produced by PRJ_to_DF. It needs to follow a specific format.
    Also creates a .csv file with the TIF file paths to be replaced in the QGIS project."""
    
    # -------------------- Initiate ------------------------------------------------------------------------------------------
    d_Pa = U.get_MdlN_paths(MdlN)                                                        # Get paths
    Xmin, Ymin, Xmax, Ymax, cellsize, N_R, N_C = U.Mdl_Dmns_from_INI(d_Pa['Pa_INI'])   # Get dimensions
    
    DF = UIM.PRJ_to_DF(MdlN) # Read PRJ file to DF
    
    # -------------------- Process time-indepenent packages (most) ------------------------------------------------------------
    print(f'\n --- Converting time-independant package IDF files to TIF ---')
    DF_Rgu = DF[( DF["time"].isna()   ) &       # Only keep regular (time independent) packages
                ( DF['path'].notna()  ) &  
                ( DF['suffix']=='.idf')]        # Non time packages have NaN in 'time' Fld. Failed packages have '-', so they'll also be excluded.

    for i, Par in enumerate(DF_Rgu['parameter'].unique()[:]): # Iterate over parameters
        print(f"\t{i:<2}, {Par:<30} ... ", end='')

        try:
            DF_Par = DF_Rgu[DF_Rgu['parameter']==Par] # Slice DF_Rgu for current parameter.
            DF_Par = DF_Par.drop_duplicates(subset='path', keep='first') # Drop duplicates, keep the first one. imod.formats.idf.open will do that with the list of paths anyway, so the only way to match the paths to the correct metadata is to have only one path per metadata.
            if DF_Par['package'].nunique() > 1:
                print("There are multiple packages for the same parameter. Check DF_Rgu.")
                break
            else:
                Pkg = DF_Par['package'].iloc[0] # Get the package name

            ## Prepare directoreis and filenames
            Mdl = ''.join([c for c in MdlN if not c.isdigit()])
            Pkg_MdlN = Mdl + str(DF_Par['MdlN'].str.extract(r'(\d+)').astype(int).max().values[0])
            Pa_TIF = PJ(d_Pa['Pa_Mdl'], 'PoP', 'In', Pkg, Pkg_MdlN, f"{Pkg}_{Par}_{Pkg_MdlN}.tif")  # Full path to TIF file

            ## Build a dictionary mapping each band’s name to its row’s metadata. We're assuming that the order the paths are read into DA is the same as the order in DF_Par.
            d_MtDt = {}

            if os.path.exists(Pa_TIF):
                print(f'\u274C - {PBN(Pa_TIF)} already exists. Skipping.')
                continue
            else:
                MDs(PDN(Pa_TIF), exist_ok=True) # Make sure the directory exists

                ## Read files-paths to xarray Data Array (DA), then write them to TIF file(s).
                if DF_Par.shape[0] > 1: # If there are multiple paths for the same parameter
                    for i, R in DF_Par.iterrows():
                        d_MtDt[f"{R['parameter']}_L{R['layer']}_{R['MdlN']}"] = {('origin_path' if col == 'path' else col):
                                                                                str(val) for col, val in R.items()}
                    DA = imod.formats.idf.open(list(DF_Par['path']), pattern="{name}_L{layer}_").sel(x=slice(Xmin, Xmax), y=slice(Ymax, Ymin))
                    DA_to_MBTIF(DA, Pa_TIF, d_MtDt)
                    print(f'🟢 - multi-band')
                else:
                    try:
                        DA = imod.formats.idf.open(list(DF_Par['path']), pattern="{name}_L{layer}_").sel(x=slice(Xmin, Xmax), y=slice(Ymax, Ymin))
                        d_MtDt[f"{DF_Par['parameter'].values[0]}_L{DF_Par['layer'].values[0]}_{DF_Par['MdlN'].values[0]}"] = {('origin_path' if col == 'path' else col): str(val) for col, val in R.items()}
                        DA_to_TIF(DA.squeeze(drop=True), Pa_TIF, d_MtDt) # .squeeze cause 2D arrays have extra dimension with size 1 sometimes.
                        print(f'🟢 - single-band with L attribute')
                    except:
                        DA = imod.formats.idf.open(list(DF_Par['path']), pattern="{name}_").sel(x=slice(Xmin, Xmax), y=slice(Ymax, Ymin))
                        d_MtDt[f"{DF_Par['parameter'].values[0]}_{DF_Par['MdlN'].values[0]}"] = {('origin_path' if col == 'path' else col): str(val) for col, val in R.items()}
                        DA_to_TIF(DA.squeeze(drop=True), Pa_TIF, d_MtDt) # .squeeze cause 2D arrays have extra dimension with size 1 sometimes.
                        print(f'🟢 - single-band without L attribute')
        except Exception as e:
            print(f"\u274C - Error: {e}")
    
    # -------------------- Process time-dependent packages (RIV, DRN, WEL) --------------------
    ## RIV & DRN
    print(f'\n --- Converting time dependant packages ---')
    DF_time = DF[ ( DF["time"].notna() ) &
                ( DF["time"]!='-'    ) &
                ( DF['path'].notna() )] # Non time packages have NaN in 'time' Fld. Failed packages have '-', so they'll also be excluded.         

    for i, R in DF_time[DF_time['package'].isin(('DRN', 'RIV'))].iterrows():
        print(f"\t{f"{R['package']}_{R['parameter']}":<30} ... ", end='')

        Pa_TIF = PJ(d_Pa['Pa_Mdl'], 'PoP', 'In', R['package'], R['MdlN'], PBN(re.sub(r'\.idf$', '.tif', R['path'], flags=re.IGNORECASE)))  # Full path to TIF file
        
        if os.path.exists(Pa_TIF):
            print(f'\u274C - {PBN(Pa_TIF)} already exists. Skipping.')
            continue
        else:
            try:    
                MDs(PDN(Pa_TIF), exist_ok=True) # Make sure the directory exists

                ## Build a dictionary mapping each band’s name to its row’s metadata.
                d_MtDt = {f"{R['parameter']}_L{R['layer']}_{R['MdlN']}" : {('origin_path' if col == 'path' else col): str(val) for col, val in R.items()}}

                DA = imod.formats.idf.open(R['path'], pattern=f'{{name}}_{Mdl}').sel(x=slice(Xmin, Xmax), y=slice(Ymax, Ymin))
                DA_to_TIF(DA.squeeze(drop=True), Pa_TIF, d_MtDt) # .squeeze cause 2D arrays have extra dimension with size 1 sometimes.
                print(f'🟢 - IDF converted to TIF - single-band without L attribute')
            except Exception as e:
                print(f"\u274C - Error: {e}")

    ## WEL
    DF_WEL = DF.loc[DF['package']=='WEL']

    for i, R in DF_WEL.iloc[3:6].iterrows():
        print(f"\t{PBN(R['path']):<30} ... ", end='')

        Pa_GPKG = PJ(d_Pa['Pa_Mdl'], 'PoP', 'In', R['package'], R['MdlN'], PBN(re.sub(r'\.ipf$', '.gpkg', R['path'], flags=re.IGNORECASE)))  # Full path to TIF file

        if os.path.exists(Pa_GPKG):
            print(f'\u274C - file {PBN(Pa_GPKG)} exists. Skipping.')
            continue
        else:

            try:
                DF_IPF = imod.formats.ipf.read(R['path'])
                DF_IPF = DF_IPF.loc[ ( (DF_IPF['x']>Xmin) & (DF_IPF['x']<Xmax ) ) &
                                    ( (DF_IPF['y']>Ymin) & (DF_IPF['y']<Ymax ) )].copy() # Slice to OBS within the Mdl Aa
                
                if ('q_m3' in DF_IPF.columns) and ('id' not in DF_IPF.columns):
                    DF_IPF.rename(columns={'q_m3':'id'}, inplace=True) # One of the IPF files has q_m3 instead of id in it's fields. Don't ask me why, but it has to be dealt with.

                #666 I'll only save the average flow now
                DF_IPF_AVG = DF_IPF.groupby('id')[DF_IPF.select_dtypes(include=np.number).columns].agg(np.mean)
                _GDF_AVG = gpd.GeoDataFrame(DF_IPF_AVG, geometry=gpd.points_from_xy(DF_IPF_AVG['x'], DF_IPF_AVG['y'])).set_crs(crs=crs)

                MDs(PDN(Pa_GPKG), exist_ok=True) # Make sure the directory exists
                _GDF_AVG.to_file(Pa_GPKG, driver="GPKG") #, layer=PBN(Pa_GPKG))
                print(f'🟢 - IPF average values (per id) converted to GPKG')
            except:
                print('\u274C')
    # -------------------- Process derived packages/parameters (Thk, T) -------------------------------------------------------
    d_Clc_In = {} # Dictionary to store calculated inputs.

    ## Thk. TOP and BOT files have been QA'd in C:\OD\WS_Mdl\code\PrP\Mdl_In_to_MM\Mdl_In_to_MM.ipynb
    print(f' --- Converting calculated inputs to TIF ---')

    DA_TOP = imod.formats.idf.open(list(DF_Rgu[DF_Rgu['parameter']=='top']['path']), pattern="{name}_L{layer}_").sel(x=slice(Xmin, Xmax), y=slice(Ymax, Ymin))
    DA_BOT = imod.formats.idf.open(list(DF_Rgu[DF_Rgu['parameter']=='bottom']['path']), pattern="{name}_L{layer}_").sel(x=slice(Xmin, Xmax), y=slice(Ymax, Ymin))
    DA_Kh = imod.formats.idf.open(list(DF_Rgu[DF_Rgu['parameter']=='kh']['path']), pattern="{name}_L{layer}_").sel(x=slice(Xmin, Xmax), y=slice(Ymax, Ymin))

    DA_Thk = (DA_TOP - DA_BOT).squeeze(drop=True) # Let's make a dictionary to store Info about each parameter
    MdlN_Pkg = Mdl + str(max(DF_Rgu.loc[DF_Rgu['package'].isin(['TOP', 'BOT']), 'MdlN'].str.extract(r'(\d+)')[0])) #666 the largest number from the TOP and BOT MdlNs
    d_Clc_In['Thk'] = {'Par': 'thickness',
        'DA': DA_Thk,
        'MdlN_Pkg': MdlN_Pkg,
        'MtDt': {**{f"thickness_L{i+1}_{MdlN_Pkg}": {"layer": f"L{i+1}"} for i in range(DA_Thk.shape[0])},  # Per-layer metadata
                 "all": {"description": "Layer thickness calculated as 'top - bottom' per layer.",
                 "source_files": f"""{'-'*200}\nTOP: {' '*30} {" | | ".join(DF_Rgu.loc[DF_Rgu['package'] == 'TOP', 'path'])}        {'-'*200}\nBOT: {' '*30} {" | | ".join(DF_Rgu.loc[DF_Rgu['package'] == 'BOT', 'path'])} """}}}
    ## T
    DA_T = DA_Thk * DA_Kh
    MdlN_Pkg = Mdl + str(max(DF_Rgu.loc[DF_Rgu['package'].isin(['TOP', 'BOT', 'NPF']), 'MdlN'].str.extract(r'(\d+)')[0])) # the largest number from the TOP and BOT MdlNs
    d_Clc_In['T'] = {'Par': 'transmissivity',
        'DA': DA_T,
        'MdlN_Pkg': MdlN_Pkg,                    
        'MtDt': {**{f"transmissivity_L{i+1}_{MdlN_Pkg}": {"layer": f"L{i+1}"} for i in range(DA_Thk.shape[0])},  # Per-layer metadata
        'all': {"description": "Layer transmissivity (horizontal) calculated as '(top - bottom)*Kh' per layer.",
                "source_files": f"""{'-'*200}TOP: {' '*30} {" | | ".join(DF_Rgu.loc[DF_Rgu['package'] == 'TOP', 'path'])} 
                    {'-'*200}BOT: {' '*30} {" | | ".join(DF_Rgu.loc[DF_Rgu['package'] == 'BOT', 'path'])}
                    {'-'*200}NPF: {' '*30} {" | | ".join(DF_Rgu.loc[DF_Rgu['package'] == 'NPF', 'path'])}"""}}}
    
    for i, Par in enumerate(d_Clc_In.keys()):
        print(f"\t{d_Clc_In[Par]['Par']:<30} ... ", end='')

        Pa_TIF = PJ(d_Pa['Pa_Mdl'], 'PoP', 'Clc_In', Par, d_Clc_In[Par]['MdlN_Pkg'], f"{Par}_{d_Clc_In[Par]['MdlN_Pkg']}.tif")  # Full path to TIF file #666 need to think which MdlN to use. It's hard to do the same as with the other packages.
        
        if os.path.exists(Pa_TIF):
            print(f'\u274C - {PBN(Pa_TIF)} already exists. Skipping.')
            continue
        else:
            try:
                MDs(PDN(Pa_TIF), exist_ok=True) # Make sure the directory exists

                ## Write DAs to TIF files.
                DA = d_Clc_In[Par]['DA'].squeeze(drop=True)
                d_MtDt = d_Clc_In[Par]['MtDt']

                if not DA.rio.crs: # Ensure DA_Thk has a CRS (if missing, set it)
                    DA.rio.write_crs(crs, inplace=True)  # Replace with correct CRS

                if len(DA.shape) == 3: # If there are multiple paths for the same parameter
                    DA_to_MBTIF(DA, Pa_TIF, d_MtDt)
                    print(f'🟢 - multi-band')
                elif len(DA.shape) == 2:
                    DA_to_TIF(DA.squeeze(drop=True), Pa_TIF, d_MtDt) # .squeeze cause 2D arrays have extra dimension with size 1 sometimes.
                    print(f'🟢 - single-band')
            except Exception as e:
                print(f"\u274C - Error: {e}")
    print(Sign)

# HD_IDF speciic PoP (could be extended/generalized at a later stage) --------------------------------------------------------------
def HD_IDF_Agg_to_TIF(MdlN:str, rules=None, N_cores:int = None, crs:str = crs, Gp: list[str] = ['year', 'month'], Agg_F: str = 'mean'):
    """
    General wrapper to:
      1) read all IDF metadata into a DataFrame,
      2) filter by `rules`,
      3) add any needed Gp columns (season, Hy_year, quarter),
      4) group by `Gp`,
      5) for each group, apply `agg_func` along time and write a single‐band TIFF.

    Parameters
    ----------
    MdlN : str
        Model name (e.g. 'NBr13').
    rules : None or str
        A pandas-query string to subset/filter the IDF-DF before Gp (e.g. "(L == 1)").
    N_cores : int or None
        Number of worker processes for parallel execution. By default: None → use (cpu_count() - 2).
    crs : str
        Coordinate reference system for the output TIFs. By default: G.crs.
    Gp : list of str
        Which DataFrame columns to group by. Common examples:
        - ['year','month']        → monthly aggregates
        - ['season_year','season']→ seasonal aggregates
        - ['Hy_year']             → hydrological‐year aggregates
        - ['year','quarter']      → quarterly aggregates
    agg_func : str
        Name of the aggregation method to call on the xarray.DataArray (e.g. 'mean','min','max','median').
        This must exactly match a DataArray method (e.g. XA.mean(dim='time')).
    """
    print(Pre_Sign)
    print(f"*** {MdlN} *** - HD_IDF_Agg_to_TIF\n")

    # 1. Get paths
    d_Pa = U.get_MdlN_paths(MdlN)
    Pa_PoP, Pa_HD = [ d_Pa[v] for v in ['Pa_PoP', 'Pa_Out_HD'] ]

    # 2. Read the IDF files to DF. Add extracols. Apply rules. Group.
    DF = U.HD_Out_IDF_to_DF(Pa_HD)
    if rules:
        DF = DF.query(rules)
    DF_Gp = DF.groupby(Gp)['path']
    
    # 3. Prep Out Dir
    Pa_Out_Dir  = PJ(Pa_PoP, f'Out/{MdlN}/HD_Agg')
    os.makedirs(Pa_Out_Dir, exist_ok=True)
    
    # 4. Decide N of cores
    if N_cores is None:
        N_cores = max(os.cpu_count() - 2, 1)    # Leave 2 cores free for other tasks by default. If there aren't enough cores available, set to 1.
    
    # 5. Launch one job per group
    start = DT.now()
    with PPE(max_workers=N_cores) as E:
        futures = []
        for Gp_keys, paths in DF_Gp:
            group_name = HD_Agg_name(Gp_keys, Gp) # user‐defined helper to turn keys → a nice string, e.g. "2010_1" or "2020_Winter"

            # we’ll write one single‐band GeoTiff per group
            Pa_Out = PJ(Pa_Out_Dir, f"HD_{group_name}_{MdlN}.tif")

            params = {'MdlN': str(MdlN), 'N_cores': str(N_cores), 'crs': str(crs), 'rules': str(rules)}

            futures.append(E.submit(_HD_IDF_Agg_to_TIF_process,
                                    paths=list(paths),
                                    Agg_F=Agg_F,
                                    Pa_Out=Pa_Out,
                                    crs=crs,
                                    params=params))

        for f in futures: # wait & report
            print('\t', f.result(), 'elapsed:', DT.now() - start)

    print(f"🟢🟢🟢 | Total elapsed time: {DT.now() - start}")
    print(Sign)

def _HD_IDF_Agg_to_TIF_process(paths, Agg_F, Pa_Out, crs, params):
    """
    Only for use within HD_IDF_Mo_Avg_to_MBTIF - to utilize multiprocessing.
    Reads IDFs, aggregates along time, writes each layer as a single-band TIF.
    """
    XA     = imod.formats.idf.open(paths)
    XA_agg = getattr(XA, Agg_F)(dim='time')
    base   = Pa_Out[:-4]  # strip “.tif”
    for layer in XA_agg.layer.values:
        DA     = XA_agg.sel(layer=layer).drop_vars('layer')
        Out    = f"{base}_L{layer}.tif"
        d_MtDt = {f'{Agg_F}':  {'AVG'          :   float(DA.mean().values),
                                'coordinates'   :   XA.coords,
                                'variable'      :   os.path.splitext(PBN(Pa_Out))[0],
                                'details'       :   f'Calculated using WS_Mdl.geo.py using the following params: {params}'}}

        DA_to_TIF(DA, Out, d_MtDt, crs=crs)
    return f"{os.path.basename(base)} 🟢 "

def HD_Agg_name(group_keys, grouping): #666 could be moved to util
    if not isinstance(group_keys, (tuple, list)):
        group_keys = (group_keys,)

    if grouping == ['year', 'month']:           # year & month → "YYYYMM"
        year, month = group_keys
        return f"{year}{month:02d}"

    if grouping == ['month']:                   # month alone → "MM"
        (month,) = group_keys
        return f"{month:02d}"

    if grouping == ['year']:                    # year alone → "YYYY"
        (year,) = group_keys
        return str(year)

    if grouping == ['season_year', 'season']:   # season_year & season → "YYYY_Season"
        season_year, season = group_keys
        return f"{season_year}_{season}"

    if grouping == ['season']:                  # season alone → "Season"
        (season,) = group_keys
        return season

    if grouping == ['water_year']:              # water_year → "WYYY"
        (wy,) = group_keys
        return f"WY{wy}"

    if grouping == ['year', 'quarter']:         # year & quarter → "YYYY_Q#"
        year, quarter = group_keys
        return f"{year}_{quarter}"

    if grouping == ['quarter']:                 # quarter alone → "Q#"
        (quarter,) = group_keys
        return quarter

    return "_".join(str(k) for k in group_keys) # fallback: join all keys with underscore

def HD_IDF_GXG_to_TIF(MdlN: str, N_cores:int=None, crs:str=crs, rules:str=None):
    """Reads Sim Out IDF files from the model directory and calculates GXG for each L. Saves them as MultiBand TIF files - each band representing one of the GXG params for a L."""

    print(Pre_Sign)
    print(f"*** {MdlN} *** - HD_IDF_GXG_to_TIF\n")

    # Get paths
    d_Pa = U.get_MdlN_paths(MdlN)
    Pa_PoP, Pa_HD = [ d_Pa[v] for v in ['Pa_PoP', 'Pa_Out_HD'] ]

    # Read DF and apply rules to DF if rules is not None.
    DF = U.HD_Out_IDF_to_DF(Pa_HD)
    if rules is not None:
        DF = DF.query(rules)

    if N_cores is None:
        N_cores = max(os.cpu_count() - 2, 1)
    start = DT.now() # Start time

    with PPE(max_workers=N_cores) as E:
        futures = [E.submit(_HD_IDF_GXG_to_TIF_per_L, DF, L, MdlN, Pa_PoP, Pa_HD, crs)
                   for L in DF['L'].unique()]
        for f in futures:
            print('\t', f.result(), '- Elapsed time (from start):', DT.now() - start)

    print('🟢🟢🟢 | Total elapsed:', DT.now() - start)
    print(Sign)

def _HD_IDF_GXG_to_TIF_per_L(DF, L, MdlN, Pa_PoP, Pa_HD, crs):
    """Only for use within HD_IDF_GXG_to_TIF - to utilize multiprocessing."""

    # Load HD files corresponding to the L to an XA
    l_IDF_L = list(DF.loc[DF['L']==1, 'path'])
    XA = imod.idf.open(l_IDF_L)

    # Calculate Variables
    GXG = imod.evaluate.calculate_gxg(XA.squeeze())
    GXG = GXG.rename_vars({var: var.upper() for var in GXG.data_vars})
    N_years_GXG = GXG['N_YEARS_GXG'].values if GXG['N_YEARS_GXG'].values.max() != GXG['N_YEARS_GXG'].values.min() else int(GXG['N_YEARS_GXG'].values[0, 0])
    N_years_GVG = GXG['N_YEARS_GVG'].values if GXG['N_YEARS_GVG'].values.max() != GXG['N_YEARS_GVG'].values.min() else int(GXG['N_YEARS_GVG'].values[0, 0])
    GXG["GHG_m_GLG"] = GXG["GHG"] - GXG["GLG"]
    GXG = GXG[["GHG", "GLG", "GHG_m_GLG", "GVG"]]

    # Save to TIF
    MDs(PJ(Pa_PoP, 'Out', MdlN, 'GXG'), exist_ok=True)
    for V in GXG.data_vars:
        Pa_Out = PJ(Pa_PoP, 'Out', MdlN, 'GXG', f'{V}_L{L}_{MdlN}.tif')

        d_MtDt = {f'{V}_L{L}_{MdlN}':
                        {'AVG'          :   float(GXG[V].mean().values),
                        'coordinates'   :   XA.coords,
                        'N_years'       :   N_years_GVG if V=='GVG' else N_years_GXG,
                        'variable'          :   os.path.splitext(PBN(Pa_Out))[0],
                        'details'       :   f'{MdlN} {V} calculated from (path: {Pa_HD}), via function described in: https://deltares.github.io/imod-python/api/generated/evaluate/imod.evaluate.calculate_gxg.html'}}
        
        DA = GXG[V]
        DA_to_TIF(DA, Pa_Out, d_MtDt, crs=crs, _print=False)

    return f"L{L} 🟢"

# ----------------------------------------------------------------------------------------------------------------------------------

# MM Update ------------------------------------------------------------------------------------------------------------------------
def Up_MM(MdlN, MdlN_MM_B=None):
    """Updates the MM (QGIS projct containing model data)."""
    
    print(Pre_Sign)
    print(f" *****   Creating MM for {MdlN}   ***** ")
    
    d_Pa = U.get_MdlN_paths(MdlN)
    Pa_QGZ, Pa_QGZ_B = d_Pa['Pa_MM'], d_Pa['Pa_MM_B']
    Mdl = U.get_Mdl(MdlN)

    if MdlN_MM_B is not None: # Replace MdlN_B with another MdlN if requested.
        Pa_QGZ_B = Pa_QGZ_B.replace(d_Pa['MdlN_B'], MdlN_MM_B)

    MDs(PBN(Pa_QGZ), exist_ok=True)      # Ensure destination folder exists
    sh.copy(Pa_QGZ_B, Pa_QGZ)                               # Copy the QGIS file
    print(f"Copied QGIS project from {Pa_QGZ_B} to {Pa_QGZ}.\nUpdating layer path ...")

    Pa_temp = PJ(PDN(Pa_QGZ), 'temp') # Path to temporarily extract QGZ contents
    MDs(Pa_temp, exist_ok=True)

    with ZF.ZipFile(Pa_QGZ_B, 'r') as zip_ref:     # Unzip .qgz
        zip_ref.extractall(Pa_temp)

    Pa_QGS = PJ(Pa_temp, PBN(Pa_QGZ_B).replace('.qgz', '.qgs'))
    tree = ET.parse(Pa_QGS)
    root = tree.getroot()

    for i, DS in enumerate(root.iter('datasource')):                  # Update datasource paths
        DS_text = DS.text
        # print(i, DS_text)
        
        if not DS_text:
            # print(' - X - Not text')
            # print('-'*50)
            continue

        if '|' in DS_text:
            path, suffix = DS_text.split('|', 1)
        else:
            path, suffix = DS_text, ''

        if Mdl in path:
            matches = re.findall(rf'{re.escape(Mdl)}(\d+)', path)
            if len(set(matches)) > 1:
                print(f"🔴 ERROR: multiple non-identical {Mdl}Ns found in path: {matches}")
                sys.exit("Fix the path containing non-identical MdlNs, then re-run me.")
            else:
                MdlX = f"{Mdl}{matches[0]}"
                
                Pa_full = os.path.normpath(PJ(PDN(Pa_QGZ), path.replace(MdlX, MdlN)))
                if (MdlX != MdlN) and (os.path.exists(Pa_full)):
                    Pa_X = path.replace(MdlX, MdlN)
                    DS.text = f"{Pa_X}|{suffix}" if suffix else Pa_X
                    print(f" - 🟢 Updated {MdlX} → {MdlN} in {Pa_full}")
                # else:
                    # print(" - OK (no change)")
        # else:
        #     print(" - No Mdl in path")
        # print('-'*50)

    tree.write(Pa_QGS, encoding='utf-8', xml_declaration=True)    # Save the modified .qgs file

    with ZF.ZipFile(Pa_QGZ, 'w', ZF.ZIP_DEFLATED) as zipf:   # Zip back into .qgz
        for foldername, _, filenames in os.walk(Pa_temp):
            for filename in filenames:
                filepath = PJ(foldername, filename)
                arcname = os.path.relpath(filepath, Pa_temp)
                zipf.write(filepath, arcname)

    sh.rmtree(Pa_temp)  # Remove the temporary folder
    print(f"\n🟢🟢🟢 | MM for {MdlN} has been updated.")
    print(Sign)
# ----------------------------------------------------------------------------------------------------------------------------------

# OUTDATED -------------------------------------------------------------------------------------------------------------------------
def A_to_Raster_n_IDF(A, IDF_MtDt, Pa_Out, field='HD_L1', crs="EPSG:4326"):
    """ This was used in PoP_HD_IDF a long time ago and is now outdated."""
    # 1. Write a GeoTIFF raster with rasterio
    nrows, ncols = A.shape

    transform = from_bounds(
        west=IDF_MtDt['xmin'],
        south=IDF_MtDt['ymin'],
        east=IDF_MtDt['xmax'],
        north=IDF_MtDt['ymax'],
        width=ncols,
        height=nrows)

    meta = {
        "driver": "GTiff",
        "height": nrows,
        "width": ncols,
        "count": 1,
        "dtype": str(A.dtype),
        "crs": crs,        # use your known CRS here
        "transform": transform,
    }

    tif_path = Pa_Out + ".tif"
    with rasterio.open(tif_path, "w", **meta) as dst:
        dst.write(A, 1)  # Write band 1
    print(f"{tif_path} has been saved (GeoTIFF).")

    # 2. Write the same data as an iMOD IDF
    #    Create xarray DataArray with spatial coords
    x = IDF_MtDt['xmin'] + IDF_MtDt['dx'] * (0.5 + np.arange(ncols))
    # Common convention is top-to-bottom descending:
    # but if your 'ymax' < 'ymin', you'll invert accordingly.
    y = IDF_MtDt['ymax'] - IDF_MtDt['dy'] * (0.5 + np.arange(nrows))

    DA = xr.DataArray(A, coords={"y": y, "x": x}, dims=["y", "x"], name=field)

    # Write the IDF
    idf_path = Pa_Out + ".idf"
    imod.idf.write(idf_path, DA)
    print(f"{idf_path} has been saved (iMOD IDF).")
# ----------------------------------------------------------------------------------------------------------------------------------
