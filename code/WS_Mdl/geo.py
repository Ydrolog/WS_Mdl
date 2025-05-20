# ***** Geospatial Functions *****
import os
from .utils import Sign, Pre_Sign
from . import utils as Utl
import rasterio
from rasterio.transform import from_bounds
import imod
from datetime import datetime as DT
from typing import Optional, Dict
import numpy as np
import xarray as xr
from concurrent.futures import ProcessPoolExecutor as PPE
import re
from pathlib import Path

crs = "EPSG:28992"

# Convert --------------------------------------------------------------------------------------------------------------------------
def IDF_to_TIF(path_IDF: str, path_TIF: Optional[str] = None, MtDt: Optional[Dict] = None, crs=crs):
    """ Converts IDF file to TIF file.
        If path_TIF is not provided, it'll be the same as path_IDF, except for the file type ending.
        crs (coordinate reference system) is set to the Amerfoot crs by default, but can be changed for other projects."""
    print(Pre_Sign)
    try:
        A, MtDt = imod.idf.read(path_IDF)

        Ogn_DT = DT.fromtimestamp(os.path.getctime(path_IDF)).strftime('%Y-%m-%d %H:%M:%S') # Get OG (IDF) file's date modified.
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

        if not path_TIF:
            path_TIF = os.path.splitext(path_IDF)[0] + '.tif'

        with rasterio.open(path_TIF, "w", **meta) as Dst:
            Dst.write(A, 1)  # Write band 1

            Cvt_MtDt = {'COMMENT':( f"Converted from IDF on {Cvt_DT}."
                                    f"Original file created on {Ogn_DT}."
                                    f"Original IDF file location: {path_IDF}")}
                    
            if MtDt: # If project metadata exists, store it separately
                project_metadata = {f"USER_{k}": str(v) for k, v in MtDt.items()}
                Cvt_MtDt.update(project_metadata)

            Dst.update_tags(**Cvt_MtDt)
        print(f"\u2713 {path_TIF} has been saved (GeoTIFF) with conversion and project metadata.")
    except Exception as e:
        print(f"\u274C \n{e}")
    print(Sign)

    # def l_IDF_to_TIF(l_IDF, Dir_Out):
    #     """#666 under construction. The aim of this is to make a multi-band tif file instead of multiple single-band tif files, for each parameter."""
    #     DA = imod.formats.idf.open(l_IDF)#.sel(x=slice(Xmin, Xmax), y=slice(Ymax, Ymin)) # Read IDF files to am xarray.DataArray and slice it to model area (read from INI file)
    #     DA = DA.rio.write_crs(crs)  # Set Dutch RD New projection
    #     DA.rio.to_raster(Dir_Out)

def DA_to_TIF(DA, path_Out, d_MtDt, crs=crs, _print=False):
    """ Write a 2D xarray.DataArray (shape = [y, x]) to a single-band GeoTIFF.
    - DA: 2D xarray.DataArray with shape [y, x]
    - path_Out: Path to the output GeoTIFF file.
    - d_MtDt: Dictionary with metadata for this single band.
      Must contain exactly 1 item: {band_description: band_metadata_dict}
    - crs: Coordinate Reference System (optional)."""    
    
    if len(d_MtDt) != 1: # We expect exactly one band, so parse the single (key, value) from d_MtDt
        raise ValueError("DA_to_TIF expects exactly 1 item in d_MtDt for a 2D DataArray.")

    (band_key, band_meta) = list(d_MtDt.items())[0]

    transform = DA.rio.transform() # Build transform from DA

    with rasterio.open(path_Out,
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
        print(f"DA_to_TIF finished successfully for: {path_Out}")

def DA_to_MBTIF(DA, path_Out, d_MtDt, crs=crs, _print=False):
    """ Write a 3D xarray.DataArray (shape = [n_bands, y, x]) to a GeoTIFF. This bypasses rioxarray.to_raster() entirely, letting us set per-band descriptions and metadata in a single pass.
    - DA: 3D xarray.DataArray with shape [n_bands, y, x]
    - path_Out: Path to the output GeoTIFF file.
    - d_MtDt: Dictionary with metadata to be written to the GeoTIFF file. Each key is a band index (1-based) and each value is a dictionary of metadata tags.
    - crs: Coordinate Reference System (optional)."""

    band_keys, band_MtDt = zip(*d_MtDt.items())

    transform = DA.rio.transform()

    with rasterio.open(path_Out, #666 add ask-to-overwrite function (preferably to any function/command in this Lib that writes a file.)
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
        print(f"DA_to_MBTIF finished successfully for: {path_Out}")

# HD_IDF speciic PoP (could be extended/generalized at a later stage) --------------------------------------------------------------
def HD_IDF_Mo_Avg_to_MBTIF(MdlN: str, N_cores=None, crs=crs):
    """Reads Out IDF files from the model directory and calculates Mo Avg for each L. Saves them as MultiBand TIF files - each band representing the Mo Avg HD for each L."""

    d_paths = Utl.get_MdlN_paths(MdlN)
    path_PoP, path_MdlN = [ d_paths[v] for v in ['path_PoP', 'path_MdlN'] ]
    path_HD = os.path.join(path_MdlN, 'GWF_1/MODELOUTPUT/HEAD/HEAD')

    DF = Utl.HD_Out_IDF_to_DF(path_HD) # Read the IDF files to a DataFrame
    DF_grouped = DF.groupby(['year', 'month'])['path']

    path_Mo_AVG_Fo = os.path.join(path_PoP, f'Out/{MdlN}/HD_Mo_AVG') # path where Mo AVG files are stored
    os.makedirs(path_Mo_AVG_Fo, exist_ok=True) # Create the directory if it doesn't exist

    if N_cores is None:
        N_cores = max(os.cpu_count() - 2, 1) # Leave 2 cores free for other tasks. If there aren't enough cores available, set to 1.

    start = DT.now() # Start time
    with PPE(max_workers=N_cores) as E:
        futures = [E.submit(_HD_IDF_Mo_Avg_to_MBTIF_process_Mo, year, month, list(paths), MdlN, path_Mo_AVG_Fo, path_HD, crs)
                   for (year, month), paths in DF_grouped]
        for f in futures:
            print('\t', f.result(), '- Elapsed time (from start):', DT.now() - start)

    print('*** {MdlN} *** - Total elapsed:', DT.now() - start)

def _HD_IDF_Mo_Avg_to_MBTIF_process_Mo(year, month, paths, MdlN, path_Mo_AVG_Fo, path_HD, crs):
    """ """
    XA = imod.formats.idf.open(list(paths)) # Read the files to an Xarray
    XA_mean = XA.mean(dim='time') # Calculate monthly mean

    path_Out = os.path.join(path_Mo_AVG_Fo, f'HD_AVG_{year}_{month:02d}_{MdlN}.tif') # Create the output path

    d_MtDt = {str(L): {'layer': L} for L in XA.coords['layer'].values}
    d_MtDt['all'] = {'parameters': XA.coords,
                     'Description': f'Monthly mean GW heads per layer for {year}-{month:02d}, produced by aggregating {MdlN} output IDF files. Output IDF files are in {path_HD}.'}
    DA_to_MBTIF(XA_mean, path_Out, d_MtDt, crs=crs, _print=False)
    return f"*** {MdlN} *** - {year}-{month:02d} ✔ "
#       ----------------------------------------------------------------------------------------------------------------------------

def HD_IDF_GXG_to_MBTIF(MdlN: str, N_cores=None, crs=crs):

    d_paths = Utl.get_MdlN_paths(MdlN)
    path_PoP, path_MdlN = [ d_paths[v] for v in ['path_PoP', 'path_MdlN'] ]
    path_HD = os.path.join(path_MdlN, 'GWF_1/MODELOUTPUT/HEAD/HEAD')

    # Get list of layers in the model
    l_L = sorted({int(match.group(1)) for f in Path(path_HD).glob("HEAD_*.IDF")
                if (match := re.compile(r"_L(\d+)\.IDF$").search(f.name))})
    
    # Make a dictionary of the IDF files for each layer
    d_IDF_GXG = {i: sorted(f for f in Path(path_HD).glob(f"HEAD_*_L{i}.IDF")
                        if re.search(r'HEAD_(\d{4})(\d{2})(\d{2})', f.name)
                        and int((m := re.search(r'HEAD_(\d{4})(\d{2})(\d{2})', f.name)).group(3)) in {14, 28})
                for i in l_L}

    if N_cores is None:
        N_cores = max(os.cpu_count() - 2, 1)
    start = DT.now() # Start time

    start = DT.now()
    with PPE(max_workers=N_cores) as E:
        futures = [E.submit(_HD_IDF_GXG_to_MBTIF_process_L, L, d_IDF_GXG, MdlN, path_PoP, path_HD, crs)
                   for L in d_IDF_GXG.keys()]
        for f in futures:
            print('\t', f.result(), '- Elapsed time (from start):', DT.now() - start)

    print('Total elapsed:', DT.now() - start)

def _HD_IDF_GXG_to_MBTIF_process_L(L, d_IDF_GXG, MdlN, path_PoP, path_HD, crs):
    XA = imod.idf.open(d_IDF_GXG[L])
    GXG = imod.evaluate.calculate_gxg(XA.squeeze())
    GXG = GXG.rename_vars({var: var.upper() for var in GXG.data_vars})
    GXG = GXG.rename_vars({'N_YEARS_GXG': 'N_years_GXG', 'N_YEARS_GVG': 'N_years_GVG'})
    GXG["GHG_m_GLG"] = GXG["GHG"] - GXG["GLG"]
    GXG = GXG[["GHG", "GLG", "GHG_m_GLG", "GVG", "N_years_GXG", "N_years_GVG"]]

    path_Out = os.path.join(path_PoP, 'Out', MdlN, 'GXG', f'GXG_L{L}_{MdlN}.tif')
    os.makedirs(os.path.dirname(path_Out), exist_ok=True)

    d_MtDt = {str(i): {f'{var}_AVG': float(GXG[var].mean().values) for var in GXG.data_vars} 
              for i in range(len(GXG.data_vars))}
    
    d_MtDt['all'] = {'parameters': XA.coords,
                     'Description': f'{MdlN} GXG (path: {path_HD})\nFor more info see: https://deltares.github.io/imod-python/api/generated/evaluate/imod.evaluate.calculate_gxg.html'}

    DA = GXG.to_array(dim="band")
    DA_to_MBTIF(DA, path_Out, d_MtDt, crs=crs, _print=False)
    return f"GXG_L{L} ✔"

# ----------------------------------------------------------------------------------------------------------------------------------

# OUTDATED -------------------------------------------------------------------------------------------------------------------------
def A_to_Raster_n_IDF(A, IDF_MtDt, path_Out, field='HD_L1', crs="EPSG:4326"):
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

    tif_path = path_Out + ".tif"
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
    idf_path = path_Out + ".idf"
    imod.idf.write(idf_path, DA)
    print(f"{idf_path} has been saved (iMOD IDF).")
# ----------------------------------------------------------------------------------------------------------------------------------