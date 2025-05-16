# ***** Geospatial Functions *****
import os
from .utils import Sign, Pre_Sign
import rasterio
from rasterio.transform import from_bounds
import imod
from datetime import datetime as DT
from typing import Optional, Dict
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
# ----------------------------------------------------------------------------------------------------------------------------------