from pathlib import Path

import numpy as np
import rasterio

from WS_Mdl.core.defaults import CRS
from WS_Mdl.core.style import sprint


def to_TIF(DA, Pa_Out, d_MtDt, CRS=CRS, _print=False):
    """Write a 2D xarray.DataArray (shape = [y, x]) to a single-band GeoTIFF.
    - DA: 2D xarray.DataArray with shape [y, x]
    - Pa_Out: Path to the output GeoTIFF file.
    - d_MtDt: Dictionary with metadata for this single band.
      Must contain exactly 1 item: {band_description: band_metadata_dict}
    - CRS: Coordinate Reference System (optional)."""

    Pa_Out = Path(Pa_Out)  # Ensure Pa_Out is a Path object for consistent handling

    if len(d_MtDt) != 1:  # We expect exactly one band, so parse the single (key, value) from d_MtDt
        raise ValueError('DA_to_TIF expects exactly 1 item in d_MtDt for a 2D DataArray.')

    (band_key, band_meta) = list(d_MtDt.items())[0]

    transform = DA.rio.transform()  # Build transform from DA

    Pa_Out.parent.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

    with rasterio.open(
        Pa_Out,
        'w',
        driver='GTiff',
        height=DA.shape[0],
        width=DA.shape[1],
        count=1,  # single band
        dtype=str(DA.dtype),
        CRS=CRS,
        transform=transform,
    ) as Dst:
        Dst.write(DA.values, 1)  # Write the 2D data as band 1
        Dst.set_band_description(1, band_key)  # Give the band a useful name
        Dst.update_tags(1, **band_meta)  # Write each row field as a separate metadata tag on this band
    if _print:
        sprint(f'🟢 - DA_to_TIF finished successfully for: {Pa_Out}')


def to_MBTIF(DA, Pa_Out, d_MtDt, CRS=CRS, _print=False, decimals=3):
    """Write a 3D xarray.DataArray (shape = [n_bands, y, x]) to a GeoTIFF. This bypasses rioxarray.to_raster() entirely, letting us set per-band descriptions and metadata in a single pass.
    - DA: 3D xarray.DataArray with shape [n_bands, y, x]
    - Pa_Out: Path to the output GeoTIFF file.
    - d_MtDt: Dictionary with metadata to be written to the GeoTIFF file. Each key is a band index (1-based) and each value is a dictionary of metadata tags.
    - CRS: Coordinate Reference System (optional).
    - decimals: Number of decimal places to round array values to (default: 3)."""

    # Separate band metadata from global metadata
    band_items = [(k, v) for k, v in d_MtDt.items() if k != 'all']
    band_keys, band_MtDt = zip(*band_items) if band_items else ([], [])

    transform = DA.rio.transform()

    # Ensure we have the right number of bands
    n_bands = DA.shape[0]

    with rasterio.open(
        Pa_Out,  # 666 add ask-to-overwrite function (preferably to any function/command in this Lib that writes a file.)
        'w',
        driver='GTiff',
        height=DA.shape[1],
        width=DA.shape[2],
        count=n_bands,
        dtype=str(DA.dtype),
        CRS=CRS,
        transform=transform,
        photometric='MINISBLACK',
    ) as Dst:
        for i in range(n_bands):  # Write each band.
            # Round the values before writing
            band_values = np.round(DA[i].values, decimals=decimals)
            Dst.write(band_values, i + 1)  # Write the actual pixels for this band (i+1 is the band index in Rasterio)
            if band_keys and i < len(band_keys):
                Dst.set_band_description(
                    i + 1, band_keys[i]
                )  # Set a band description that QGIS will show as "Band 01: <description>"
                Dst.update_tags(i + 1, **band_MtDt[i])  # Write each row field as a separate metadata tag on this band

        if 'all' in d_MtDt:  # If "all" exists, write dataset-wide metadata (NOT tied to a band)
            Dst.update_tags(**d_MtDt['all'])  # Set global metadata for the whole dataset

    if _print:
        sprint(f'DA_to_MBTIF finished successfully for: {Pa_Out}')
