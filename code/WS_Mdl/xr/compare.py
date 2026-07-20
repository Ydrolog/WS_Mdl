from pathlib import Path

import numpy as np
import rasterio
import xarray as xra
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_bounds

from WS_Mdl.core.style import set_verbose, sprint
from WS_Mdl.core.text import replace_MdlN
from WS_Mdl.xr.convert import to_MBTIF

__all__ = ['Diff_TIF', 'Diff_MBTIF', 'Diff_PoP_Par']


def Diff_TIF(Pa_TIF1, Pa_TIF2, Pa_TIF_Out=None, band_name='Diff', resampling='nearest', verbose=True):
    """Calculate ``TIF1 - TIF2`` for two single-band GeoTIFFs.

    The output covers the union of both extents and is aligned to the grid of
    the raster with the larger spatial extent. The other raster is reprojected
    onto that grid. Cells not covered by both inputs are written as NoData.

    Parameters
    ----------
    resampling : str
        A name from :class:`rasterio.enums.Resampling`. ``'nearest'`` avoids
        inventing intermediate values; ``'bilinear'`` can be useful for
        continuous surfaces such as groundwater heads.
    """
    Pa_TIF1, Pa_TIF2 = Path(Pa_TIF1), Path(Pa_TIF2)
    Pa_TIF_Out = Path(Pa_TIF_Out) if Pa_TIF_Out is not None else Pa_TIF1.with_name(f'{Pa_TIF1.stem}_diff.tif')

    try:
        resampling_method = Resampling[resampling]
    except KeyError as e:
        choices = ', '.join(Resampling.__members__)
        raise ValueError(f'Unknown resampling method {resampling!r}. Choose from: {choices}') from e

    if verbose:
        sprint(f'Calculating difference: {Pa_TIF1.name} - {Pa_TIF2.name}')

    with rasterio.open(Pa_TIF1) as src1, rasterio.open(Pa_TIF2) as src2:
        if src1.count != 1 or src2.count != 1:
            raise ValueError(f'Diff_TIF requires single-band inputs; got {src1.count} and {src2.count} bands.')
        if src1.crs is None or src2.crs is None:
            raise ValueError('Both input TIFFs must have a CRS.')

        bounds2 = transform_bounds(src2.crs, src1.crs, *src2.bounds, densify_pts=21)
        area1 = (src1.bounds.right - src1.bounds.left) * (src1.bounds.top - src1.bounds.bottom)
        area2 = (bounds2[2] - bounds2[0]) * (bounds2[3] - bounds2[1])
        template = src1 if area1 >= area2 else src2
        target_crs = template.crs

        bounds1 = transform_bounds(src1.crs, target_crs, *src1.bounds, densify_pts=21)
        bounds2 = transform_bounds(src2.crs, target_crs, *src2.bounds, densify_pts=21)
        left = min(bounds1[0], bounds2[0])
        bottom = min(bounds1[1], bounds2[1])
        right = max(bounds1[2], bounds2[2])
        top = max(bounds1[3], bounds2[3])

        xres, yres = abs(template.transform.a), abs(template.transform.e)
        anchor_x, anchor_y = template.transform.c, template.transform.f
        left = anchor_x + np.floor((left - anchor_x) / xres) * xres
        right = anchor_x + np.ceil((right - anchor_x) / xres) * xres
        top = anchor_y + np.ceil((top - anchor_y) / yres) * yres
        bottom = anchor_y + np.floor((bottom - anchor_y) / yres) * yres
        width = int(round((right - left) / xres))
        height = int(round((top - bottom) / yres))
        transform = from_origin(left, top, xres, yres)

        aligned = []
        for src in (src1, src2):
            data = np.full((height, width), np.nan, dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=data,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=transform,
                dst_crs=target_crs,
                dst_nodata=np.nan,
                resampling=resampling_method,
            )
            aligned.append(data)

        diff = aligned[0] - aligned[1]
        profile = template.profile.copy()
        profile.update(
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype='float32',
            crs=target_crs,
            transform=transform,
            nodata=np.nan,
        )
        Pa_TIF_Out.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(Pa_TIF_Out, 'w', **profile) as dst:
            dst.write(diff, 1)
            dst.set_band_description(1, band_name)
            dst.update_tags(1, description=band_name)

    if verbose:
        sprint(f'🟢 - Difference saved to {Pa_TIF_Out}')

    return Pa_TIF_Out


def Diff_MBTIF(Pa_TIF1, Pa_TIF2, band_name=None, Pa_TIF_Out=None, verbose=True):
    """
    Calculates the difference between two Multi-band TIF files (TIF1 - TIF2).
    Aligns both files by their labelled coordinates before subtraction.
    """

    Pa_TIF1, Pa_TIF2 = Path(Pa_TIF1), Path(Pa_TIF2)

    if Pa_TIF_Out is None:
        Pa_TIF_Out = Pa_TIF1.parent / f'{Pa_TIF1.stem}_diff.tif'

    if verbose:
        sprint(f'Calculating difference: {Pa_TIF1.name} - {Pa_TIF2.name}')

    ds1 = xra.open_dataset(Pa_TIF1, engine='rasterio')
    ds2 = xra.open_dataset(Pa_TIF2, engine='rasterio')
    try:
        da1 = ds1['band_data']
        da2 = ds2['band_data']

        if set(da1.dims) != set(da2.dims):
            raise ValueError(f'Dimensions do not match: {da1.dims} vs {da2.dims}')

        da1, da2 = xra.align(da1, da2, join='inner')
        empty_dims = [dim for dim, size in da1.sizes.items() if size == 0]
        if empty_dims:
            raise ValueError(f'No overlapping coordinates for dimension(s): {empty_dims}')

        da_diff = da1 - da2

        band_name = band_name if band_name is not None else 'Diff'
        d_MtDt = {
            f'{band_name}_Band_{i + 1}': {'description': f'{band_name} Band {i + 1}'} for i in range(da_diff.shape[0])
        }

        set_verbose(False)
        try:
            to_MBTIF(da_diff, Pa_TIF_Out, d_MtDt, _print=verbose)
        finally:
            set_verbose(True)

        if verbose:
            sprint(f'🟢 - Difference saved to {Pa_TIF_Out}')
    finally:
        ds1.close()
        ds2.close()


def Diff_PoP_Par(
    MdlN,
    B: str = None,
    Param: str = 'all',
):  # Improve to be able to do all parameters in PoP/Out folder.
    """
    Calcs Diff between TIF files of two MdLNs stored in M.Pa.PoP_Out_MdlN/Param folder. Saves the Diff as a TIF in the same dir as the MdlN PoP_Out files.
    Warning!: It doesn't check if the dates of the PoP_Out of the two MdLNs are the same.
    """
    from WS_Mdl.core.mdl import Mdl_N

    M = Mdl_N(MdlN)

    B = B if B is not None else M.B

    Pa_PoP = M.Pa.PoP_Out_MdlN / Param
    l_Fi = [i for i in Pa_PoP.rglob('*.tif') if MdlN in i.name]
    # print(*l_GXG_Fi, sep='\n')

    if not l_Fi:
        raise FileNotFoundError(f"🔴 - No .tif files found for '{MdlN}' in: {Pa_PoP}")

    for Fi in l_Fi:
        Fi_2 = Pa_PoP / replace_MdlN(Fi, MdlN, B)
        Pa_Out = Fi.parent / replace_MdlN(Fi.name, MdlN, f'{MdlN}m{"".join(i for i in B if i.isdigit())}')
        # print(Pa_TIF_1, Pa_TIF_2, Pa_Out, end='\n---------------\n', sep='\n')
        try:
            try:
                Diff_TIF(Fi, Fi_2, band_name=f'{MdlN}m{B}', Pa_TIF_Out=Pa_Out)
            except Exception as e:
                print(type(e), e)
                Diff_MBTIF(Fi, Fi_2, band_name=f'{MdlN}m{B}', Pa_TIF_Out=Pa_Out)
        except (FileNotFoundError, OSError) as e:
            print(f'🔴 - Missing/unreadable file(s) for:\n{Fi}:\n {e}', end='------------\n')
