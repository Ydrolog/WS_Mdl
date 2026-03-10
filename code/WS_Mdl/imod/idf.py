import glob
import os
import re
from datetime import datetime as DT
from pathlib import Path
from typing import Dict, Optional

import imod
import pandas as pd
import rasterio
import xarray as xra
from rasterio.transform import from_bounds
from tqdm import tqdm
from WS_Mdl.core.defaults import CRS
from WS_Mdl.core.style import Sep, sprint
from WS_Mdl.xr import convert as xr_convert


def HD_Out_to_DF(
    path, add_extra_cols: bool = True
):  # 666 can make it save DF (e.g. to CSV) if a 2nd path is provided. Unecessary for now.
    """
    Reads all .IDF files in `path` into a DataFrame with columns:
      - path, file, type, year, month, day, L
    If add_extra_cols=True, also adds:
      - season (Winter/Spring/Summer/Autumn)
      - season_year (roll Winter Dec→Feb into next calendar year)
      - quarter (Q1-Q4)
      - Hy_year (hydrological year: Oct-Sep → Oct-Dec roll into next year)

      Parameters are extracted from filnames, based on a standard format. Hence, don't use this for other groups of IDF files, unless you're sure they follow the same format."""  # 666 can be generalized later, to work on all sorts of IDF files.

    path = Path(path)
    Se_Fi_path = pd.Series([path / i for i in path.iterdir() if i.is_file() and i.suffix.lower() == '.idf'])
    DF = pd.DataFrame({'path': Se_Fi_path, 'file': Se_Fi_path.apply(lambda x: x.name)})
    DF[['type', 'year', 'month', 'day', 'L']] = (
        DF['file']
        .str.extract(r'^(?P<type>[A-Z]+)_(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})\d{6}_L(?P<L>\d+)\.IDF$')
        .astype({'year': int, 'month': int, 'day': int, 'L': int})
    )

    if add_extra_cols:
        # 1) season & season_year
        month2season = {
            12: 'Winter',
            1: 'Winter',
            2: 'Winter',
            3: 'Spring',
            4: 'Spring',
            5: 'Spring',
            6: 'Summer',
            7: 'Summer',
            8: 'Summer',
            9: 'Autumn',
            10: 'Autumn',
            11: 'Autumn',
        }

        DF['season'] = DF['month'].map(month2season)
        DF['season_year'] = DF.apply(
            lambda r: r.year + 1 if r.month == 12 else r.year, axis=1
        )  # roll December into next year's winter

        # 2) quarter (calendar)
        DF['quarter'] = DF['month'].apply(lambda m: f'Q{((m - 1) // 3) + 1}')

        # 3) GHG “water” year (Apr–Mar) months 4–12 → water_year = year+1; months 1–3 → water_year = year
        DF['GW_year'] = DF.apply(lambda r: r.year if r.month >= 4 else r.year - 1, axis=1)

    # DF.to_csv(PJ(path, 'contents.csv'), index=False)

    return DF


def stack_to_DF(S_Pa_IDF):
    """Reads all .IDF Fis listed in a S_Fi_IDF into DF['IDF']. Returns the DF containing Fi_names and the IDF contents.
    Pa_Fo is the path of the Fo where th files are stored in."""

    DF = pd.DataFrame({'path': S_Pa_IDF, 'IDF': None})

    for i, p in tqdm(DF['path'].items(), desc='Loading .IDF files', total=len(DF['path'])):
        if p.endswith('.IDF'):  # Ensure only .IDF files are processed
            try:  # Read the .IDF file into an xA DataA
                DF.at[i, 'IDF'] = imod.idf.read(p)
            except Exception as e:
                print(f'Error reading {p}: {e}')
    return DF


def to_TIF(Pa_IDF: str, Pa_TIF: Optional[str] = None, MtDt: Optional[Dict] = None, CRS=CRS):
    """Converts IDF file to TIF file.
    If Pa_TIF is not provided, it'll be the same as Pa_IDF, except for the file type ending.
    CRS (coordinate reference system) is set to the Amerfoot CRS by default, but can be changed for other projects."""
    sprint(Sep)
    try:
        A, MtDt = imod.idf.read(Pa_IDF)

        Ogn_DT = DT.fromtimestamp(os.path.getctime(Pa_IDF)).strftime(
            '%Y-%m-%d %H:%M:%S'
        )  # Get OG (IDF) file's date modified.
        Cvt_DT = DT.now().strftime('%Y-%m-%d %H:%M:%S')  # Get current time, to write time of convertion to comment

        N_R, N_C = A.shape

        transform = from_bounds(
            west=MtDt['xmin'],
            south=MtDt['ymin'],
            east=MtDt['xmax'],
            north=MtDt['ymax'],
            width=N_C,
            height=N_R,
        )
        meta = {
            'driver': 'GTiff',
            'height': N_R,
            'width': N_C,
            'count': 1,
            'dtype': str(A.dtype),
            'CRS': CRS,
            'transform': transform,
        }

        if not Pa_TIF:
            Pa_TIF = os.path.splitext(Pa_IDF)[0] + '.tif'

        with rasterio.open(Pa_TIF, 'w', **meta) as Dst:
            Dst.write(A, 1)  # Write band 1

            Cvt_MtDt = {
                'COMMENT': (
                    f'Converted from IDF on {Cvt_DT}.'
                    f'Original file created on {Ogn_DT}.'
                    f'Original IDF file location: {Pa_IDF}'
                )
            }

            if MtDt:  # If project metadata exists, store it separately
                project_metadata = {f'USER_{k}': str(v) for k, v in MtDt.items()}
                Cvt_MtDt.update(project_metadata)

            Dst.update_tags(**Cvt_MtDt)
        sprint(f'🟢 {Pa_TIF} has been saved (GeoTIFF) with conversion and project metadata.')
    except Exception as e:
        sprint(f'🔴 \n{e}')
    sprint(Sep)


def to_MBTIF(l_IDF, Pa_TIF: Optional[str] = None, MtDt: Optional[Dict] = None, CRS=CRS):
    """
    Converts multiple IDF files to a single multi-band TIF file with proper layer ordering.

    Parameters:
    - l_IDF: List of IDF file paths OR glob pattern string (e.g., "HEAD_19930101_L*_NBr1.IDF")
    - Pa_TIF: Output TIF file path (optional, defaults to first IDF name with .tif extension)
    - MtDt: Additional metadata dictionary (optional)
    - CRS: Coordinate reference system (default: 'EPSG:28992')
    """

    sprint(Sep)
    try:
        # Handle glob pattern or list of files
        if isinstance(l_IDF, str):
            idf_files = sorted(glob.glob(l_IDF))
            if not idf_files:
                raise ValueError(f'No files found matching pattern: {l_IDF}')
            sprint(f'Found {len(idf_files)} files matching pattern: {l_IDF}')
        else:
            idf_files = l_IDF

        # Sort files by layer number if they contain layer info (L#)
        def extract_layer_num(filepath):
            filename = os.path.basename(filepath)
            match = re.search(r'_L(\d+)_', filename)
            return int(match.group(1)) if match else 0

        idf_files = sorted(idf_files, key=extract_layer_num)  # sort by L number

        # Generate output path if not provided
        if Pa_TIF is None:
            Pa_TIF = os.path.splitext(idf_files[0])[0] + '_multiband.tif'

        # Load each IDF individually and stack with proper layer coordinates
        da_list = []
        d_MtDt = {}

        for idf_file in idf_files:
            # Load single IDF
            da_single = imod.formats.idf.open([idf_file])
            if hasattr(da_single, 'data_vars') and len(da_single.data_vars) > 0:
                da_single = da_single[list(da_single.data_vars)[0]]

            # Remove singleton non-spatial dimensions
            for dim in list(da_single.dims):
                if dim not in ['x', 'y'] and da_single.sizes[dim] == 1:
                    da_single = da_single.squeeze(dim)

            # Create layer name and add metadata
            filename = os.path.splitext(os.path.basename(idf_file))[0]
            da_single = da_single.expand_dims('layer').assign_coords(layer=[filename])

            # Build metadata for this band
            try:
                _, idf_metadata = imod.idf.read(idf_file)
                band_metadata = {
                    'origin_path': idf_file,
                    'creation_time': DT.fromtimestamp(os.path.getctime(idf_file)).strftime('%Y-%m-%d %H:%M:%S'),
                    'conversion_time': DT.now().strftime('%Y-%m-%d %H:%M:%S'),
                }
                if idf_metadata:
                    for k, v in idf_metadata.items():
                        band_metadata[f'idf_{k}'] = str(v)
                d_MtDt[filename] = band_metadata
            except Exception as e:
                d_MtDt[filename] = {
                    'origin_path': idf_file,
                    'conversion_time': DT.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'error': f'Could not read IDF metadata: {str(e)}',
                }

            da_list.append(da_single)

        # Concatenate all layers and set CRS
        DA = xra.concat(da_list, dim='layer').rio.write_CRS(CRS)

        # Add global metadata if provided
        if MtDt:
            d_MtDt['all'] = MtDt

        # Write multi-band TIF
        xr_convert.to_MBTIF(DA, Pa_TIF, d_MtDt, CRS=CRS, _print=True)
        sprint(f'🟢 {Pa_TIF} has been saved as multi-band GeoTIFF with {len(idf_files)} bands.')

    except Exception as e:
        sprint(f'🔴 Error in IDFs_to_MBTIF: {e}')

    sprint(Sep)
