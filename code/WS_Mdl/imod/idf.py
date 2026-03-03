from pathlib import Path

import pandas as pd


def HD_Out_IDF_to_DF(
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


def IDFs_to_DF(S_Pa_IDF):
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
