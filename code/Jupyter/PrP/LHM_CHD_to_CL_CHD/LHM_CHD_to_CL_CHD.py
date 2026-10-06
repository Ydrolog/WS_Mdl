"""
Climatology of constant-head (CHD) boundary conditions (iMOD IDF files).
 
PURPOSE
    For future-climate runs we assume that the CHD heads stay the same as in the past. To make every
    simulated year identical (and to keep the seasonal cycle) we average the CHD heads per time step of the year.
 
METHOD
    1. Read all files named  LHM_HD_<YYYYMMDD>_L<layer>_<in-sim>.idf  from one folder. The CHD time steps are
       the 14th and 28th of every month, so a year has 24 time steps.
    2. For every (day, month, layer) take the mean head over the selected years (default 1991-2025).
       Example: all 14-Jan files of L1 (1991, 1992, ..., 2025) -> one mean 14-Jan file for L1.
       NaN cells (no CHD / inactive cells) are ignored in the mean; cells that are NaN in every year stay NaN.
    3. Write the means as  CL_HD_<DDMM>_L<layer>.idf   (e.g. CL_HD_1401_L1.idf)
       => 24 time steps x number of layers files. DDMM has no year, so a run can use the same file in every year.
       The output has no simulation number: put it in its own folder, e.g. In/CHD/CL/ (OUTPUT_DIR).
    4. The input only exists for the odd layers. The even layers are written as a copy of the odd layer above
       (L2 <- L1, L4 <- L3, ...). This is needed for the SHD (starting heads) script, which uses all layers.
    5. A _metadata.txt is written next to the output files (source simulation, years, date, warnings).
 
USAGE
    python make_chd_climatology.py INPUT_DIR OUTPUT_DIR
    python make_chd_climatology.py INPUT_DIR OUTPUT_DIR --start-year 1991 --end-year 2025 \
           --days 14 28 --in-sim NBr111
"""
 
# %% Imports
import argparse
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
 
import imod
import pandas as pd
import xarray as xr
 
# %% Constants
# File name of an input CHD file, e.g. LHM_HD_19910114_L1_NBr111.idf
#   groups: 1 = year, 2 = month, 3 = day, 4 = layer, 5 = simulation number
NAME_RE = re.compile(r'^LHM_HD_(\d{4})(\d{2})(\d{2})_L(\d+)_(.+)\.idf$', re.IGNORECASE)
 
 
# %% Functions
def index_files(input_dir, start_year, end_year, days, in_sim):
    """Find the input files and index them as {(day, month, layer): {year: path}}.
 
    Only files that are needed are kept: the selected years, the selected days of the month, odd layers
    and the requested simulation number. Even layers are skipped because they are derived from the odd layers.
    """
    groups = defaultdict(dict)
    for p in sorted(input_dir.glob('*.idf')):
        m = NAME_RE.match(p.name)
        if not m:
            continue  # not a CHD file 
        yr, mo, dy, lay, sim = int(m[1]), int(m[2]), int(m[3]), int(m[4]), m[5]
        if sim != in_sim:
            continue  # other simulation
        if dy not in days or not (start_year <= yr <= end_year) or lay % 2 == 0:
            continue
        groups[(dy, mo, lay)][yr] = p
 
    if not groups:
        sys.exit('No matching files found (check folder, years and --in-sim).')
    return groups
 
 
def mean_over_years(paths, in_sim):
    """Open the files of ONE (day, month, layer) with imod and return their mean over time.
 
    imod.idf.open reads the IDF files lazily and stacks them in one DataArray with dims (time, layer, y, x).
    The time comes from {time} in the file name (YYYYMMDD) and nodata (1e20) is converted to NaN.
    Returns the mean DataArray (layer, y, x) and the number of cells that are valid in some, but not all, years.
    """
    DA = imod.idf.open(paths, pattern=f'{{name}}_{{time}}_L{{layer}}_{in_sim}')
    n_valid = DA.notnull().sum('time')  # number of years with a value, per cell
    n_partial = int(((n_valid > 0) & (n_valid < DA.sizes['time'])).sum())
    return DA.mean('time', skipna=True).compute(), n_partial
 
 
def copy_to_even_layers(DA, n_layers):
    """Add the even layers: L2 <- L1, L4 <- L3, ... (only up to n_layers)."""
    DA_even = DA.assign_coords(layer=DA.layer + 1)
    DA_even = DA_even.sel(layer=DA_even.layer <= n_layers)
    return xr.concat([DA, DA_even], dim='layer').sortby('layer')
 
 
# %% Main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input_dir', type=Path)
    ap.add_argument('output_dir', type=Path)
    ap.add_argument('--start-year', type=int, default=1991)
    ap.add_argument('--end-year', type=int, default=2020)
    ap.add_argument('--days', type=int, nargs='+', default=[14, 28])
    ap.add_argument('--in-sim', default='NBr111', help='simulation number in the input file names (default: NBr111)')
    ap.add_argument('--prefix', default='CL_HD')
    ap.add_argument('--n-layers', type=int, default=37, help='total number of model layers (default: 37)')
    ap.add_argument('--no-fill-even', action='store_true', help='do NOT create even layers (L2 <- L1, L4 <- L3, ...)')
    args = ap.parse_args()
 
    args.output_dir.mkdir(parents=True, exist_ok=True)
 
    # 1. Index the input files: (day, month, layer) -> {year: path}
    groups = index_files(args.input_dir, args.start_year, args.end_year, args.days, args.in_sim)
    years_expected = set(range(args.start_year, args.end_year + 1))
    slots = sorted({(mo, dy) for dy, mo, _ in groups})  # time steps of the year, in calendar order
    layers = sorted({lay for _, _, lay in groups})
    print(f'{len(slots)} time steps x {len(layers)} odd layers, years {args.start_year}-{args.end_year}, sim {args.in_sim}')
 
    # 2. Per time step of the year: mean over the years for every (odd) layer, then write the IDF files
    problems = []
    n_written = 0
    for mo, dy in slots:
        l_mean = []
        for lay in layers:
            files = groups.get((dy, mo, lay), {})
            if not files:
                problems.append(f'{dy:02d}-{mo:02d} L{lay}: no files at all, layer skipped')
                continue
 
            missing = sorted(years_expected - set(files))
            if missing:
                problems.append(f'{dy:02d}-{mo:02d} L{lay}: missing years {missing}')
 
            DA_mean, n_partial = mean_over_years(list(files.values()), args.in_sim)
            if n_partial:
                problems.append(f'{dy:02d}-{mo:02d} L{lay}: {n_partial} cells not valid in all years')
            l_mean.append(DA_mean)
 
        if not l_mean:
            continue
        DA = xr.concat(l_mean, dim='layer')  # (layer, y, x) with the odd layers
 
        # Even layers have no data of their own: copy the odd layer above
        if not args.no_fill_even:
            DA = copy_to_even_layers(DA, args.n_layers)
 
        # imod.idf.save needs a time coordinate to build the file name. The year is arbitrary: only {time:%d%m} is used.
        DA = DA.expand_dims(time=[pd.Timestamp(2001, mo, dy)])
        DA.name = args.prefix
        imod.idf.save(
            args.output_dir / args.prefix,
            DA,
            pattern=f'{{name}}_{{time:%d%m}}_L{{layer}}.idf',
        )
        n_written += DA.sizes['layer']
        print(f'  {dy:02d}-{mo:02d}: {DA.sizes["layer"]} layers written')
 
    print(f'\nDone: {n_written} files written to {args.output_dir}')
 
    # 3. Document where these files come from, in the same folder (the file names themselves contain no sim number or period)
    lines = [
        f'Climatology of CHD heads, made on {date.today():%Y-%m-%d} by {Path(__file__).name}.',
        f'Source: {args.input_dir} (files LHM_HD_<YYYYMMDD>_L<layer>_{args.in_sim}.idf).',
        f'Mean over the years {args.start_year}-{args.end_year}, per time step of the year (days {args.days} of every month).',
        f'Files: {args.prefix}_<DDMM>_L<layer>.idf, {n_written} files, layers up to L{args.n_layers}.',
        'Even layers: ' + ('copy of the odd layer above (L2 <- L1, L4 <- L3, ...).' if not args.no_fill_even else 'not created.'),
        f'Warnings while making them: {len(problems)}' + (' (see the console output of the run).' if problems else '.'),
    ]
    if problems:
        lines += ['', 'WARNINGS:'] + [f'  - {s}' for s in problems]
    (args.output_dir / '_metadata.txt').write_text('\n'.join(lines) + '\n')
    if problems:
        print('\nWARNINGS:')
        for s in problems:
            print('  -', s)
 
 
if __name__ == '__main__':
    main()