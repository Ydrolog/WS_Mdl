import struct
from pathlib import Path

import numpy as np
import pandas as pd
from WS_Mdl.core.style import Sep, sprint


def Bin_to_text(bin_path):
    """
    Converts a binary file to text.
    Assumes hardcoded record structure (currently RIV: L, R, C, Stage, Cond, Rbot).
    Attempts to detect if it is a flat C-binary or Fortran Unformatted binary.
    """
    bin_path = Path(bin_path)

    if not bin_path.exists():
        print(f'🔴 - File not found: {bin_path}')
        return

    def _Bin_to_text_helper(df, bin_path):
        out_path = bin_path.with_suffix('.txt')
        try:
            with open(out_path, 'w') as f:
                # Write header
                f.write(f'# {" ".join(df.columns)}\n')
                # Convert to structured array for savetxt
                rec_array = df.to_records(index=False)
                # Use savetxt for speed and formatting
                np.savetxt(f, rec_array, fmt='%d %d %d %.6f %.6f %.6f')

            print(f'   🟢 - Saved text file to: {out_path}')
            print('   First 5 lines:')
            print(df.head())
        except Exception as e:
            print(f'Failed to save text file: {e}')

    file_size = bin_path.stat().st_size
    print(f'--- Processing: {bin_path}')
    print(f'--- Size: {file_size} bytes')

    # Define the expected record structure for RIV package (or similar)
    # Layer(int), Row(int), Column(int), Stage(float), Cond(float), Rbot(float)
    # Ints are typically 32-bit (4 bytes), Floats 64-bit (8 bytes).
    # Total = 36 bytes.
    record_dtype = np.dtype(
        [('k', '<i4'), ('i', '<i4'), ('j', '<i4'), ('stage', '<f8'), ('cond', '<f8'), ('rbot', '<f8')]
    )
    record_size = 36

    # --- Attempt 1: Flat C-Binary (contiguous records) ---
    if file_size % record_size == 0 and file_size > 0:
        n_rec = file_size // record_size
        print(f'   Shape detection: Matches {n_rec} records of {record_size} bytes.')

        try:
            data = np.fromfile(bin_path, dtype=record_dtype)
            df = pd.DataFrame(data)

            # Sanity check: k, i, j should be valid
            # MF6 is 1-based.
            valid_idx = True
            if not ((df['k'] >= 0).all() and (df['i'] >= 0).all()):
                valid_idx = False

            if valid_idx:
                print('   🟢 - Data looks like valid indices.')
                _Bin_to_text_helper(df, bin_path)
                return
            else:
                print('   🟡 - Indices check failed (negative values found). Trying Fortran format...')

        except Exception as e:
            print(f'   Could not read as flat binary: {e}')

    # --- Attempt 2: Fortran Unformatted (Header + Data + Footer) ---
    # Structure: [Size (4b)] [Data] [Size (4b)]

    try:
        with open(bin_path, 'rb') as f:
            header_bytes = f.read(4)
            if len(header_bytes) == 4:
                header_val = struct.unpack('<i', header_bytes)[0]
                print(f'   First Fortran marker: {header_val} (Expected data size)')

                # Check if this marker matches the file size (Simple case: single record)
                # File should be: 4 + header_val + 4 = header_val + 8
                if header_val + 8 == file_size:
                    print('   🔴 - Identified as single-record Fortran Unformatted file.')

                    # Check if data size matches multiple of record_size
                    if header_val % record_size == 0:
                        n_rec = header_val // record_size
                        print(f'   Contains {n_rec} records.')

                        data_bytes = f.read(header_val)
                        footer_bytes = f.read(4)
                        footer_val = struct.unpack('<i', footer_bytes)[0]

                        if footer_val == header_val:
                            # Parse data
                            data = np.frombuffer(data_bytes, dtype=record_dtype)
                            df = pd.DataFrame(data)
                            _Bin_to_text_helper(df, bin_path)
                            return
                        else:
                            print('   🔴 - Footer marker mismatch.')
                    else:
                        print(f'   🔴 - Data size {header_val} is not a multiple of {record_size}.')
                else:
                    print('   Does not match single-record Fortran structure (Size + 8 != FileSize).')
    except Exception as e:
        print(f'   Error reading Fortran format: {e}')

    print('\n🔴🔴🔴 - Failed to automatically convert. The file format might differ from (L,R,C,Stage,Cond,Rbot).')


def Vtr_to_TIF(Pa_Vtr, Fld, Pa_TIF, MdlN=None, CRS=None, SigDig=4):
    """
    Converts a vector file to a single-band TIF file.
    - MdlN: Model number (e.g. 'NBr13').
    - Pa_Vtr: Path to the vector file.
    - Pa_TIF: Path to the output TIF file.
    - CRS: Coordinate Reference System for the output TIF.
    - SigDig: Number of significant digits to round to (default 4).
    """
    sprint(Sep)

    # importing here to keep Bin_to_text (only other function in this module ATM) lightweight.
    import geopandas as gpd
    import imod
    import xarray as xra
    from WS_Mdl.core.df import round_Cols
    from WS_Mdl.core.mdl import Mdl_N
    from WS_Mdl.imod.ini import Mdl_Dmns
    from WS_Mdl.xr.convert import to_TIF

    if CRS is None:
        from WS_Mdl.core.defaults import CRS

    if not MdlN:
        try:
            MdlN = Path(Pa_Vtr).stem.split('_')[-1]
        except Exception as e:
            raise ValueError(f'🔴 - Could not determine MdlN from Pa_Vtr ({e}). Provide MdlN explicitly.')
    sprint(f'*** {MdlN} *** - Vtr_to_TIF\n')

    # Load V file
    GDF = gpd.read_file(Pa_Vtr, columns=[Fld])
    sprint(f'🟢 - Loaded vector file from {Pa_Vtr.name}')

    # Prepare metadata
    d_MtDt = {
        f'V_{MdlN}': {
            'AVG': float(GDF[Fld].mean()),
            'variable': f'{Fld}',
            'details': f'{MdlN} vector file converted to TIF via Vtr_to_TIF function.',
        }
    }

    # Rasterize
    try:
        # Get dimensions from INI
        M = Mdl_N(MdlN)
        Pa_INI = M.Pa.INI
        Xmin, Ymin, Xmax, Ymax, cellsize, N_R, N_C = Mdl_Dmns(Pa_INI)

        x = np.arange(Xmin + cellsize / 2, Xmax, cellsize)
        y = np.arange(Ymax - cellsize / 2, Ymin, -cellsize)

        like = xra.DataArray(data=np.nan, coords={'y': y, 'x': x}, dims=('y', 'x'))
        sprint(f'🟢 - Created template grid from {Pa_INI.name} with dimensions: {like.shape}')
    except Exception as e:
        raise ValueError(f'🔴 - Could not create grid from INI ({e}). Ensure MdlN is correct and INI file exists.')

    GDF = round_Cols(GDF)
    DA = imod.prepare.rasterize(GDF, like=like, column=Fld)

    if SigDig:
        sprint(f'🟢 - Rounding to {SigDig} significant digits.')
        if not np.issubdtype(DA.dtype, np.floating):
            DA = DA.astype(float)

        vals = DA.values
        valid = (vals != 0) & np.isfinite(vals)
        if valid.any():
            v = vals[valid]
            magnitude = np.floor(np.log10(np.abs(v)))
            factor = 10.0 ** (SigDig - magnitude - 1)
            vals[valid] = np.rint(v * factor) / factor
            DA.values = vals

    # Cast to float32 to reduce file size and precision artifacts
    DA = DA.astype('float32')

    # Write to TIF
    to_TIF(DA, Pa_TIF, d_MtDt, CRS=CRS)
    sprint(f'🟢🟢🟢 - Saved vector to TIF at {Pa_TIF.name}')
    sprint(Sep)
