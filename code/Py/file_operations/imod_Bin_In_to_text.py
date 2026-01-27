import os
import struct

import numpy as np
import pandas as pd


def convert_riv_bin_to_text(bin_path):
    """
    Converts a binary RIV file to text.
    Attempts to detect if it is a flat C-binary or Fortran Unformatted binary.
    Assumes content is a list of RIV cells: Layer, Row, Column, Stage, Cond, Rbot.
    """
    if not os.path.exists(bin_path):
        print(f'❌ File not found: {bin_path}')
        return

    file_size = os.path.getsize(bin_path)
    print(f'📂 Processing: {bin_path}')
    print(f'   Size: {file_size} bytes')

    # Define the expected record structure for RIV package
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
                print('   ✅ Data looks like valid indices.')
                save_txt(df, bin_path)
                return
            else:
                print('   ⚠️ Indices check failed (negative values found). Trying Fortran format...')

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
                    print('   ✅ Identified as single-record Fortran Unformatted file.')

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
                            save_txt(df, bin_path)
                            return
                        else:
                            print('   ❌ Footer marker mismatch.')
                    else:
                        print(f'   ❌ Data size {header_val} is not a multiple of {record_size}.')
                else:
                    print('   Does not match single-record Fortran structure (Size + 8 != FileSize).')
    except Exception as e:
        print(f'   Error reading Fortran format: {e}')

    print('\n⚠️ Failed to automatically convert. The file format might differ from (L,R,C,Stage,Cond,Rbot).')


def save_txt(df, bin_path):
    out_path = bin_path + '.txt'
    try:
        with open(out_path, 'w') as f:
            # Write header
            f.write('# k i j stage cond rbot\n')
            # Convert to structured array for savetxt
            rec_array = df.to_records(index=False)
            # Use savetxt for speed and formatting
            np.savetxt(f, rec_array, fmt='%d %d %d %.6f %.6f %.6f')

        print(f'   💾 Saved text file to: {out_path}')
        print('   First 5 lines:')
        print(df.head())
    except Exception as e:
        print(f'Failed to save text file: {e}')


if __name__ == '__main__':
    # Path provided by user
    target_file = r'g:\models\NBr\Sim\NBr46\modflow6\imported_model\rivriv\riv-0.bin'
    convert_riv_bin_to_text(target_file)
