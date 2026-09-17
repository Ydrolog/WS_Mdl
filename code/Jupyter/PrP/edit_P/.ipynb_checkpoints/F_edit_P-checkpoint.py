import numpy as np


def r_asc(file_path):
    """Reads an .asc file and returns the metadata and array data."""
    with open(file_path, 'r') as file:
        # Read metadata (first 6 lines)
        header = [next(file) for _ in range(6)]
        metadata = {line.split()[0]: line.split()[1] for line in header}

    # Load data into numpy array
    data = np.loadtxt(file_path, skiprows=6)
    return metadata, data


def save_asc(file_path, metadata, data):
    """Saves array data with metadata to an .asc file."""
    with open(file_path, 'w') as file:
        # Write metadata
        for key, value in metadata.items():
            file.write(f'{key} {value}\n')
        # Write data
        np.savetxt(file, data, fmt='%.6f')  # Adjust `fmt` for precision
