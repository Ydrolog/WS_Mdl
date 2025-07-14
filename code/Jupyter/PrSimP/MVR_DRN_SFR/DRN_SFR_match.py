"""
Utilities to match each model grid cell to the nearest cell where the
MODFLOW 6 SFR package is active.

This is an interim, geometry-based approach that simply finds the
nearest SFR-active cell in index (L, R, C) space using a
KD-tree.  Later you can replace the KD-tree query with a routing-aware
metric.

Functions
---------
match_cells_to_SFR(A_SFR_Atv, A_DRN_Atv=None, return_DF=True)
    Build a KD-tree of SFR-active cells and return either a
    ``pandas.DataFrame`` mapping each target cell to its nearest SFR cell
    (with Euclidean distance in index space) **or** an ``xarray.DataArray``
    holding the destination indices.

Example
-------
>>> DF = match_cells_to_SFR(A_SFR_Atv, A_DRN_Atv)
>>> DF.head()
  Tgt_L  Tgt_R  Tgt_C  SFR_L  SFR_R  SFR_C  distance
0            0          12          34          0       11       30  4.123106
1            0          13          35          0       11       30  3.605551
... etc.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
try:
    from scipy.spatial import cKDTree as KDTree
except ImportError:  # pragma: no cover
    raise ImportError("SciPy is required for KD-tree functionality — install it via"  \
                      " `pip install scipy`.")

__all__ = ["match_cells_to_SFR", "_build_SFR_tree", "_as_index_array",]

def _as_index_array(mask_da: xr.DataArray) -> np.ndarray:
    """Return an (N, 3) array of integer cell indices for *True* (SFR active) cells."""
    # Ensure we have 3 dimensions to cover (L, R, C)
    ndim = mask_da.ndim
    idx_arrays = np.where(mask_da.values)
    coords = np.column_stack(idx_arrays)
    if ndim == 2:  # insert L=0 for 2‑D grids
        coords = np.insert(coords, 0, 0, axis=1)
    return coords.astype(int)


def _build_SFR_tree(A_SFR_Atv: xr.DataArray) -> tuple[KDTree, np.ndarray]:
    """Return KDTree built on indices of SFR-active cells."""
    SFR_i = _as_index_array(A_SFR_Atv.astype(bool))
    if SFR_i.size == 0:
        raise ValueError("No active SFR cells found in `A_SFR_Atv`.")
    tree = KDTree(SFR_i)
    return tree, SFR_i


def match_cells_to_SFR(A_SFR_Atv: xr.DataArray,
                       A_DRN_Atv: xr.DataArray | None = None,
                       *,
                       return_DF: bool = True,):
    """Match each selected cell to the nearest SFR-active cell.

    Parameters
    ----------
    A_SFR_Atv
        ``xarray.DataArray`` with 1 for SFR-active cells, 0 otherwise.
    A_DRN_Atv
        Optional mask for DRN-active cells.  If supplied, only these
        cells are matched; otherwise **all** cells in the model domain
        are processed.
    return_DF
        If *True* (default) return a ``pandas.DataFrame``; otherwise
        return a new ``xarray.DataArray`` whose values hold the indices
        of the nearest SFR cell.

    Returns
    -------
    pandas.DataFrame | xr.DataArray
        Mapping of target cells to nearest SFR cells.
    """
    tree, SFR_i = _build_SFR_tree(A_SFR_Atv) # Build KD-tree from SFR-active cells

    Tgt_mask = (A_DRN_Atv.astype(bool) if A_DRN_Atv is not None else xr.ones_like(A_SFR_Atv, dtype=bool)) # Create target mask
    Tgt_i = _as_index_array(Tgt_mask) # 

    # Query KD‑tree (k=1 gives the single nearest neighbour)
    dist, nearest_pos = tree.query(Tgt_i, k=1)
    nearest_i = SFR_i[nearest_pos]

    if return_DF:
        DF = pd.DataFrame({"Tgt_L": Tgt_i[:, 0],
                           "Tgt_R": Tgt_i[:, 1],
                           "Tgt_C": Tgt_i[:, 2],
                           "SFR_L": nearest_i[:, 0],
                           "SFR_R": nearest_i[:, 1],
                           "SFR_C": nearest_i[:, 2],
                           "distance": dist,})
        
        DF[DF.columns.drop('distance')] = DF[DF.columns.drop('distance')] + 1 # Add 1 to all indices to convert from 0-based to 1-based indexing.
        return DF

    # Otherwise assemble an xarray.DataArray with tuple‑encoded indices
    # (expensive but sometimes handy).
    matched = np.empty(A_SFR_Atv.shape, dtype=object)
    matched[:] = None
    for t, s in zip(Tgt_i, nearest_i, strict=True):
        matched[tuple(t[1:])] = tuple(s)  # ignore L in target index for 2‑D grids

    DA = xr.DataArray(matched,
                      dims=A_SFR_Atv.dims,
                      coords=A_SFR_Atv.coords,
                      name="nearest_SFR_i",)
    return DA


if __name__ == "__main__":  # pragma: no cover
    # Very small self-test
    nz, ny, nx = 1, 5, 5
    A_SFR_Atv = xr.DataArray(np.random.randint(0, 2, size=(ny, nx)),
                             dims=("R", "C"), )
    
    A_DRN_Atv = xr.DataArray(np.random.randint(0, 2, size=(ny, nx)),
                             dims=("R", "C"), )
    print("SFR mask:\n", A_SFR_Atv.values)
    print("DRN mask:\n", A_DRN_Atv.values)
    result_DF = match_cells_to_SFR(A_SFR_Atv, A_DRN_Atv)
    print(result_DF.head())
