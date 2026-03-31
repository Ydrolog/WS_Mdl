import numpy as np
import pandas as pd
from scipy.interpolate import griddata

from WS_Mdl.core.style import sprint, warn


def all_layers(DA, dim='layer'):
    """
    Fills NA values in the given DataArray using linear interpolation, then fills any remaining NA values with the closest neighbor.
    """
    for L in DA[dim].values:
        if DA.sel({dim: L}).squeeze(drop=True).copy().isnull().all():  # If they're all null, repeat previous layer
            sprint(f'Layer {L} has no values. Repeating the previous layer.', style=warn)
            # Just copy the previous layer again to maintain the same number of layers, even if they're all the same
            L_to_copy = (
                L - 1 if L - 1 > 0 else L + 1
            )  # If it's the first layer and it's all null, copy the next layer instead (assuming it's not also all null)
            DA_L = DA.sel({dim: L_to_copy}).squeeze(drop=True).copy()
            DA_L = DA_L.assign_coords({dim: L})  # Otherwise it gets the previous L number...
        else:  # Otherwise, use the current layer
            DA_L = DA.sel({dim: L}).squeeze(drop=True).copy()
            # DA_L.plot()
            # print('INITIAL')
            # plt.show()

            A = DA_L.to_numpy().copy()
            mask = np.isnan(A)
            points = np.column_stack(np.where(~mask))
            values = A[~mask]
            grid = np.column_stack(np.where(mask))

            # Perform linear interpolation
            filled = griddata(points, values, grid, method='linear')

            # arr[mask] = filled
            # DA_L.data = arr
            # DA_L.plot()
            # print('LINEAR INTERPOLATION')
            # plt.show()

            # Perform nearest neighbor interpolation for remaining NaN values
            mask_rem = np.isnan(filled)
            filled[mask_rem] = griddata(points, values, grid[mask_rem], method='nearest')

            A[mask] = filled
            DA_L.data = A

            # Collapse to y,x only so save doesn't append pattern per extra dims
            DA_L.name = 'DA'
            DA_L = DA_L.assign_coords(time=pd.to_datetime(DA.time.values[0]))

            # DA_L.plot()
            # print('NEAREST NEIGHBOR INTERPOLATION FOR NAN VALUES')
            # plt.show()

        if np.isnan(DA_L).any():
            sprint(f'Warning: NaN values remain in DA_L for layer {L} after interpolation.', style=warn)

        # out_dir = Path("G:/models/NBr/In/DA/NBr57")
        # out_dir.mkdir(parents=True, exist_ok=True)
        # out_file = out_dir / f"DA_{pd.to_datetime(DA.time.values[0]).strftime('%Y%m%d')}_L{int(L)}_{MdlN_S}.idf"
        # imod.idf.save(out_file, DA_L, nodata=-999.9900, pattern=f"DA_{{time:%Y%m%d}}_L{{layer}}_{MdlN_S}.idf")

        DA_L = DA_L.broadcast_like(DA.sel({dim: L}))
        DA.sel({dim: L}).data = DA_L.data

    return DA
