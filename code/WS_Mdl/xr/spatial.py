from WS_Mdl.core.style import sprint
from WS_Mdl.core.mdl import Mdl_N


def get_value(A, X, Y, dx, dy, L=None, method='nearest', validate=True):
    """
    - Gets value from xarray DataArray A at coordinates X, Y, L (if provided).
    - If validate is True, checks if value coordinates are within the same cell.
    """

    sel = {'x': X, 'y': Y}
    if L:
        sel['layer'] = L

    value = A.sel(sel, method=method)

    if validate:
        X_A, Y_A = value['x'].values, value['y'].values
        if not (abs(X - X_A) <= abs(dx / 2)) or not (abs(Y - Y_A) <= abs(dy / 2)):
            print(
                f'🟡 - Retrieved value coordinates (X: {X_A}, Y: {Y_A}) differ from requested coordinates (X: {X}, Y: {Y}) by more than half the cell size (dx: {dx}, dy: {dy}).\nThat may be valid if the resolution of the two arrays is different, but you should double-check.'
            )

    return value


def clip_Mdl_area(A, MdlN):
    """
    - Clips xarray DataArray A to the model area defined by MdlN's INI.window.
    - Returns the clipped DataArray.
    """

    M = Mdl_N(MdlN)
    Xmin, Ymin, Xmax, Ymax = (float(i) for i in M.INI.window.split(','))

    if A.y.values[0] > A.y.values[-1]:
        A = A.reindex(y=A.y[::-1])
        sprint(f"🟡 - Reversed y-axis of DataArray to match model area orientation.")

    A_clipped = A.sel(x=slice(Xmin, Xmax), y=slice(Ymin, Ymax))
    return A_clipped
