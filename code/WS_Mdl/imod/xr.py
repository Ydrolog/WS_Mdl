import xarray as xra
from WS_Mdl.core.path import MdlN_Pa
from WS_Mdl.core.style import sprint
from WS_Mdl.imod.ini import Mdl_Dmns


def clip_Mdl_Aa(
    xr_data: xra.DataArray | xra.Dataset,
    MdlN: str = None,
    Pa_INI: str = None,
    l_L=None,
    Lmin: int = None,
    Lmax: int = None,
    x_dim: str = 'x',
    y_dim: str = 'y',
    L_dim: str = 'layer',
) -> xra.DataArray | xra.Dataset:
    """
    Clips an xarray DataArray or Dataset to the model area defined in an INI file, with optional layer subsetting.

    Parameters:
    -----------
    xr_data : xr.DataArray or xr.Dataset
        The xarray data to clip
    MdlN : str, optional
        Model name to automatically get INI path via MdlN_Pa
    Pa_INI : str, optional
        Direct path to INI file (alternative to MdlN)
    l_L : list-like, optional
        List of layers to select (e.g., [1, 3, 5])
    Lmin : int, optional
        Minimum layer index (inclusive)
    Lmax : int, optional
        Maximum layer index (inclusive)
    x_dim : str, default 'x'
        Name of the x coordinate dimension
    y_dim : str, default 'y'
        Name of the y coordinate dimension
    L_dim : str, default 'layer'
        Name of the layer coordinate dimension

    Returns:
    --------
    xr.DataArray or xr.Dataset
        Clipped xarray data

    Raises:
    -------
    ValueError
        If neither MdlN nor Pa_INI is provided, or if required dimensions are missing
    """

    # Get INI file path
    if Pa_INI is None:
        if MdlN is None:
            raise ValueError('Either MdlN or Pa_INI must be provided')
        d_Pa = MdlN_Pa(MdlN)
        Pa_INI = d_Pa['INI']

    # Get model dimensions from INI file
    Xmin, Ymin, Xmax, Ymax, cellsize, N_R, N_C = Mdl_Dmns(Pa_INI)

    # Check if required dimensions exist
    if x_dim not in xr_data.coords:
        raise ValueError(f"X dimension '{x_dim}' not found in data coordinates: {list(xr_data.coords.keys())}")
    if y_dim not in xr_data.coords:
        raise ValueError(f"Y dimension '{y_dim}' not found in data coordinates: {list(xr_data.coords.keys())}")

    sprint(f'Clipping xarray data to model area: X=[{Xmin}, {Xmax}], Y=[{Ymin}, {Ymax}]')

    # Check y-coordinate order and adjust slice accordingly
    y_coords = xr_data.coords[y_dim].values
    if len(y_coords) > 1 and y_coords[0] > y_coords[-1]:  # Descending order (big to small)
        y_slice = slice(Ymax, Ymin)
        sprint('Y coordinates in descending order, using slice(Ymax, Ymin)')
    else:  # Ascending order (small to big) or single value
        y_slice = slice(Ymin, Ymax)
        sprint('Y coordinates in ascending order, using slice(Ymin, Ymax)')

    # Clip to spatial extent
    clipped = xr_data.sel({x_dim: slice(Xmin, Xmax), y_dim: y_slice})

    # Handle layer subsetting if layer dimension exists
    if L_dim in xr_data.coords:
        if l_L is not None:
            sprint(f'Selecting specific layers: {l_L}')
            clipped = clipped.sel({L_dim: l_L})
        elif Lmin is not None or Lmax is not None:
            # Build slice for layer range
            layer_slice = slice(Lmin, Lmax)
            sprint(f'Selecting layer range: {Lmin} to {Lmax}')
            clipped = clipped.sel({L_dim: layer_slice})
    elif l_L is not None or Lmin is not None or Lmax is not None:
        sprint(f"Warning: Layer subsetting requested but dimension '{L_dim}' not found in data")

    sprint('🟢🟢 - Successfully clipped xarray data to model area')
    return clipped
