import imod
from WS_Mdl.core import Mdl_N
from WS_Mdl.imod.defaults import l_MSW_Par


def load_Par_Out(MdlN: str, Par: str, min_date: str, max_date: str) -> tuple:
    """Returns MSW Out Par - xarray DataArray &  the area array."""

    if Par not in l_MSW_Par:
        raise ValueError(f"Parameter '{Par}' is not supported.\nUse one of: {', '.join(l_MSW_Par)}")

    M = Mdl_N(MdlN)  # Load Mdl_N class

    # Load Area file
    Pa_Param = M.Pa.MSW / f'{Par}'
    A = imod.idf.open(Pa_Param / 'area_L1.IDF')  # Area array

    # Load IDF files
    l_Fi = sorted(Fi for Fi in Pa_Param.glob(f'{Par}_*_L*.IDF') if min_date <= Fi.stem.split('_')[1] <= max_date)
    A_Par = imod.idf.open(l_Fi)

    return A_Par, A
