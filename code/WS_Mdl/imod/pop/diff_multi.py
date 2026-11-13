from WS_Mdl.core.mdl import Mdl_N
from WS_Mdl.core.text import replace_MdlN
from WS_Mdl.xr.compare import Diff_MBTIF, Diff_TIF

__all__ = ['Diff_PoP_Par', 'Diff_PoP_ScN']


def Diff_PoP_Par(
    MdlN,
    B: str = None,
    Param: str = 'all',
):  # Improve to be able to do all parameters in PoP/Out folder.
    """
    Calcs Diff between TIF files of two MdLNs stored in M.Pa.PoP_Out_MdlN/Param folder. Saves the Diff as a TIF in the same dir as the MdlN PoP_Out files.
    Warning!: It doesn't check if the dates of the PoP_Out of the two MdLNs are the same.
    """

    M = Mdl_N(MdlN)

    B = B if B is not None else M.B

    Pa_PoP = M.Pa.PoP_Out_MdlN / Param
    l_Fi = [i for i in Pa_PoP.rglob('*.tif') if MdlN in i.name]
    # print(*l_GXG_Fi, sep='\n')

    if not l_Fi:
        raise FileNotFoundError(f"🔴 - No .tif files found for '{MdlN}' in: {Pa_PoP}")

    for Fi in l_Fi:
        Fi_2 = Pa_PoP / replace_MdlN(Fi, MdlN, B)
        Pa_Out = Fi.parent / replace_MdlN(Fi.name, MdlN, f'{MdlN}m{"".join(i for i in B if i.isdigit())}')
        # print(Pa_TIF_1, Pa_TIF_2, Pa_Out, end='\n---------------\n', sep='\n')
        try:
            try:
                Diff_TIF(Fi, Fi_2, band_name=f'{MdlN}m{B}', Pa_TIF_Out=Pa_Out)
            except Exception as e:
                print(type(e), e)
                Diff_MBTIF(Fi, Fi_2, band_name=f'{MdlN}m{B}', Pa_TIF_Out=Pa_Out)
        except (FileNotFoundError, OSError) as e:
            print(f'🔴 - Missing/unreadable file(s) for:\n{Fi}:\n {e}', end='------------\n')


def Diff_PoP_ScN(ScN: str, ScN_B: str, MdlN: str, Param: str = ''):
    """
    Diff (ScN minus ScN_B) between the TIFs in PoP/Out/<ScN>/<Param> and PoP/Out/<ScN_B>/<Param>.
    MdlN only supplies the paths (any member). The Diff is saved next to the ScN files as ..._<ScN>m<ScN_B>.tif.
    Warning!: It doesn't check if the periods and layers of the two scenarios are the same.
    """
    M = Mdl_N(MdlN)
    Pa_A = M.Pa.PoP / 'Out' / ScN / Param
    Pa_B = M.Pa.PoP / 'Out' / ScN_B / Param

    l_Fi = [i for i in Pa_A.rglob('*.tif') if i.name.endswith(f'_{ScN}.tif')]  # excludes earlier Diff files
    if not l_Fi:
        raise FileNotFoundError(f"🔴 - No .tif files found for '{ScN}' in: {Pa_A}")

    for Fi in l_Fi:
        Fi_2 = Pa_B / Fi.parent.relative_to(Pa_A) / Fi.name.replace(f'_{ScN}.tif', f'_{ScN_B}.tif')
        Pa_Out = Fi.parent / Fi.name.replace(f'_{ScN}.tif', f'_{ScN}m{ScN_B}.tif')
        try:
            try:
                Diff_TIF(Fi, Fi_2, band_name=f'{ScN}m{ScN_B}', Pa_TIF_Out=Pa_Out)
            except Exception as e:
                print(type(e), e)
                Diff_MBTIF(Fi, Fi_2, band_name=f'{ScN}m{ScN_B}', Pa_TIF_Out=Pa_Out)
        except (FileNotFoundError, OSError) as e:
            print(f'🔴 - Missing/unreadable file(s) for:\n{Fi}:\n {e}', end='------------\n')