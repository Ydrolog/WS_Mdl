import subprocess
from datetime import datetime as DT
from datetime import timedelta

import imod
import WS_Mdl.core.style as style
from WS_Mdl.core.mdl import Mdl_N
from WS_Mdl.core.style import Sep, sprint


def from_HD_Out(MdlN: str, date_B_YMD: str, MdlN_B: bool | str = True):
    """
    Creates SHD files from HD Out of B Simulation.
    - MdlN: Name of the model to create SHD files for.
    - date_B_YMD: Date of the HD Out to use, in YMD format (e.g. '20100113'). SHD date will be the next one.
    - MdlN_B: B Sim (as per RunLog) will be used by default. If you want to use a different Sim as B, pass it as an argument.
    """
    sprint(Sep, verbose_Out=False)

    M = Mdl_N(MdlN)

    if MdlN_B is True:
        MdlN_B = M.B
    M_B = Mdl_N(MdlN_B)

    date_B = DT.strptime(date_B_YMD, '%Y%m%d')
    date_S_YMD = DT.strftime(date_B + timedelta(days=1), '%Y%m%d')
    # 1. Read HD, select SP.
    HDs = (
        imod.mf6.open_hds(M_B.Pa.HD_Out_Bin, M_B.Pa.DIS_GRB, simulation_start_time=DT.strptime(M_B.INI.sdate, '%Y%m%d'))
        .sel(time=date_B)
        .compute()
    )

    Pa_Out = M.Pa.In / f'SHD/{MdlN}'
    imod.idf.save(Pa_Out, HDs, pattern=f'SHD_{date_S_YMD}_L{{layer}}_{MdlN}.idf')

    # 2. Write SHD block # For now it's just printed so you can copy it to PRJ. In the future, we can automate the addition to PRJ as well.
    sprint('The SHD block for your PRJ file has been copied to your clipboard!\n', style=style.blue, verbose_In=True)

    l_text = ['0001,(SHD),1, Starting Heads', f'001,{str(HDs.layer.data.shape[0]).zfill(3)}']
    for i in range(HDs.layer.data.shape[0]):
        l_text.append(
            rf" 1,2, {i + 1:03},   1.000000    ,   0.000000    ,  -999.9900    , '..\..\In\SHD\{MdlN}\SHD_{date_S_YMD}_L{i + 1}_{MdlN}.idf' >>> (shd) starting heads (idf) <<<"
        )
    subprocess.run('clip', input='\n'.join(l_text), text=True, check=True)

    # 3. Write metadata file in the same folder
    with open(Pa_Out / '_metadata.txt', 'w') as f:
        f.write(
            f'This file was produced by WS_Mdl/imod/mf6/shd.py\nfrom: {M_B.Pa.HD_Out_Bin}\nfor the date: {date_S_YMD}.'
        )

    sprint(Sep)
