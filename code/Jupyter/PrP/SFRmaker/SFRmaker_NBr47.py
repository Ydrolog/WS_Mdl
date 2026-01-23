# 0. Basics
## 0.0. Imports
import importlib as IL
from datetime import datetime as DT
from os.path import dirname as PDN
from os.path import join as PJ

import WS_Mdl.geo as G
import WS_Mdl.utils as U
import WS_Mdl.utils_imod as UIM

IL.reload(U)
IL.reload(UIM)
IL.reload(G)
# Import sfrmaker and other necessary packages for SFR network creation
from itables import init_notebook_mode

init_notebook_mode(all_interactive=True)
## 0.1. Options
MdlN = 'NBr47'
MdlN_SFR_In = 'NBr40'
Pa_Gpkg_ = PJ(U.Pa_WS, rf'g:\models\NBr\PrP\SFR\BrabantseDelta\Gpkg\WBD_detail_SW_NW_cleaned_{MdlN_SFR_In}.gpkg')
U.set_verbose(False)
# Load paths and variables from PRJ & INI
d_Pa = U.get_MdlN_Pa(MdlN)
Pa_PRJ = d_Pa['PRJ']
Dir_PRJ = PDN(Pa_PRJ)
d_INI = U.INI_to_d(d_Pa['INI'])
Xmin, Ymin, Xmax, Ymax, cellsize, N_R, N_C = U.Mdl_Dmns_from_INI(d_Pa['INI'])
SP_date_1st, SP_date_last = [DT.strftime(DT.strptime(d_INI[f'{i}'], '%Y%m%d'), '%Y-%m-%d') for i in ['SDATE', 'EDATE']]
dx = dy = float(d_INI['CELLSIZE'])
U.Mdl_Dmns_from_INI(d_Pa['INI'])
l_X_Y_Cols = ['Xstart', 'Ystart', 'Xend', 'Yend']
l_Circ_IDs = [6561, 8788, 18348]
