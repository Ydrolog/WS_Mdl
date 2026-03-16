## --- Imports ---
from WS_Mdl.core.mdl import Mdl_N
from WS_Mdl.core.log import update_log
from WS_Mdl.io.sim import freeze_pixi_env, get_elapsed_time_str
from WS_Mdl.imod.mf6.obs import add as add_OBS
from WS_Mdl.imod.mf6.write import add_OBS_to_MF_In
from WS_Mdl.imod.prj import to_TIF as PRJ_to_TIF
from WS_Mdl.imod.pop.gxg import HD_Bin_GXG_to_MBTIF
from WS_Mdl.io.qgis import update_MM

from snakemake.io import temp
from datetime import datetime as DT
from pathlib import Path
import subprocess as sp
import os
import shutil as sh
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
os.environ["PYTHONUNBUFFERED"] = "1"

# --- Variables ---

## Options
MdlN        =   "NBr52"
MdlN_MM_B   =   'NBr13'
MdlN_OBS    =   'NBr50'
Mdl         =   ''.join([i for i in MdlN if i.isalpha()])
# rule_       =   "(L == 1)"
iMOD5       =   True 

## Paths
M           =   Mdl_N(MdlN, iMOD5=iMOD5)
d_Pa        =   M.Pa
Pa_WS       =   d_Pa.WS
workdir:        Pa_WS
Pa_Mdl      =   d_Pa.Mdl
Pa_Sim      =   d_Pa.Sim
Pa_MdlN     =   d_Pa.MdlN
Pa_Smk      =   d_Pa.Smk.parent
Pa_temp     =   Pa_Smk / 'temp'
Pa_BAT_RUN  =   Pa_MdlN / 'RUN.BAT'

Pa_HD_OBS_Ogn   =   d_Pa.In / 'OBS' / 'HD' / MdlN / f'{MdlN}.HD.OBS6'
Pa_HD_OBS_WEL   =   Pa_MdlN / 'GWF_1' / f'MODELINPUT/{MdlN}.OBS6'
Pa_HD_OBS       =   Pa_MdlN / 'GWF_1' / f'MODELINPUT/{MdlN}.HD.OBS6'
Pa_NAM          =   Pa_MdlN / 'GWF_1' / f'{MdlN}.NAM'

Dir_RIV_OBS     =   Pa_Mdl / 'In' / 'OBS' / 'RIV' / MdlN_OBS
Dir_DRN_OBS     =   Pa_Mdl / 'In' / 'OBS' / 'DRN' / MdlN_OBS
l_Fi_RIV_OBS    =   [p.name for p in Dir_RIV_OBS.iterdir() if p.is_file()]
l_Fi_DRN_OBS    =   [p.name for p in Dir_DRN_OBS.iterdir() if p.is_file()]
Pa_RIV_OBS_Src  =   [Dir_RIV_OBS / i for i in l_Fi_RIV_OBS]
Pa_DRN_OBS_Src  =   [Dir_DRN_OBS / i for i in l_Fi_DRN_OBS]
Pa_RIV_OBS_Dst  =   [d_Pa.Sim_In / i.replace(MdlN_OBS, MdlN) for i in l_Fi_RIV_OBS]
Pa_DRN_OBS_Dst  =   [d_Pa.Sim_In / i.replace(MdlN_OBS, MdlN) for i in l_Fi_DRN_OBS]

Pa_HED, Pa_CBC  =   [Pa_MdlN / 'GWF_1' / 'MODELOUTPUT' / i for i in ['HEAD/HEAD.HED', 'BUDGET/BUDGET.CBC']]

l_Fi_to_git     =   [Pa_WS / i for i in ['pixi.toml', 'pixi.lock', 'code/WS_Mdl']] # If any of these code files changes, the 
git_hash        =   shell(f"git -C {Pa_WS} rev-parse HEAD", read=True).strip()
git_tag         =   shell(f"git -C {Pa_WS} describe --tags --always", read=True, allow_error=True).strip() or "no_tag"

## Temp files (for completion validation)
log_Init        =   Pa_Smk / f"temp/Log_init_{MdlN}"
log_Sim         =   Pa_Smk / f"temp/Log_Sim_{MdlN}"
log_PRJ_to_TIF  =   Pa_Smk / f"temp/Log_PRJ_to_TIF_{MdlN}"
log_GXG         =   Pa_Smk / f"temp/Log_GXG_{MdlN}"
log_Up_MM       =   Pa_Smk / f"temp/Log_Up_MM_{MdlN}"
log_freeze_env  =   Pa_temp / f"Log_freeze_env_{MdlN}"

# --- Rules ---

def fail(job, execution): # Gets activated if any rule fails.
    update_log(MdlN, {  'Sim end DT': DT.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'End Status': 'Failed'})
onerror: fail


rule all: # Final rule
    input:
        log_Sim,
        log_Up_MM,
        log_freeze_env
        
## -- PrP --
rule log_Init: # Sets status to running, and writes other info about therun. Has to complete before anything else.
    output:
        temp(log_Init)
    run:
        import socket
        device = socket.gethostname()
        d_INI = M.INI

        update_log(MdlN, {  'End Status':       'Running',
                            'PrP start DT':     DT.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'Sim device name':  device,
                            'Sim Dir':          Pa_Sim,
                            '1st SP date':      DT.strptime(d_INI['SDATE'], "%Y%m%d").strftime("%Y-%m-%d"),
                            'last SP date':     DT.strptime(d_INI['EDATE'], "%Y%m%d").strftime("%Y-%m-%d")})
        Path(output[0]).touch() # Create the file to mark the rule as done.
        
        # Move this to a new rule later, when you've made it safe for multiple rules to write to the same file. Low priority...
        update_log(MdlN, {  'Git hash': git_hash,
                        'Git tag': git_tag}) # Log git info

rule freeze_pixi_env:
    input:
        l_Fi_to_git
    output:
        temp(log_freeze_env)
    run:
        git_hash, git_tag = freeze_pixi_env(MdlN)
        update_log(MdlN, {  'Git hash': git_hash,
                        'Git tag': git_tag}) # Log git info
        Path(output[0]).touch() # Create the file to mark the rule as done.

rule Mdl_Prep: # Prepares Sim Ins (from Ins) via BAT file.
    input:
        log_Init,
        BAT = M.Pa.BAT,
        INI = M.Pa.INI,
        PRJ = M.Pa.PRJ
    output:
        d_Pa.NAM_Sim
    run:
        shell(f"call {input.BAT}")

        # Remove standard iMOD PoP that converts .HD to .IDF 
        with open(Pa_BAT_RUN, 'r+') as f:
            content = f.readlines()
            i = next(i for i, line in enumerate(content) if "ECHO MODFLOW finished, postprocessing started" in line)

            content = content[: i + 1]

            f.seek(0)
            f.writelines(content)
            f.truncate()

## -- PrSimP --
rule add_HD_OBS_WEL:
    input:
        d_Pa.NAM_Sim
    output:
        Pa_HD_OBS_WEL
    run:
        add_OBS(MdlN, iMOD5=iMOD5)

# This rule doesn't work cause the OBS file contains all cells, and MF6 doesn't allow OBS to be added to cells that are excluded from the Sim. Cells are excluded (by iMOD) cause not all layers are present everywhere.
# rule add_HD_OBS:
#     input:
#         Pa_BAT_RUN
#     output:
#         Pa_HD_OBS
#     run:cs
#         sh.copy2(Pa_HD_OBS_Ogn, Pa_HD_OBS)
#         U.add_PKG_to_NAM(MdlN=MdlN, str_PKG=f' OBS6 ./GWF_1/MODELINPUT/{MdlN}.HD.OBS6 OBS_HD',  iMOD5=iMOD5)

rule add_RIV_OBS_copy: # By copying file.
    input:
        d_Pa.NAM_Sim,
        Pa_RIV_OBS_Src,
    output:
        Pa_RIV_OBS_Dst
    run:
        PKG = 'RIV'
        for i in range(len(Pa_RIV_OBS_Src)):
            Fi = Path(str(Pa_RIV_OBS_Src[i]).replace(MdlN_OBS, MdlN))
            sh.copy2(Pa_RIV_OBS_Src[i], Pa_RIV_OBS_Dst[i])
            add_OBS_to_MF_In(str_OBS=f" OBS6 FILEIN ./GWF_1/MODELINPUT/{Fi.name}", Pa=d_Pa.Sim_In / f"{Fi.stem}6")

rule add_DRN_OBS_copy: # By copying file.
    input:
        d_Pa.NAM_Sim,
        Pa_DRN_OBS_Src
    output:
        Pa_DRN_OBS_Dst
    run:
        PKG = 'DRN'
        for i in range(len(Pa_DRN_OBS_Src)):
            Fi = Path(str(Pa_DRN_OBS_Src[i]).replace(MdlN_OBS, MdlN))
            sh.copy2(Pa_DRN_OBS_Src[i], Pa_DRN_OBS_Dst[i])
            add_OBS_to_MF_In(str_OBS=f" OBS6 FILEIN ./GWF_1/MODELINPUT/{Fi.name}", Pa=d_Pa.Sim_In / f"{Fi.stem}6")

## -- Sim ---
rule Sim: # Runs the simulation via BAT file.
    input:
        Pa_HD_OBS_WEL,
        Pa_RIV_OBS_Dst,
        Pa_DRN_OBS_Dst,
        # Pa_HD_OBS
    output:
        temp(log_Sim)
    run:
        DT_Sim_Start = DT.now()
        update_log(MdlN, {  'Sim start DT'  :   DT_Sim_Start.strftime("%Y-%m-%d %H:%M:%S")})
        sp.run(["cmd.exe", "/c", Pa_BAT_RUN], cwd=Pa_MdlN, check=True)
        Path(output[0]).touch() 
        update_log(MdlN, {  'Sim end DT'    :   DT.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'Sim Dur'       :   get_elapsed_time_str(DT_Sim_Start),
                            'End Status'    :   'Completed'})

# -- PoP ---
rule PRJ_to_TIF:
    input:
        log_Sim
    output:
        temp(log_PRJ_to_TIF)
    run:
        PRJ_to_TIF(MdlN, iMOD5=iMOD5, ) # Convert PRJ to TIFs
        update_log(MdlN, {  'PRJ_to_TIF':   1})
        Path(output[0]).touch() # Create the file to mark the rule as done.

rule GXG:
    input:
        log_Sim
    output:
        temp(log_GXG)
    run:
        HD_Bin_GXG_to_MBTIF(MdlN) # Calculate GXG and save as TIFs
        update_log(MdlN, {  'GXG':   '1'})
        Path(output[0]).touch() # Create the file to mark the rule as done.

rule Up_MM:
    input:
        log_PRJ_to_TIF,
        log_GXG
    output:
        log_Up_MM
    run:
        update_MM(MdlN, MdlN_MM_B=MdlN_MM_B)      # Update MM 
        update_log(MdlN, {  'PoP end DT':   DT.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'End Status':   'PoPed',
                        'Up_MM'     :   1}) # Update log
        Path(output[0]).touch()     # Create the file to mark the rule as done.
