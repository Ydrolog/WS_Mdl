## --- Imports ---
from WS_Mdl.core.mdl import Mdl_N
from WS_Mdl.core.log import Up_log
from WS_Mdl.io.sim import freeze_pixi_env, get_elapsed_time_str
# from WS_Mdl.imod.mf6.obs import add as add_OBS
from WS_Mdl.imod.mf6.write import add_OBS_to_MF_In

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
iMOD5       =   True 

## Paths
M           =   Mdl_N(MdlN, iMOD5=iMOD5)
M.Pa.Mdl    =   M.Pa.Mdl
workdir:        M.Pa.Mdl
Pa_Smk      =   M.Pa.Mdl / 'code/snakemake'
Pa_temp     =   Pa_Smk / 'temp'

Pa_HD_OBS_WEL   =   M.Pa.MdlN / f'modflow6/imported_model/{MdlN}.OBS6' # 666

Dir_RIV_OBS     =   M.Pa.Mdl / 'In' / 'OBS' / 'RIV' / MdlN_OBS
Dir_DRN_OBS     =   M.Pa.Mdl / 'In' / 'OBS' / 'DRN' / MdlN_OBS
l_Fi_RIV_OBS    =   [p.name for p in Dir_RIV_OBS.iterdir() if p.is_file()]
l_Fi_DRN_OBS    =   [p.name for p in Dir_DRN_OBS.iterdir() if p.is_file()]
Pa_RIV_OBS_Src  =   [Dir_RIV_OBS / i for i in l_Fi_RIV_OBS]
Pa_DRN_OBS_Src  =   [Dir_DRN_OBS / i for i in l_Fi_DRN_OBS]
Pa_RIV_OBS_Dst  =   [M.Pa.Sim_In / i.replace(MdlN_OBS, MdlN) for i in l_Fi_RIV_OBS]
Pa_DRN_OBS_Dst  =   [M.Pa.Sim_In / i.replace(MdlN_OBS, MdlN) for i in l_Fi_DRN_OBS]

l_Fi_to_git     =   [M.Pa.WS / i for i in ['pixi.toml', 'pixi.lock', 'code/WS_Mdl']] # If any of these code files changes, the 
git_hash        =   shell(f"git -C {M.Pa.WS} rev-parse HEAD", read=True).strip()
git_tag         =   shell(f"git -C {M.Pa.WS} describe --tags --always", read=True, allow_error=True).strip() or "no_tag"

## Temp files (for completion validation)
log_Init        =   Pa_Smk / f"temp/Log_init_{MdlN}"
log_Sim         =   Pa_Smk / f"temp/Log_Sim_{MdlN}"
log_RIV_OBS     =   Pa_Smk / f"temp/Log_RIV_OBS_{MdlN}"
log_DRN_OBS     =   Pa_Smk / f"temp/Log_DRN_OBS_{MdlN}"
log_PRJ_to_TIF  =   Pa_Smk / f"temp/Log_PRJ_to_TIF_{MdlN}"
log_GXG         =   Pa_Smk / f"temp/Log_GXG_{MdlN}"
log_Up_MM       =   Pa_Smk / f"temp/Log_Up_MM_{MdlN}"
log_freeze_env  =   Pa_temp / f"Log_freeze_env_{MdlN}"

# --- Rules ---

def fail(job, execution): # Gets activated if any rule fails.
    Up_log(MdlN, {  'Sim end DT': DT.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'End Status': 'Failed'})
onerror: fail


rule all: # Final rule
    input:
        log_Up_MM,
        log_freeze_env
        
## -- PrP --
rule log_Init: # Sets status to running, and writes other info about therun. Has to complete before anything else.
    output:
        temp(log_Init)
    run:
        import socket
        device = socket.gethostname()
        Up_log(MdlN, {  'End Status':       'Running',
                            'PrP start DT':     DT.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'Sim device name':  device,
                            'Sim Dir':          M.Pa.MdlN,
                            '1st SP date':      DT.strptime(M.INI.SDATE, "%Y%m%d").strftime("%Y-%m-%d"),
                            'last SP date':     DT.strptime(M.INI.EDATE, "%Y%m%d").strftime("%Y-%m-%d")})
        Path(output[0]).touch() # Create the file to mark the rule as done.

rule freeze_pixi_env:
    input:
        l_Fi_to_git
    output:
        temp(log_freeze_env)
    run:
        git_hash, git_tag = freeze_pixi_env(MdlN)
        Up_log(MdlN, {  'Git hash': git_hash,
                        'Git tag': git_tag}) # Log git info
        Path(output[0]).touch() # Create the file to mark the rule as done.

rule Mdl_Prep: # Prepares Sim Ins (from Ins) via BAT file.
    input:
        log_Init,
        BAT = M.Pa.BAT,
        INI = M.Pa.INI,
        PRJ = M.Pa.PRJ
    output:
        M.Pa.NAM_Sim
    run:
        shell(f"call {input.BAT}")

        # Remove standard iMOD PoP that converts .HD to .IDF 
        with open(M.Pa.BAT_RUN, 'r+') as f:
            content = f.readlines()
            i = next(i for i, line in enumerate(content) if "ECHO MODFLOW finished, postprocessing started" in line)

            content = content[: i + 1]

            f.seek(0)
            f.writelines(content)
            f.truncate()

## -- PrSimP --
# rule add_HD_OBS_WEL:
#     input:
#         M.Pa.NAM_Sim
#     output:
#         Pa_HD_OBS_WEL
#     run:
#         add_OBS(MdlN, iMOD5=iMOD5)

rule add_RIV_OBS_copy: # By copying file.
    input:
        M.Pa.NAM_Sim,
        Pa_RIV_OBS_Src,
    output:
        Pa_RIV_OBS_Dst
    run:
        PKG = 'RIV'
        for i in range(len(Pa_RIV_OBS_Src)):
            Fi = Path(str(Pa_RIV_OBS_Src[i]).replace(MdlN_OBS, MdlN))
            sh.copy2(Pa_RIV_OBS_Src[i], Pa_RIV_OBS_Dst[i])
            add_OBS_to_MF_In(str_OBS=f" OBS6 FILEIN ./GWF_1/MODELINPUT/{Fi.name}", Pa=M.Pa.Sim_In / f"{Fi.stem}6")

rule add_DRN_OBS_copy: # By copying file.
    input:
        M.Pa.NAM_Sim,
        Pa_DRN_OBS_Src
    output:
        Pa_DRN_OBS_Dst
    run:
        PKG = 'DRN'
        for i in range(len(Pa_DRN_OBS_Src)):
            Fi = Path(str(Pa_DRN_OBS_Src[i]).replace(MdlN_OBS, MdlN))
            sh.copy2(Pa_DRN_OBS_Src[i], Pa_DRN_OBS_Dst[i])
            add_OBS_to_MF_In(str_OBS=f" OBS6 FILEIN ./GWF_1/MODELINPUT/{Fi.name}", Pa=M.Pa.Sim_In / f"{Fi.stem}6")

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
        Up_log(MdlN, {  'Sim start DT'  :   DT_Sim_Start.strftime("%Y-%m-%d %H:%M:%S")})
        sp.run(["cmd.exe", "/c", Pa_BAT_RUN], cwd=Pa_MdlN, check=True)
        Path(output[0]).touch() 
        Up_log(MdlN, {  'Sim end DT'    :   DT.now().strftime("%Y-%m-%d %H:%M:%S"),
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
        Up_log(MdlN, {  'PRJ_to_TIF':   1})
        Path(output[0]).touch() # Create the file to mark the rule as done.

rule GXG:
    input:
        log_Sim
    output:
        temp(log_GXG)
    run:
        HD_Bin_GXG_to_MBTIF(MdlN) # Calculate GXG and save as TIFs
        Up_log(MdlN, {  'GXG':   '1'})
        Path(output[0]).touch() # Create the file to mark the rule as done.

rule Up_MM:
    input:
        log_PRJ_to_TIF,
        log_GXG
    output:
        log_Up_MM
    run:
        update_MM(MdlN, MdlN_MM_B=MdlN_MM_B)      # Update MM 
        Up_log(MdlN, {  'PoP end DT':   DT.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'End Status':   'PoPed',
                        'Up_MM'     :   1}) # Update log
        Path(output[0]).touch()     # Create the file to mark the rule as done.
