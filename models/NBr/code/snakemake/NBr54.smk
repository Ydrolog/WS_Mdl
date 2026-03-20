## --- Imports ---
from WS_Mdl.core.mdl import Mdl_N
from WS_Mdl.core.log import update_log
from WS_Mdl.io.sim import freeze_pixi_env, get_elapsed_time_str
from WS_Mdl.imod.mf6.obs import add as add_OBS
from WS_Mdl.imod.mf6.write import add_OBS_to_MF_In

from snakemake.io import temp
from datetime import datetime as DT
from pathlib import Path
import os
import shutil as sh
import sys
sys.stdout.reconfigure(encoding='utf-8')    # Set stdout encoding to UTF-8
sys.stderr.reconfigure(encoding='utf-8')    # Set stderr encoding to UTF-8
os.environ["PYTHONUNBUFFERED"] = "1"        # Set Python to unbuffered mode (output is written immediately)

# --- Variables ---

## Options
MdlN        =   "NBr54"
MdlN_MM_B   =   'NBr52'
# Mdl         =   ''.join([i for i in MdlN if i.isalpha()])
iMOD5       =   False

## Paths
M           =   Mdl_N(MdlN, iMOD5=iMOD5)
M.Pa.Mdl      =   M.Pa.Mdl
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

def fail(job, excecution): # Gets activated if any rule fails.
    Up_log(MdlN, {  'Sim end DT': DT.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'End Status': 'Failed'})
onerror: fail


rule all: # Final rule
    input:
        log_Sim, # 666 might be redundant - as log_Up_MM is final step of the same line.
        log_Up_MM,
        log_freeze_env
        
## -- PrP --
rule log_Init: # Sets status to running, and writes other info about therun. Has to complete before anything else.
    output:
        temp(log_Init)
    run:
        import socket
        device = socket.gethostname()
        update_log(MdlN, {  'End Status':       'Running',
                            'PrP start DT':     DT.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'Sim device name':  device,
                            'Sim Dir':          Pa_Sim,
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
        from WS_Mdl.imod.prep import SFR_Mdl
        SFR_Mdl(
            MdlN = MdlN,
            Pa_Cond_A = M.Pa.WS / r"models\NBr\In\RIV\RIV_Cond_DETAILWATERGANGEN_NBr1.IDF",
            Pa_Cond_B = M.Pa.WS / r"models\NBr\In\RIV\RIV_Cond_DRN_NBr1.IDF",
            SFR_OBS_In = M.Pa.In / f'OBS/SFR/NBr40/NBr40_SFR_OBS_Pnt.csv',
            add_DRN_to_SFR=False,
)

## -- PrSimP --
rule add_HD_OBS_WEL:
    input:
        M.Pa.NAM_Sim
    output:
        Pa_HD_OBS_WEL
    run:
        UIM.add_OBS(MdlN, iMOD5=iMOD5)

rule add_RIV_OBS_copy: # By copying file.
    input:
        M.Pa.NAM_Sim,
        Pa_RIV_OBS_Src,
    output:
        Pa_RIV_OBS_Dst
    run:
        PKG = 'RIV'
        for i in range(len(Pa_RIV_OBS_Src)):
            Fi = Path(PBN(Pa_RIV_OBS_Src[i]))
            sh.copy2(Pa_RIV_OBS_Src[i], Pa_RIV_OBS_Dst[i])
            U.add_OBS_to_MF_In(str_OBS=f" OBS6 FILEIN ./GWF_1/MODELINPUT/{Fi}", Pa=PJ(M.Pa.NAM_Sim, f"{Fi.stem}6"))

rule add_DRN_OBS_copy: # By copying file.
    input:
        M.Pa.NAM_Sim,
        Pa_DRN_OBS_Src
    output:
        Pa_DRN_OBS_Dst
    run:
        PKG = 'DRN'
        for i in range(len(Pa_DRN_OBS_Src)):
            Fi = Path(PBN(Pa_DRN_OBS_Src[i]))
            sh.copy2(Pa_DRN_OBS_Src[i], Pa_DRN_OBS_Dst[i])
            U.add_OBS_to_MF_In(str_OBS=f" OBS6 FILEIN ./GWF_1/MODELINPUT/{Fi}", Pa=PJ(M.Pa.NAM_Sim, f"{Fi.stem}6"))

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
        os.chdir(M.Pa.MdlN) # Change directory to the model folder.
        DT_Sim_Start = DT.now()
        Up_log(MdlN, {  'Sim start DT'  :   DT_Sim_Start.strftime("%Y-%m-%d %H:%M:%S")})
        sp.run(Pa_BAT_RUN, shell=True, check=True)
        pathlib.Path(output[0]).touch() 
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
        from WS_Mdl.imod.prj import to_TIF as PRJ_to_TIF
        G.PRJ_to_TIF(MdlN, iMOD5=iMOD5, ) # Convert PRJ to TIFs
        Up_log(MdlN, {  'PRJ_to_TIF':   1})
        pathlib.Path(output[0]).touch() # Create the file to mark the rule as done.

rule GXG:
    input:
        log_Sim
    output:
        temp(log_GXG)
    run:
        from WS_Mdl.imod.pop.gxg import HD_Bin_GXG_to_MBTIF
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
        from WS_Mdl.io.qgis import update_MM
        G.Up_MM(MdlN, MdlN_MM_B=MdlN_MM_B)      # Update MM 
        Up_log(MdlN, {  'PoP end DT':   DT.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'End Status':   'PoPed',
                        'Up_MM'     :   1}) # Update log
        pathlib.Path(output[0]).touch()     # Create the file to mark the rule as done.