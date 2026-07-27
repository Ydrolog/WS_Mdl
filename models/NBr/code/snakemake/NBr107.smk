## --- Imports ---
from WS_Mdl.core.mdl import Mdl_N
from WS_Mdl.core.log import Up_log
from WS_Mdl.io.sim import get_elapsed_time_str

from snakemake.io import temp
from datetime import datetime as DT
from pathlib import Path
import subprocess as sp
import os
import shutil as sh
import sys
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True, write_through=True)    # Set stdout encoding to UTF-8
sys.stderr.reconfigure(encoding='utf-8', line_buffering=True, write_through=True)    # Set stderr encoding to UTF-8
os.environ["PYTHONUNBUFFERED"] = "1"        # Set Python to unbuffered mode (output is written immediately)

# --- Variables ---

## Options
MdlN        =   'NBr107'
iMOD5       =   False
MdlN_MM_B   =   'NBr104'

## Paths
M           =   Mdl_N(MdlN, iMOD5=iMOD5)
workdir:        M.Pa.Mdl

# MF6 Options
M.Sim.Bin_Ins   = False
M.Sim.save_head = None # We use OBS instead, which reduces Out size significantly.

MdlN_HD_OBS         =   'NBr107'
Pa_HD_OBS_Src       =   M.Pa.In / f'OBS/HD/{MdlN_HD_OBS}/HD_{MdlN_HD_OBS}.OBS6'
Pa_HD_OBS_Dst       =   M.Pa.Sim_In / f'HD_{MdlN}.OBS6'
Pa_Shp_catchment    =   M.Pa.WS / r'models\NBr\PoP\common\Pgn\Chaamse_beek\catchment_chaamsebeek_ulvenhout.shp'
PoP_end_year        =   2001
l_Diff_PoP_Par      =   ['GW_HD_AVGs/L1']

## Temp files (for completion validation)
Pa_temp             =   M.Pa.Smk.parent / 'temp'
log_Init            =   Pa_temp / f"Log_init_{MdlN}"
log_RIV_OBS         =   Pa_temp / f"Log_RIV_OBS_{MdlN}"
log_DRN_OBS         =   Pa_temp / f"Log_DRN_OBS_{MdlN}"
log_fix_MSW_area    =   Pa_temp / f"Log_fix_MSW_area_{MdlN}"
log_Sim             =   Pa_temp / f"Log_Sim_{MdlN}"
log_PRJ_to_TIF      =   Pa_temp / f"Log_PRJ_to_TIF_{MdlN}"
log_HD_AVGs         =   Pa_temp / f"Log_HD_AVGs_{MdlN}"
log_GXG             =   Pa_temp / f"Log_GXG_{MdlN}"
log_HD_Pctls        =   Pa_temp / f"Log_HD_Pctls_{MdlN}"
log_Diff            =   Pa_temp / f"Log_Diff_PoP_Par_{MdlN}"
log_WB              =   Pa_temp / f"Log_WB_{MdlN}"
log_upload          =   Pa_temp / f"Log_upload_{MdlN}"

# --- Rules ---

def fail(job, execution): # Gets activated if any rule fails.
    Up_log(MdlN, {  'Sim end DT': DT.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'End Status': 'Failed'})
onerror: fail


rule all: # Final rule
    input:
        M.Pa.MM,
        # log_upload # 666 make this the only output.

## -- PrP --
rule log_Init: # Sets status to running, and writes other info about therun. Has to complete before anything else.
    output:
        touch(temp(log_Init))
    run:
        import socket
        device = socket.gethostname()
        Up_log(MdlN, {  'End Status'        :   'Running',
                        'PrP start DT'      :   DT.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'Sim device name'   :   device,
                        'Sim Dir'           :   M.Pa.MdlN,
                        '1st SP date'       :   DT.strptime(M.INI.SDATE, "%Y%m%d").strftime("%Y-%m-%d"),
                        'last SP date'      :   DT.strptime(M.INI.EDATE, "%Y%m%d").strftime("%Y-%m-%d")})

rule Mdl_Prep: # Prepares Sim Ins (from Ins) via BAT file.
    input:
        log_Init,
        BAT = M.Pa.BAT,
        INI = M.Pa.INI,
        PRJ = M.Pa.PRJ
    output:
        M.Pa.NAM_Sim
    run:
        from WS_Mdl.imod.prep import Sim
        Sim(M, SFR=False)

## -- PrSimP --
rule add_HD_OBS_copy: # Copying so I can manually make a file that contains OBS for both OBS Pnts and whole layers.
    input:
        M.Pa.NAM_Sim
    output:
        Pa_HD_OBS_Dst
    run:
        from WS_Mdl.imod.mf6.nam import add_Pkg
        sh.copy2(Pa_HD_OBS_Src, Pa_HD_OBS_Dst) # Copy the file to create a new one with the same content.
        add_Pkg(MdlN, fr'  OBS6 .\imported_model\HD_{MdlN}.OBS6 HD_OBS')  # Add to NAM


rule add_RIV_OBS:
    input:
        M.Pa.NAM_Sim,
    output:
        touch(log_RIV_OBS)
    run:
        from WS_Mdl.imod.mf6.obs import add_within_polygon
        add_within_polygon(Pa_Shp = Pa_Shp_catchment,
            MdlN = MdlN,
            Pkg = 'RIV',
            Opt = """BEGIN OPTIONS\n  DIGITS 4\n  PRINT_INPUT\nEND OPTIONS\n\n""")
        
rule add_DRN_OBS:
    input:
        M.Pa.NAM_Sim
    output:
        touch(log_DRN_OBS)
    run:
        from WS_Mdl.imod.mf6.obs import add_within_polygon
        add_within_polygon(Pa_Shp = Pa_Shp_catchment,
            MdlN = MdlN,
            Pkg = 'DRN',
            Opt = """BEGIN OPTIONS\n DIGITS 4\n  PRINT_INPUT\nEND OPTIONS\n\n""")

rule fix_MSW_area:
    input:
        M.Pa.NAM_Sim
    output:
        touch(log_fix_MSW_area)
    run:
        sh.copy2(r"g:\code\Jupyter\PoP\compare_Ins\area_svat_NBr61.inp", M.Pa.MSW / "area_svat.inp")

## -- Sim ---
rule Sim: # Runs the simulation via BAT file.
    input:
        Pa_HD_OBS_Dst,
        log_RIV_OBS,
        log_DRN_OBS,
        log_fix_MSW_area
    output:
        touch(log_Sim)
    run:
        os.chdir(M.Pa.MdlN) # Change directory to the model folder.
        DT_Sim_Start = DT.now()
        Up_log(MdlN, {  'Sim start DT'  :   DT_Sim_Start.strftime("%Y-%m-%d %H:%M:%S")})
        M.Pa.Sim_Out.mkdir(parents=True, exist_ok=True) # Create Out folder
        sp.run([M.Pa.coupler_Exe, M.Pa.TOML], shell=True, check=True)
        Up_log(MdlN, {  'Sim end DT'    :   DT.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'Sim Dur'       :   get_elapsed_time_str(DT_Sim_Start),
                        'End Status'    :   'Completed'})

# -- PoP ---
rule PRJ_to_TIF:
    input:
        log_Sim
    output:
        touch(log_PRJ_to_TIF)
    run:
        from WS_Mdl.imod.prj import to_TIF as PRJ_to_TIF
        PRJ_to_TIF(MdlN, iMOD5=iMOD5) # Convert PRJ to TIFs
        Up_log(MdlN, {  'PRJ_to_TIF':   1})

rule p_HD_AVGs: # Process HD OBS Out Bin data into TIF files with AVG heads. Then Calc Diff to B
    input:
        log_Sim
    output:
        touch(log_HD_AVGs)
    run:
        from WS_Mdl.imod.pop.hd import c_HD_Bin_AVGs
        c_HD_Bin_AVGs(MdlN, end_year=PoP_end_year)
        Up_log(MdlN, {'p_HD_AVGs' :   1})

rule p_HD_OBS_TS:
    input:
        log_Sim
    output:
        touch(M.Pa.PoP_Out_MdlN / f'GW_HD_OBS/metadata.txt')
    run:
        from WS_Mdl.imod.pop.hd import p_HD_OBS_TS
        p_HD_OBS_TS(MdlN)
        Up_log(MdlN, {'p_HD_OBS_TS' :   1})

rule GXG:
    input:
        log_Sim
    output:
        touch(log_GXG)
    run:
        from WS_Mdl.imod.pop.gxg import HD_Bin_GXG_to_MBTIF
        HD_Bin_GXG_to_MBTIF(MdlN) # Calculate GXG and save as TIFs
        Up_log(MdlN, {  'GXG':   1})

rule p_HD_Pctls:
    input:
        log_Sim
    output:
        touch(log_HD_Pctls)
    run:
        from WS_Mdl.imod.pop.hd import c_HD_Bin_Pctls
        c_HD_Bin_Pctls(MdlN)
        Up_log(MdlN, {  'HD_Pctls':   1})


rule Diff_PoP_Par:
    input:
        log_HD_AVGs
    output:
        touch(log_Diff)
    run:
        from WS_Mdl.xr.compare import Diff_PoP_Par
        for P in l_Diff_PoP_Par:
            Diff_PoP_Par(MdlN, M.B, P)
        Up_log(MdlN, {'Diff_PoP_Par' :   ", ".join(l_Diff_PoP_Par)})

rule Up_MM:
    input:
        log_PRJ_to_TIF,
        log_HD_AVGs,
        M.Pa.PoP_Out_MdlN / f'GW_HD_OBS/metadata.txt',
        log_GXG,
        log_HD_Pctls,
        log_Diff,
    output:
        touch(M.Pa.MM)
    run:
        from WS_Mdl.io.qgis import Up_MM
        Up_MM(MdlN, MdlN_MM_B=MdlN_MM_B)      # Update MM 
        Up_log(MdlN, {  'PoP end DT':   DT.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'End Status':   'PoPed',
                        'Up_MM'     :   1}) # Update log

rule WB:
    input:
        log_Sim
    output:
        touch(log_WB)
    run:
        from WS_Mdl.imod.pop.wb import Diff_to_xlsx
        Diff_to_xlsx(MdlN, M.B)

rule Upl: # Uploads the PoP Out files to iBridges. This is the final step of the workflow.
    input:
        M.Pa.MM,
        log_WB
    output:
        touch(log_upload)
    run:
        from WS_Mdl.io.ibridges import Upl_Sim_Out
        Upl_Sim_Out(MdlN)