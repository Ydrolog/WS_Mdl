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
MdlN        =   'NBr104'
MdlN_MM_B   =   'NBr100'
iMOD5       =   False

## Paths
M           =   Mdl_N(MdlN, iMOD5=iMOD5)
workdir:        M.Pa.Mdl

# MF6 Options
M.Sim.Bin_Ins = False
M.Sim.save_head = None # We use OBS instead, which reduces Out size significantly.

# SFR Options
Pa_SFR_GPkg         = M.Pa.In / 'SFR/NBr103/WBD_detail_SW_NW_cleaned_NBr103.gpkg'
Pa_SW_Cond_A        =   M.Pa.WS / r"models\NBr\In\RIV\RIV_Cond_DETAILWATERGANGEN_NBr1.IDF"
Pa_SW_Cond_B        =   M.Pa.WS / r"models\NBr\In\RIV\RIV_Cond_DRN_NBr1.IDF"
Pa_SFR_OBS_In       =   M.Pa.In / 'OBS/SFR/NBr73/NBr73_SFR_OBS_Pnt.csv'
SFR_connect_Pkgs    =   ('DRN', 'RIV')
Pa_Shp_catchment    =   M.Pa.WS / r'models\NBr\PoP\common\Pgn\Chaamse_beek\catchment_chaamsebeek_ulvenhout.shp'
SFR_OBS_all         =   ['stage'] # ['sfr', 'downstream-flow', 'inflow', 'stage', 'from-mvr']
SFR_options         =   [f'OBS6 FILEIN {M.Pa.Sim_In / (MdlN + ".SFR6.obs")}',
                        f'BUDGET FILEOUT {MdlN}.SFR6.cbc', # 666 Remove this if it doesn't contain any useful info
                        # 'AUXILIARY line_id',
                        # f'STAGE FILEOUT SFR_Stg_{MdlN}.bin', # unnecessary because we have OBS for all reaches
                        f'PACKAGE_CONVERGENCE FILEOUT SFR_convergence_{MdlN}.CSV']
SFR_one_reach_per_cell: bool = True
# Pa_SFR_Stg_Init = M.Pa.In / f'SFR/Stg_Init/{MdlN}/Stg_Init_{MdlN}.csv'

# OBS Options
MdlN_HD_OBS   =   'NBr99'
Pa_HD_OBS_Src =   M.Pa.In / f'OBS/HD/{MdlN_HD_OBS}/GWHD_{MdlN_HD_OBS}.OBS6'
Pa_HD_OBS_Dst = M.Pa.Sim_In / f'GWHD_{MdlN}.OBS6'

# SFR PoP
l_SFR_Par_to_Rst = ['k', 'rwid', 'rgrd', 'rtp', 'rbth', 'rhk', 'man', 'ncon', 'cond']
Pa_p_SFR_In = M.Pa.PoP / f'In/SFR/{MdlN}/SFR_{MdlN}.gpkg' # Last file produced by that rule -> shows rule finshed 
MdlN_RIV_Vs = 'NBr101'
Pa_RIV_Stg_Vs_winter = Path(r'G:/models/NBr/In/RIV/NBr49/RIV_Stg_main_winter_NBr49.IDF')
Pa_RIV_Stg_Vs_summer = Path(r'G:/models/NBr/In/RIV/NBr49/RIV_Stg_main_summer_NBr49.IDF')
    # start_year
PoP_end_year = 2001
l_Diff_PoP_Par = ['SFR/Stg', 'SFR/from-mvr', 'SFR/downstream-flow', 'SFR/gwf', 'GW_HD_AVGs/L1']

## Completion validation. If you want to re-run a rule, delete the coresponding temp file.
Pa_temp             =   M.Pa.Smk.parent / 'temp'
log_Init            =   Pa_temp / f"Log_init_{MdlN}"
log_fix_MSW_area    =   Pa_temp / f"Log_fix_MSW_area_{MdlN}"
log_add_SFR_OBS     =   Pa_temp / f"Log_add_SFR_OBS_all_reaches_{MdlN}"
log_Sim             =   Pa_temp / f"Log_Sim_{MdlN}"
log_PRJ_to_TIF      =   Pa_temp / f"Log_PRJ_to_TIF_{MdlN}"
log_HD_AVGs         =   Pa_temp / f"Log_HD_AVGs_{MdlN}"
log_SFR_CBC         =   Pa_temp / f"Log_SFR_CBC_{MdlN}"
log_SFR_Stg         =   Pa_temp / f"Log_SFR_Stg_{MdlN}"
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
        log_upload

## -- PrP --
rule log_Init: # Sets status to running, and writes other info about therun. Has to complete before anything else.
    output:
        touch(temp(log_Init))
    run:
        import socket
        device = socket.gethostname()
        Up_log(MdlN, {  'End Status':       'Running',
                        'PrP start DT':     DT.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'Sim device name':  device,
                        'Sim Dir':          M.Pa.MdlN,
                        '1st SP date':      DT.strptime(M.INI.SDATE, "%Y%m%d").strftime("%Y-%m-%d"),
                        'last SP date':     DT.strptime(M.INI.EDATE, "%Y%m%d").strftime("%Y-%m-%d")})
        
rule Mdl_Prep: # Prepares Sim Ins (from Ins) via BAT file.
    input:
        log_Init,
        BAT = M.Pa.BAT,
        INI = M.Pa.INI,
        PRJ = M.Pa.PRJ
    output:
        M.Pa.NAM_Sim
    run:
        from imod.mf6.ims import SolutionPresetComplex
        M.IMS_settings = SolutionPresetComplex(['imported_model']) # Set the IMS settings to a simple preset.
        from WS_Mdl.imod.sfr.prsimp import SFR_settings
        from WS_Mdl.imod.prep import Sim
        SFR_Cfg = SFR_settings( Pa_Cond_A           = Pa_SW_Cond_A,
                                Pa_Cond_B           = Pa_SW_Cond_B,
                                Pa_Gpkg             = Pa_SFR_GPkg,
                                Pa_OBS_In           = Pa_SFR_OBS_In,
                                connect_Pkgs        = SFR_connect_Pkgs,
                                Pa_Shp_connect_Pkgs = Pa_Shp_catchment,
                                OBS_all             = SFR_OBS_all,
                                options             = SFR_options,
                                one_reach_per_cell  = SFR_one_reach_per_cell,
                                # Stg_Init            = Pa_SFR_Stg_Init
                                )
        Sim(M, SFR=SFR_Cfg)

## -- PrSimP --
rule add_HD_OBS_copy: # Copying so I can manually make a file that contains OBS for both OBS Pnts and whole layers.
    input:
        M.Pa.NAM_Sim
    output:
        Pa_HD_OBS_Dst
    run:
        from WS_Mdl.imod.mf6.nam import add_Pkg
        sh.copy2(Pa_HD_OBS_Src, Pa_HD_OBS_Dst) # Copy the file to create a new one with the same content.
        add_Pkg(MdlN, fr'  OBS6 .\imported_model\GWHD_{MdlN}.OBS6 GWHD_OBS')  # Add to NAM

rule add_SFR_OBS_all_reaches: # Add OBS for all reaches to the NAM file. This should evntually be moved inside Sim
    input:
        M.Pa.NAM_Sim
    output:
        touch(log_add_SFR_OBS)
    run:
        from WS_Mdl.imod.sfr.info import SFR_PkgD_to_DF
        import pandas as pd
        DF_SFR_PkgD = SFR_PkgD_to_DF(MdlN) # Load PkgD DF 

        for Par in SFR_OBS_all:
            DF_SFR_PkgD_w = pd.DataFrame({  'obsname': DF_SFR_PkgD.apply(lambda x: f"L{int(x['k'])}_R{int(x['i'])}_C{int(x['j'])}", axis=1),
                                            'obstype': Par, 
                                            'rno': DF_SFR_PkgD['rno']}) # Prepare for writing to OBS file's CONTINUOUTS FILOUT BLOCK

            with open(list(M.Pa.Sim_In.glob('*sfr*obs'))[0], 'a') as f: # Append to the OBS file (assumes 1)
                f.write(f"\nBEGIN CONTINUOUS FILEOUT ../Stg_{MdlN}.SFR6.bin BINARY")
                f.write('\n# --- Added by Snakemake ---\n')
                f.write(DF_SFR_PkgD_w.ws.to_MF_block())
                f.write('END CONTINUOUS\n')
                
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
        log_fix_MSW_area,
        log_add_SFR_OBS
    output:
        touch(log_Sim)
    run:
        os.chdir(M.Pa.MdlN) # Change directory to the model folder.
        DT_Sim_Start = DT.now()
        Up_log(MdlN, {  'Sim start DT'  :   DT_Sim_Start.strftime("%Y-%m-%d %H:%M:%S")})
        sp.run([M.Pa.coupler_Exe, M.Pa.TOML], shell=True, check=True) 
        Up_log(MdlN, {  'Sim end DT'    :   DT.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'Sim Dur'       :   get_elapsed_time_str(DT_Sim_Start),
                        'End Status'    :   'Completed'})

# -- PoP ---
rule PRJ_to_TIF:
    input:
        log_Sim
    output:
        touch(temp(log_PRJ_to_TIF))
    run:
        from WS_Mdl.imod.prj import to_TIF as PRJ_to_TIF
        PRJ_to_TIF(MdlN, iMOD5=iMOD5) # Convert PRJ to TIFs
        Up_log(MdlN, {  'PRJ_to_TIF':   1})

rule p_SFR_In:
    input:
        log_Sim
    output:
        Pa_p_SFR_In
    run:
        from WS_Mdl.imod.sfr.export import Par_to_Rst, SFR_to_GPkg
        Par_to_Rst(MdlN, l_SFR_Par_to_Rst)
        SFR_to_GPkg(MdlN)

rule p_SFR_CBC:
    input:
        Pa_p_SFR_In
    output:
        touch(log_SFR_CBC)
    run:
        from WS_Mdl.imod.pop.sfr import SFR_CBC_Par_to_TIF
        SFR_CBC_Par_to_TIF(MdlN)


rule p_SFR_Stg:
    input:
        Pa_p_SFR_In
    output:
        touch(log_SFR_Stg)
    run:
        # calcyulate Stg_AVGs & depth AVGs
        from WS_Mdl.imod.pop.sfr import c_Stg_AVGs
        c_Stg_AVGs(MdlN, end_year=PoP_end_year)

        # calculate Diff Vs RIV
        import imod
        from WS_Mdl.xr.spatial import clip_Mdl_area

        ## Load RIV
        RIV_summer = clip_Mdl_area(imod.idf.open(Pa_RIV_Stg_Vs_summer), MdlN)
        RIV_winter = clip_Mdl_area(imod.idf.open(Pa_RIV_Stg_Vs_winter), MdlN)
        RIV_AVG = (RIV_summer + RIV_winter) / 2

        ## Load SFR
        import rioxarray as rxr
        SFR_summer = rxr.open_rasterio(M.Pa.PoP_Out_MdlN / f'SFR/Stg/SFR_Stg_summer_AVG_{MdlN}.tif', masked=True)
        SFR_winter = rxr.open_rasterio(M.Pa.PoP_Out_MdlN / f'SFR/Stg/SFR_Stg_winter_AVG_{MdlN}.tif', masked=True)
        SFR_AVG = rxr.open_rasterio(M.Pa.PoP_Out_MdlN / f'SFR/Stg/SFR_Stg_AVG_{MdlN}.tif', masked=True)

        ## Diff & Save
        Pa_Diff = M.Pa.PoP_Out_MdlN / f'SFR/Stg/Diff'
        Pa_Diff.mkdir(parents=True, exist_ok=True)
        (SFR_summer - RIV_summer).rio.to_raster(Pa_Diff / f'SFR_Stg_summer_AVG_{MdlN}_m_{Pa_RIV_Stg_Vs_summer.stem}.tif')
        (SFR_winter - RIV_winter).rio.to_raster(Pa_Diff / f'SFR_Stg_winter_AVG_{MdlN}_m_{Pa_RIV_Stg_Vs_winter.stem}.tif')
        (SFR_AVG - RIV_AVG).rio.to_raster(Pa_Diff / f'SFR_Stg_AVG_{MdlN}_m_RIV_Stg_AVG_{Pa_RIV_Stg_Vs_winter.stem.split('_')[-1]}.tif')

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

rule Diff_PoP_Par:
    input:
        log_SFR_Stg,
        log_SFR_CBC,
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
        log_HD_AVGs,
        log_PRJ_to_TIF,
        log_Diff,
        M.Pa.PoP_Out_MdlN / f'GW_HD_OBS/metadata.txt'
    output:
        M.Pa.MM
    run:
        from WS_Mdl.io.qgis import update_MM
        update_MM(MdlN, MdlN_MM_B=MdlN_MM_B)      # Update MM 
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

rule Upl_MdlN_PoP_Out: # Uploads the PoP Out files to iBridges. This is the final step of the workflow.
    input:
        M.Pa.MM,
        log_WB
    output:
        touch(log_upload)
    run:
        from WS_Mdl.io.ibridges import Upl_MdlN_PoP_Out
        Upl_MdlN_PoP_Out(MdlN)