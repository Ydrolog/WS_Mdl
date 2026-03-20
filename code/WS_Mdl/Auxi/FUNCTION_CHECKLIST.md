# WS_Mdl Function Checklist
x: I checked it
y: I checked it, but it'll have to be editted later/after the check
z: the function can be improved in the future. (I also leave #666 next to those functions in the .py files)
-: Skip for now

Generated: 2026-03-11 10:50 UTC
Package root: G:/code/WS_Mdl

How to use:
- Tick items with [x] when checked/tested.
- Re-run this script anytime; existing checks are preserved by item ID.

## __init__.py
- [ ] FILE __init__.py <!-- id:file:__init__.py -->
  - [ ] FUNCTION __getattr__ (L28) <!-- id:func:__init__.py:__getattr__ -->

## __main__.py
- [ ] FILE __main__.py <!-- id:file:__main__.py -->
  - [ ] FUNCTION _module_name_from_path (L30) <!-- id:func:__main__.py:_module_name_from_path -->
  - [ ] FUNCTION _extract_all_names (L51) <!-- id:func:__main__.py:_extract_all_names -->
  - [ ] FUNCTION _discover_exports (L101) <!-- id:func:__main__.py:_discover_exports -->
  - [ ] FUNCTION _print_usage (L144) <!-- id:func:__main__.py:_print_usage -->
  - [ ] FUNCTION main (L150) <!-- id:func:__main__.py:main -->

## core/__init__.py
- [y] FILE core/__init__.py <!-- id:file:core/__init__.py -->
  - [x] FUNCTION __getattr__ (L60) <!-- id:func:core/__init__.py:__getattr__ -->
  - [x] FUNCTION __dir__ (L76) <!-- id:func:core/__init__.py:__dir__ -->

## core/df.py
- [ ] FILE core/df.py <!-- id:file:core/df.py -->
  - [ ] CLASS DFAccessor (L9) <!-- id:class:core/df.py:DFAccessor -->
    - [x] METHOD DFAccessor.__init__ (L19) <!-- id:method:core/df.py:DFAccessor.__init__ -->
    - [x] METHOD DFAccessor.info (L22) <!-- id:method:core/df.py:DFAccessor.info -->
    - [x] METHOD DFAccessor.memory (L30) <!-- id:method:core/df.py:DFAccessor.memory -->
    - [x] METHOD DFAccessor.Col_value_counts_grouped (L39) <!-- id:method:core/df.py:DFAccessor.Col_value_counts_grouped -->
    - [x] METHOD DFAccessor.round_Cols (L72) <!-- id:method:core/df.py:DFAccessor.round_Cols -->
    - [ ] METHOD DFAccessor.Clip_Mdl_area (L96) <!-- id:method:core/df.py:DFAccessor.Clip_Mdl_area -->
    - [ ] METHOD DFAccessor.Calc_XY (L136) <!-- id:method:core/df.py:DFAccessor.Calc_XY -->
    - [ ] METHOD DFAccessor.Calc_XY_start_end_from_Geom (L159) <!-- id:method:core/df.py:DFAccessor.Calc_XY_start_end_from_Geom -->
    - [ ] METHOD DFAccessor.to_MF_block (L175) <!-- id:method:core/df.py:DFAccessor.to_MF_block -->

## core/log.py
- [ ] FILE core/log.py <!-- id:file:core/log.py -->
  - [x] FUNCTION DF_match_MdlN (L10) <!-- id:func:core/log.py:DF_match_MdlN -->
  - [x] FUNCTION to_Se (L18) <!-- id:func:core/log.py:to_Se -->
  - [z] FUNCTION last_MdlN (L32) <!-- id:func:core/log.py:last_MdlN -->
  - [x] FUNCTION r_RunLog (L48) <!-- id:func:core/log.py:r_RunLog -->
  - [ ] FUNCTION Up_log (L52) <!-- id:func:core/log.py:Up_log -->
  - [x] FUNCTION get_B (L74) <!-- id:func:core/log.py:get_B -->

## core/mdl.py
- [ ] FILE core/mdl.py <!-- id:file:core/mdl.py -->
  - [y] CLASS Mdl_N (L9) <!-- id:class:core/mdl.py:Mdl_N -->
    - [x] METHOD Mdl_N.__post_init__ (L27) <!-- id:method:core/mdl.py:Mdl_N.__post_init__ -->
    - [x] METHOD Mdl_N.Pa (L47) <!-- id:method:core/mdl.py:Mdl_N.Pa -->
    - [x] METHOD Mdl_N.INI (L59) <!-- id:method:core/mdl.py:Mdl_N.INI -->
    - [x] METHOD Mdl_N.Dmns (L71) <!-- id:method:core/mdl.py:Mdl_N.Dmns -->
    - [x] METHOD Mdl_N.V (L78) <!-- id:method:core/mdl.py:Mdl_N.V -->
    - [x] METHOD Mdl_N.B (L95) <!-- id:method:core/mdl.py:Mdl_N.B -->
    - [x] METHOD Mdl_N.Pa_B (L102) <!-- id:method:core/mdl.py:Mdl_N.Pa_B -->

## core/metrics.py
- [-] FILE core/metrics.py <!-- id:file:core/metrics.py -->
  - [ ] CLASS Vld_Mtc (L5) <!-- id:class:core/metrics.py:Vld_Mtc -->
    - [ ] METHOD Vld_Mtc.__init__ (L23) <!-- id:method:core/metrics.py:Vld_Mtc.__init__ -->
    - [ ] METHOD Vld_Mtc.compute (L28) <!-- id:method:core/metrics.py:Vld_Mtc.compute -->

## core/path.py
- [ ] FILE core/path.py <!-- id:file:core/path.py -->
  - [x] FUNCTION get_Mdl (L24) <!-- id:func:core/path.py:get_Mdl -->
  - [x] FUNCTION imod_V (L29) <!-- id:func:core/path.py:imod_V -->
  - [x] FUNCTION MdlN_Pa (L55) <!-- id:func:core/path.py:MdlN_Pa -->
  - [ ] CLASS MdlN_PaView (L157) <!-- id:class:core/path.py:MdlN_PaView -->
    - [x] METHOD MdlN_PaView.__init__ (L162) <!-- id:method:core/path.py:MdlN_PaView.__init__ -->
    - [x] METHOD MdlN_PaView.B (L165) <!-- id:method:core/path.py:MdlN_PaView.B -->
    - [] METHOD MdlN_PaView._resolve_key (L170) <!-- id:method:core/path.py:MdlN_PaView._resolve_key -->
    - [ ] METHOD MdlN_PaView.__getattr__ (L180) <!-- id:method:core/path.py:MdlN_PaView.__getattr__ -->
    - [ ] METHOD MdlN_PaView.__getitem__ (L187) <!-- id:method:core/path.py:MdlN_PaView.__getitem__ -->
    - [ ] METHOD MdlN_PaView.__repr__ (L190) <!-- id:method:core/path.py:MdlN_PaView.__repr__ -->
    - [x] METHOD MdlN_PaView.as_dict (L196) <!-- id:method:core/path.py:MdlN_PaView.as_dict -->
    - [x] METHOD MdlN_PaView.get (L200) <!-- id:method:core/path.py:MdlN_PaView.get -->
    - [x] METHOD MdlN_PaView.keys (L206) <!-- id:method:core/path.py:MdlN_PaView.keys -->
    - [x] METHOD MdlN_PaView.items (L209) <!-- id:method:core/path.py:MdlN_PaView.items -->
    - [x] METHOD MdlN_PaView.values (L212) <!-- id:method:core/path.py:MdlN_PaView.values -->
    - [] METHOD MdlN_PaView.__iter__ (L215) <!-- id:method:core/path.py:MdlN_PaView.__iter__ -->
    - [] METHOD MdlN_PaView.__len__ (L218) <!-- id:method:core/path.py:MdlN_PaView.__len__ -->
    - [] METHOD MdlN_PaView.__contains__ (L221) <!-- id:method:core/path.py:MdlN_PaView.__contains__ -->

## core/runtime.py
- [x] FILE core/runtime.py <!-- id:file:core/runtime.py -->
  - [x] FUNCTION timed_import (L5) <!-- id:func:core/runtime.py:timed_import -->
  - [x] FUNCTION timed_Exe (L18) <!-- id:func:core/runtime.py:timed_Exe -->

## core/spatial.py
- [ ] FILE core/spatial.py <!-- id:file:core/spatial.py -->
  - [ ] FUNCTION c_Dist (L4) <!-- id:func:core/spatial.py:c_Dist -->

## core/style.py
- [ ] FILE core/style.py <!-- id:file:core/style.py -->
  - [x] FUNCTION set_verbose (L29) <!-- id:func:core/style.py:set_verbose -->
  - [x] FUNCTION _is_Mdl_N (L35) <!-- id:func:core/style.py:_is_Mdl_N -->
  - [x] FUNCTION _fmt_arg (L45) <!-- id:func:core/style.py:_fmt_arg -->
  - [x] FUNCTION sprint (L73) <!-- id:func:core/style.py:sprint -->
  - [ ] FUNCTION sinput (L97) <!-- id:func:core/style.py:sinput -->

## core/text.py
- [x] FILE core/text.py <!-- id:file:core/text.py -->
  - [x] FUNCTION r_Txt_Lns (L4) <!-- id:func:core/text.py:r_Txt_Lns -->

## imod/idf.py
- [ ] FILE imod/idf.py <!-- id:file:imod/idf.py -->
  - [ ] FUNCTION HD_Out_to_DF (L19) <!-- id:func:imod/idf.py:HD_Out_to_DF -->
  - [ ] FUNCTION stack_to_DF (L75) <!-- id:func:imod/idf.py:stack_to_DF -->
  - [ ] FUNCTION to_TIF (L90) <!-- id:func:imod/idf.py:to_TIF -->
  - [ ] FUNCTION to_MBTIF (L148) <!-- id:func:imod/idf.py:to_MBTIF -->

## imod/ini.py
- [ ] FILE imod/ini.py <!-- id:file:imod/ini.py -->
  - [ ] FUNCTION as_d (L7) <!-- id:func:imod/ini.py:as_d -->
  - [ ] FUNCTION Mdl_Dmns (L59) <!-- id:func:imod/ini.py:Mdl_Dmns -->
  - [ ] FUNCTION CeCes (L70) <!-- id:func:imod/ini.py:CeCes -->
  - [ ] CLASS INIView (L28) <!-- id:class:imod/ini.py:INIView -->
    - [ ] METHOD INIView.__init__ (L33) <!-- id:method:imod/ini.py:INIView.__init__ -->
    - [ ] METHOD INIView.__getattr__ (L37) <!-- id:method:imod/ini.py:INIView.__getattr__ -->
    - [ ] METHOD INIView.__getitem__ (L43) <!-- id:method:imod/ini.py:INIView.__getitem__ -->
    - [ ] METHOD INIView.get (L48) <!-- id:method:imod/ini.py:INIView.get -->
    - [ ] METHOD INIView.__contains__ (L53) <!-- id:method:imod/ini.py:INIView.__contains__ -->

## imod/ipf.py
- [ ] FILE imod/ipf.py <!-- id:file:imod/ipf.py -->
  - [ ] FUNCTION as_DF (L9) <!-- id:func:imod/ipf.py:as_DF -->

## imod/mf6/nam.py
- [ ] FILE imod/mf6/nam.py <!-- id:file:imod/mf6/nam.py -->
  - [ ] FUNCTION add_PKG (L6) <!-- id:func:imod/mf6/nam.py:add_PKG -->

## imod/mf6/obs.py
- [ ] FILE imod/mf6/obs.py <!-- id:file:imod/mf6/obs.py -->
  - [ ] FUNCTION add (L11) <!-- id:func:imod/mf6/obs.py:add -->

## imod/mf6/read.py
- [ ] FILE imod/mf6/read.py <!-- id:file:imod/mf6/read.py -->
  - [ ] FUNCTION MF6_block_to_DF (L8) <!-- id:func:imod/mf6/read.py:MF6_block_to_DF -->

## imod/mf6/shd.py
- [ ] FILE imod/mf6/shd.py <!-- id:file:imod/mf6/shd.py -->
  - [ ] FUNCTION from_HD_Out (L11) <!-- id:func:imod/mf6/shd.py:from_HD_Out -->

## imod/mf6/solution.py
- [ ] FILE imod/mf6/solution.py <!-- id:file:imod/mf6/solution.py -->
  - [ ] FUNCTION moderate_settings (L4) <!-- id:func:imod/mf6/solution.py:moderate_settings -->

## imod/mf6/write.py
- [ ] FILE imod/mf6/write.py <!-- id:file:imod/mf6/write.py -->
  - [ ] FUNCTION add_MVR_to_OPTIONS (L9) <!-- id:func:imod/mf6/write.py:add_MVR_to_OPTIONS -->
  - [ ] FUNCTION add_OBS_to_MF_In (L34) <!-- id:func:imod/mf6/write.py:add_OBS_to_MF_In -->

## imod/msw/mete_grid.py
- [ ] FILE imod/msw/mete_grid.py <!-- id:file:imod/msw/mete_grid.py -->
  - [ ] FUNCTION to_DF (L7) <!-- id:func:imod/msw/mete_grid.py:to_DF -->
  - [ ] FUNCTION add_missing_Cols (L15) <!-- id:func:imod/msw/mete_grid.py:add_missing_Cols -->
  - [ ] FUNCTION Cvt_to_AbsPa (L32) <!-- id:func:imod/msw/mete_grid.py:Cvt_to_AbsPa -->

## imod/msw/read.py
- [ ] FILE imod/msw/read.py <!-- id:file:imod/msw/read.py -->
  - [ ] FUNCTION MSW_In_to_DF (L9) <!-- id:func:imod/msw/read.py:MSW_In_to_DF -->

## imod/pop/gxg.py
- [ ] FILE imod/pop/gxg.py <!-- id:file:imod/pop/gxg.py -->
  - [ ] FUNCTION HD_Bin_GXG_to_MBTIF (L19) <!-- id:func:imod/pop/gxg.py:HD_Bin_GXG_to_MBTIF -->
  - [ ] FUNCTION GXG_Diff (L96) <!-- id:func:imod/pop/gxg.py:GXG_Diff -->
  - [ ] FUNCTION HD_IDF_GXG_to_TIF (L111) <!-- id:func:imod/pop/gxg.py:HD_IDF_GXG_to_TIF -->

## imod/pop/hd.py
- [ ] FILE imod/pop/hd.py <!-- id:file:imod/pop/hd.py -->
  - [ ] FUNCTION HD_IDF_Agg_to_TIF (L14) <!-- id:func:imod/pop/hd.py:HD_IDF_Agg_to_TIF -->
  - [ ] FUNCTION HD_Agg_name (L136) <!-- id:func:imod/pop/hd.py:HD_Agg_name -->

## imod/pop/sfr.py
- [ ] FILE imod/pop/sfr.py <!-- id:file:imod/pop/sfr.py -->
  - [ ] FUNCTION stage_TS (L21) <!-- id:func:imod/pop/sfr.py:stage_TS -->

## imod/pop/text.py
- [ ] FILE imod/pop/text.py <!-- id:file:imod/pop/text.py -->
  - [ ] FUNCTION Agg_OBS (L9) <!-- id:func:imod/pop/text.py:Agg_OBS -->

## imod/pop/wb.py
- [ ] FILE imod/pop/wb.py <!-- id:file:imod/pop/wb.py -->
  - [ ] FUNCTION Diff_to_xlsx (L10) <!-- id:func:imod/pop/wb.py:Diff_to_xlsx -->

## imod/prep.py
- [ ] FILE imod/prep.py <!-- id:file:imod/prep.py -->
  - [ ] FUNCTION Mdl_Prep (L14) <!-- id:func:imod/prep.py:Mdl_Prep -->

## imod/prj.py
- [ ] FILE imod/prj.py <!-- id:file:imod/prj.py -->
  - [ ] FUNCTION r_with_OBS (L21) <!-- id:func:imod/prj.py:r_with_OBS -->
  - [ ] FUNCTION to_DF (L110) <!-- id:func:imod/prj.py:to_DF -->
  - [ ] FUNCTION o_with_OBS (L197) <!-- id:func:imod/prj.py:o_with_OBS -->
  - [ ] FUNCTION regrid (L234) <!-- id:func:imod/prj.py:regrid -->
  - [ ] FUNCTION regrid_DA (L277) <!-- id:func:imod/prj.py:regrid_DA -->
  - [ ] FUNCTION to_TIF (L341) <!-- id:func:imod/prj.py:to_TIF -->

## imod/sfr/export.py
- [ ] FILE imod/sfr/export.py <!-- id:file:imod/sfr/export.py -->
  - [ ] FUNCTION Par_to_Rst (L18) <!-- id:func:imod/sfr/export.py:Par_to_Rst -->
  - [ ] FUNCTION SFR_to_GPkg (L72) <!-- id:func:imod/sfr/export.py:SFR_to_GPkg -->

## imod/sfr/info.py
- [ ] FILE imod/sfr/info.py <!-- id:file:imod/sfr/info.py -->
  - [ ] FUNCTION SFR_PkgD_to_DF (L7) <!-- id:func:imod/sfr/info.py:SFR_PkgD_to_DF -->
  - [ ] FUNCTION SFR_ConnD_to_DF (L56) <!-- id:func:imod/sfr/info.py:SFR_ConnD_to_DF -->
  - [ ] FUNCTION reach_to_cell_id (L82) <!-- id:func:imod/sfr/info.py:reach_to_cell_id -->
  - [ ] FUNCTION reach_to_XY (L99) <!-- id:func:imod/sfr/info.py:reach_to_XY -->
  - [ ] FUNCTION get_SFR_OBS_Out_Pas (L115) <!-- id:func:imod/sfr/info.py:get_SFR_OBS_Out_Pas -->

## imod/xr.py
- [ ] FILE imod/xr.py <!-- id:file:imod/xr.py -->
  - [ ] FUNCTION clip_Mdl_Aa (L7) <!-- id:func:imod/xr.py:clip_Mdl_Aa -->

## io/convert.py
- [ ] FILE io/convert.py <!-- id:file:io/convert.py -->
  - [x] FUNCTION Bin_to_text (L9) <!-- id:func:io/convert.py:Bin_to_text -->
  - [ ] FUNCTION Vtr_to_TIF (L118) <!-- id:func:io/convert.py:Vtr_to_TIF -->

## io/ibridges.py
- [ ] FILE io/ibridges.py <!-- id:file:io/ibridges.py -->
  - [ ] FUNCTION l_Fis_Exc (L13) <!-- id:func:io/ibridges.py:l_Fis_Exc -->
  - [ ] FUNCTION Pw (L36) <!-- id:func:io/ibridges.py:Pw -->
  - [ ] FUNCTION Upl (L61) <!-- id:func:io/ibridges.py:Upl -->
  - [ ] FUNCTION Dl (L100) <!-- id:func:io/ibridges.py:Dl -->
  - [ ] CLASS iB_session (L42) <!-- id:class:io/ibridges.py:iB_session -->
    - [ ] METHOD iB_session.__init__ (L43) <!-- id:method:io/ibridges.py:iB_session.__init__ -->
    - [ ] METHOD iB_session.info (L49) <!-- id:method:io/ibridges.py:iB_session.info -->

## io/qgis.py
- [ ] FILE io/qgis.py <!-- id:file:io/qgis.py -->
  - [ ] FUNCTION update_MM (L15) <!-- id:func:io/qgis.py:update_MM -->

## io/sim.py
- [ ] FILE io/sim.py <!-- id:file:io/sim.py -->
  - [ ] FUNCTION S_from_B (L18) <!-- id:func:io/sim.py:S_from_B -->
  - [ ] FUNCTION S_from_B_undo (L71) <!-- id:func:io/sim.py:S_from_B_undo -->
  - [ ] FUNCTION RunSim (L92) <!-- id:func:io/sim.py:RunSim -->
  - [ ] FUNCTION RunMng (L139) <!-- id:func:io/sim.py:RunMng -->
  - [ ] FUNCTION reset_Sim (L204) <!-- id:func:io/sim.py:reset_Sim -->
  - [ ] FUNCTION remove_Sim_Out (L316) <!-- id:func:io/sim.py:remove_Sim_Out -->
  - [ ] FUNCTION rerun_Sim (L450) <!-- id:func:io/sim.py:rerun_Sim -->
  - [ ] FUNCTION get_elapsed_time_str (L498) <!-- id:func:io/sim.py:get_elapsed_time_str -->
  - [ ] FUNCTION run_cmd (L510) <!-- id:func:io/sim.py:run_cmd -->
  - [ ] FUNCTION freeze_pixi_env (L514) <!-- id:func:io/sim.py:freeze_pixi_env -->

## io/text.py
- [ ] FILE io/text.py <!-- id:file:io/text.py -->
  - [ ] FUNCTION o_ (L7) <!-- id:func:io/text.py:o_ -->
  - [ ] FUNCTION o_VS (L27) <!-- id:func:io/text.py:o_VS -->
  - [ ] FUNCTION Sim_Cfg (L47) <!-- id:func:io/text.py:Sim_Cfg -->
  - [ ] FUNCTION o_LSTs (L62) <!-- id:func:io/text.py:o_LSTs -->
  - [ ] FUNCTION o_NAMs (L79) <!-- id:func:io/text.py:o_NAMs -->

## scripts/*
- [x] FILE scripts/Bin_to_text.py <!-- id:file:scripts/Bin_to_text.py -->
- [x] FILE scripts/Dir_Fo_size.py <!-- id:file:scripts/Dir_Fo_size.py -->
- [ ] FILE scripts/DVC_add_pattern.py <!-- id:file:scripts/DVC_add_pattern.py -->
- [ ] FILE scripts/DVC_add_pattern_deep.py <!-- id:file:scripts/DVC_add_pattern_deep.py -->
- [x] FILE scripts/gen_function_checklist.py <!-- id:file:scripts/gen_function_checklist.py -->
- [ ] FILE scripts/IDF_to_TIF.py <!-- id:file:scripts/IDF_to_TIF.py -->
- [ ] FILE scripts/map_DVC.py <!-- id:file:scripts/map_DVC.py -->
- [ ] FILE scripts/map_gitignore.py <!-- id:file:scripts/map_gitignore.py -->
- [x] FILE scripts/o_.py <!-- id:file:scripts/o_.py -->
- [x] FILE scripts/o_LST.py <!-- id:file:scripts/o_LST.py -->
- [x] FILE scripts/o_LSTs.py <!-- id:file:scripts/o_LSTs.py -->
- [x] FILE scripts/o_NAMs.py <!-- id:file:scripts/o_NAMs.py -->
- [x] FILE scripts/o_VS.py <!-- id:file:scripts/o_VS.py -->

## scripts/p_TS_range.py
- [ ] FILE scripts/p_TS_range.py <!-- id:file:scripts/p_TS_range.py -->
  - [ ] FUNCTION main (L11) <!-- id:func:scripts/p_TS_range.py:main -->

## scripts/remove_Sim_Out.py
- [ ] FILE scripts/remove_Sim_Out.py <!-- id:file:scripts/remove_Sim_Out.py -->
  - [ ] FUNCTION main (L7) <!-- id:func:scripts/remove_Sim_Out.py:main -->

## scripts/rerun_Sim.py
- [ ] FILE scripts/rerun_Sim.py <!-- id:file:scripts/rerun_Sim.py -->
  - [ ] FUNCTION main (L6) <!-- id:func:scripts/rerun_Sim.py:main -->

## scripts/reset_Sim.py
- [ ] FILE scripts/reset_Sim.py <!-- id:file:scripts/reset_Sim.py -->
  - [ ] FUNCTION main (L6) <!-- id:func:scripts/reset_Sim.py:main -->

## scripts/RunMng.py
- [ ] FILE scripts/RunMng.py <!-- id:file:scripts/RunMng.py -->
  - [ ] FUNCTION main (L6) <!-- id:func:scripts/RunMng.py:main -->

## scripts/S_from_B.py
- [ ] FILE scripts/S_from_B.py <!-- id:file:scripts/S_from_B.py -->
  - [ ] FUNCTION main (L7) <!-- id:func:scripts/S_from_B.py:main -->

## scripts/S_from_B_undo.py
- [ ] FILE scripts/S_from_B_undo.py <!-- id:file:scripts/S_from_B_undo.py -->
  - [ ] FUNCTION main (L6) <!-- id:func:scripts/S_from_B_undo.py:main -->

## scripts/SFR_Par_to_Rst.py
- [ ] FILE scripts/SFR_Par_to_Rst.py <!-- id:file:scripts/SFR_Par_to_Rst.py -->
  - [ ] FUNCTION main (L5) <!-- id:func:scripts/SFR_Par_to_Rst.py:main -->

## scripts/SFR_stage_TS.py
- [ ] FILE scripts/SFR_stage_TS.py <!-- id:file:scripts/SFR_stage_TS.py -->
  - [ ] FUNCTION main (L7) <!-- id:func:scripts/SFR_stage_TS.py:main -->

## scripts/SFR_to_GPkg.py
- [ ] FILE scripts/SFR_to_GPkg.py <!-- id:file:scripts/SFR_to_GPkg.py -->
  - [ ] FUNCTION main (L7) <!-- id:func:scripts/SFR_to_GPkg.py:main -->

## scripts/Sim_Cfg.py
- [ ] FILE scripts/Sim_Cfg.py <!-- id:file:scripts/Sim_Cfg.py -->
  - [ ] FUNCTION main (L6) <!-- id:func:scripts/Sim_Cfg.py:main -->

## viz/ts.py
- [ ] FILE viz/ts.py <!-- id:file:viz/ts.py -->
  - [ ] FUNCTION SFR_reach_TS (L11) <!-- id:func:viz/ts.py:SFR_reach_TS -->
  - [ ] FUNCTION range (L133) <!-- id:func:viz/ts.py:range -->

## xr/compare.py
- [ ] FILE xr/compare.py <!-- id:file:xr/compare.py -->
  - [ ] FUNCTION Diff_MBTIF (L9) <!-- id:func:xr/compare.py:Diff_MBTIF -->

## xr/convert.py
- [ ] FILE xr/convert.py <!-- id:file:xr/convert.py -->
  - [ ] FUNCTION to_TIF (L10) <!-- id:func:xr/convert.py:to_TIF -->
  - [ ] FUNCTION to_MBTIF (L47) <!-- id:func:xr/convert.py:to_MBTIF -->

## xr/diagnostics.py
- [ ] FILE xr/diagnostics.py <!-- id:file:xr/diagnostics.py -->
  - [ ] FUNCTION _describe_da (L6) <!-- id:func:xr/diagnostics.py:_describe_da -->
  - [ ] FUNCTION _compare_dataarrays (L62) <!-- id:func:xr/diagnostics.py:_compare_dataarrays -->
  - [ ] CLASS DataArrayAccessor (L171) <!-- id:class:xr/diagnostics.py:DataArrayAccessor -->
    - [ ] METHOD DataArrayAccessor.__init__ (L174) <!-- id:method:xr/diagnostics.py:DataArrayAccessor.__init__ -->
    - [ ] METHOD DataArrayAccessor.describe (L177) <!-- id:method:xr/diagnostics.py:DataArrayAccessor.describe -->
    - [ ] METHOD DataArrayAccessor.compare (L181) <!-- id:method:xr/diagnostics.py:DataArrayAccessor.compare -->
  - [ ] CLASS DatasetAccessor (L205) <!-- id:class:xr/diagnostics.py:DatasetAccessor -->
    - [ ] METHOD DatasetAccessor.__init__ (L208) <!-- id:method:xr/diagnostics.py:DatasetAccessor.__init__ -->
    - [ ] METHOD DatasetAccessor.describe (L211) <!-- id:method:xr/diagnostics.py:DatasetAccessor.describe -->

## xr/spatial.py
- [ ] FILE xr/spatial.py <!-- id:file:xr/spatial.py -->
  - [ ] FUNCTION get_value (L1) <!-- id:func:xr/spatial.py:get_value -->

