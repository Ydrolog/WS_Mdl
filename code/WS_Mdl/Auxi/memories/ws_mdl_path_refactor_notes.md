- `Mdl_N(MdlN).Pa` is the preferred path-access pattern in WS_Mdl.
- Path objects are used heavily; avoid string-only ops on paths (`.split`, `.lower`, `os.path.exists`) and prefer `.name`, `.suffix`, `.exists()`, `.parent.mkdir()`.
- For legacy functions with explicit `iMOD5` argument, preserve behavior via `MdlN_PaView(MdlN, iMOD5=...)` only when it differs from model autodetection.
- `Mdl_N(..., iMOD5=True|False)` can now override version autodetection; `M.Pa` uses this override to pass `iMOD5` into `MdlN_PaView` and produce stable path structures.
- Legacy code still accesses `MdlN_PaView` via `Pa.*`/`Pa[...]` keys like `Pa_Mdl`; keep backward-compat key resolution (`Pa_<key>` -> `<key>`) to prevent `AttributeError` in workflows like `imod.prj.to_TIF`.
- In `WS_Mdl/imod/prj.py`, avoid `re.sub(...).name` because `re.sub` returns `str`; use `Path(value).with_suffix(...)` before `.name` to avoid `'str' object has no attribute 'name'`.
- `Mdl_N.Pa_B` should return baseline-only paths (`MdlN_PaView(self.B, ...)`) rather than `MdlN_PaView(self.MdlN).B(self.B)`, which mixes current keys with `_B` keys and can resolve wrong paths in callers.



- In `WS_Mdl/io/qgis.py` (`update_MM`), `Path.rglob()` must be called with a pattern (e.g. `rglob('*')`); when replacing `os.walk`, iterate paths and filter `is_file()` before writing to zip.
