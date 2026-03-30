- RunMng/Snakemake failures in `log_Init` and `freeze_pixi_env` can come from `WS_Mdl/core/log.py:update_log` using `import filelock as FL` then `FL(...)`; `filelock` is a module, so use `from filelock import FileLock as FL` or `filelock.FileLock(...)`.
- The correct class name is `FileLock` (capital L). `Filelock` will raise an import error.

- `send2trash` failures with `[WinError -2144927711] OLE error 0x80270021` mean `COPYENGINE_E_ACCESS_DENIED_SRC` (Shell source access denied); often file lock/access issue while recycling `Sim` or `PoP/Out` folders.

- Mitigation added in `WS_Mdl/io/sim.py` and `WS_Mdl/utils.py`: recycle-bin moves now retry (`_send2trash_with_retries`) and print lock-specific hints when Windows reports lock/access-denied shell errors.
