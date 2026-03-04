"""Public export surface for ``WS_Mdl.core``.

Why this module uses ``_EXPORTS_TO_MODULES`` instead of eager imports:
- Eagerly importing every core submodule at package import time pulls in heavy
    dependency chains (for example DataFrame/numeric stacks) even when a caller
    needs only one lightweight symbol.
- Lazy resolution keeps startup fast and more robust for CLI usage where only
    a single function may be requested.
- The explicit mapping keeps the public API clear while still deferring actual
    imports until the symbol is first accessed.
"""

import importlib

# Map each exported symbol name to the module that owns the implementation.
#
# This is intentionally explicit (instead of scanning) so the public contract
# remains stable and easy to audit.
_EXPORTS_TO_MODULES = {
    # Style utilities
    'Sep': 'WS_Mdl.core.style',
    'Sep_2': 'WS_Mdl.core.style',
    'bold': 'WS_Mdl.core.style',
    'dim': 'WS_Mdl.core.style',
    'sprint': 'WS_Mdl.core.style',
    'style_reset': 'WS_Mdl.core.style',
    'warn': 'WS_Mdl.core.style',
    'CuCh': 'WS_Mdl.core.style',
    'set_verbose': 'WS_Mdl.core.style',
    # Path/model-location utilities
    'REPO_ROOT': 'WS_Mdl.core.path',
    'Pa_WS': 'WS_Mdl.core.path',
    'Pa_RunLog': 'WS_Mdl.core.path',
    'Pa_log_Out': 'WS_Mdl.core.path',
    'Pa_log_Cfg': 'WS_Mdl.core.path',
    'MdlN_Pa': 'WS_Mdl.core.path',
    'imod_V': 'WS_Mdl.core.path',
    'get_Mdl': 'WS_Mdl.core.path',
    # Model object aliases
    'Mdl_N': 'WS_Mdl.core.mdl',
    'MdlN': 'WS_Mdl.core.mdl',
    # Logging helpers
    'DF_match_MdlN': 'WS_Mdl.core.log',
    'to_Se': 'WS_Mdl.core.log',
    'last_MdlN': 'WS_Mdl.core.log',
    'r_RunLog': 'WS_Mdl.core.log',
    # Text/IO helpers
    'r_Txt_Lns': 'WS_Mdl.core.text',
    # DataFrame accessor and runtime helper
    'DFAccessor': 'WS_Mdl.core.df',
    'timed_import': 'WS_Mdl.core.runtime',
}

# Public names exported by ``from WS_Mdl.core import *`` and used by the CLI
# export discovery in ``WS_Mdl.__main__``.
__all__ = [
    # ----- Style related -----
    'Sep',
    'Sep_2',
    'bold',
    'dim',
    'sprint',
    'style_reset',
    'warn',
    'CuCh',
    'set_verbose',
    # ----- Path related -----
    'REPO_ROOT',
    'Pa_WS',
    'Pa_RunLog',
    'Pa_log_Out',
    'Pa_log_Cfg',
    'MdlN_Pa',
    'imod_V',
    'get_Mdl',
    # ----- MdlN relted -----
    'Mdl_N',
    # ----- Log related -----
    'DF_match_MdlN',
    'to_Se',
    'last_MdlN',
    'r_RunLog',
    # ----- Text related -----
    'r_Txt_Lns',
    # ----- DataFrame accessor -----
    'DFAccessor',
    # ----- Runtime related -----
    'timed_import',
]


def __getattr__(name: str):
    """Resolve exported attributes lazily on first access.

    Flow:
    1) Check whether ``name`` is part of the explicit export mapping.
    2) Import only the module that owns this name.
    3) Fetch the attribute and cache it in ``globals()`` for fast reuse.
    """

    # Reject unknown attributes with the standard module-level error type.
    module_name = _EXPORTS_TO_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    # Import only the module required for this specific symbol.
    module = importlib.import_module(module_name)

    # Backward-compatible alias: expose Mdl_N as MdlN.
    if name == 'MdlN':
        value = getattr(module, 'Mdl_N')
    else:
        value = getattr(module, name)

    # Cache resolved values to avoid repeated imports/lookups.
    globals()[name] = value
    return value
