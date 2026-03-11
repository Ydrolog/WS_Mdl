"""
Public export surface for WS_Mdl.core.

This module intentionally resolves exports lazily. Importing WS_Mdl.core should not immediately import every "heavy" dependency chain used by submodules.

How to use:
- Importing from WS_Mdl.core (e.g. `from WS_Mdl.core import Mdl_N`) will load the specific submodule containing that export (in this case, WS_Mdl.core.mdl) and cache the result for future accesses.
- This allows for faster initial load times and avoids unnecessary imports if only a subset of utilities is needed in a given context.
"""

import importlib

# Keep module path aliases private so they do not shadow actual submodules
# during statements like: from WS_Mdl.core import log
_MOD_STYLE = 'WS_Mdl.core.style'
_MOD_PATH = 'WS_Mdl.core.path'
_MOD_LOG = 'WS_Mdl.core.log'

# Map exported public names to (module_path, attribute_name).
# Keeping this explicit makes the public contract stable and easy to audit.
_EXPORTS_TO_MODULES = {
    # Style utilities
    'Sep': (_MOD_STYLE, 'Sep'),
    'Sep_2': (_MOD_STYLE, 'Sep_2'),
    'bold': (_MOD_STYLE, 'bold'),
    'dim': (_MOD_STYLE, 'dim'),
    'sprint': (_MOD_STYLE, 'sprint'),
    'sinput': (_MOD_STYLE, 'sinput'),
    'style_reset': (_MOD_STYLE, 'style_reset'),
    'warn': (_MOD_STYLE, 'warn'),
    'CuCh': (_MOD_STYLE, 'CuCh'),
    'set_verbose': (_MOD_STYLE, 'set_verbose'),
    # Path/model-location utilities
    'REPO_ROOT': (_MOD_PATH, 'REPO_ROOT'),
    'Pa_WS': (_MOD_PATH, 'Pa_WS'),
    'Pa_RunLog': (_MOD_PATH, 'Pa_RunLog'),
    'Pa_log_Out': (_MOD_PATH, 'Pa_log_Out'),
    'Pa_log_Cfg': (_MOD_PATH, 'Pa_log_Cfg'),
    'MdlN_Pa': (_MOD_PATH, 'MdlN_Pa'),
    'MdlN_PaView': (_MOD_PATH, 'MdlN_PaView'),
    'imod_V': (_MOD_PATH, 'imod_V'),
    'get_Mdl': (_MOD_PATH, 'get_Mdl'),
    # Model object aliases
    'Mdl_N': ('WS_Mdl.core.mdl', 'Mdl_N'),
    # Logging helpers
    'DF_match_MdlN': (_MOD_LOG, 'DF_match_MdlN'),
    'to_Se': (_MOD_LOG, 'to_Se'),
    'last_MdlN': (_MOD_LOG, 'last_MdlN'),
    'r_RunLog': (_MOD_LOG, 'r_RunLog'),
    'get_B': (_MOD_LOG, 'get_B'),
    # Text/IO helpers
    'r_Txt_Lns': ('WS_Mdl.core.text', 'r_Txt_Lns'),
    # DataFrame accessor and runtime helpers
    'DFAccessor': ('WS_Mdl.core.df', 'DFAccessor'),
    # Timing utilities
    'timed_import': ('WS_Mdl.core.runtime', 'timed_import'),
    'timed_execution': ('WS_Mdl.core.runtime', 'timed_execution'),
}

# Public names exported by "from WS_Mdl.core import *".
# Keep this derived from the mapping to prevent drift.
__all__ = list(_EXPORTS_TO_MODULES)


def __getattr__(name: str):
    """Resolve exported attributes lazily on first access."""

    target = _EXPORTS_TO_MODULES.get(name)
    if target is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    module_name, attr_name = target
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)

    # Cache resolved values for fast subsequent access.
    globals()[name] = value
    return value


def __dir__():
    """Expose lazy exports in dir(WS_Mdl.core) for discoverability."""

    return sorted(set(globals()) | set(__all__))
