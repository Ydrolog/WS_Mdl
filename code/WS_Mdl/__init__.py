"""Top-level package initializer for WS_Mdl.

This module keeps imports lightweight by exposing subpackages lazily.
Instead of importing everything at startup, it imports a subpackage only when that attribute is accessed (for example: ``WS_Mdl.core``).

It also exposes selected lightweight core modules directly at package root
for convenience (for example: ``WS_Mdl.path``).
"""

import importlib

# Public attributes exposed from package root and resolved lazily.
_LAZY_EXPORTS = {
    'core': 'WS_Mdl.core',
    'io': 'WS_Mdl.io',
    'imod': 'WS_Mdl.imod',
    'viz': 'WS_Mdl.viz',
    'xr': 'WS_Mdl.xr',
    'path': 'WS_Mdl.core.path',
    'mdl': 'WS_Mdl.core.mdl',
    'style': 'WS_Mdl.core.style',
    'runtime': 'WS_Mdl.core.runtime',
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str):
    """Lazily resolve top-level package attributes.

    If ``name`` is one of the exported attributes in ``__all__``, import its
    configured module path, cache it in ``globals()``, and return it. This
    avoids eager imports and reduces startup overhead.
    """

    module_name = _LAZY_EXPORTS.get(name)
    if module_name:
        # Import the requested module only when needed.
        module = importlib.import_module(module_name)

        # Cache the module so repeated attribute access is fast.
        globals()[name] = module
        return module

    # Standard Python behavior for unknown attributes.
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
