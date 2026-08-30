"""Retired embedded dashboard bundle.

The dashboard JavaScript now lives in ``dashboard/ui/src`` and is bundled by
Vite. ``_APP_JS`` remains as an empty compatibility constant for downstream
code that imported the old private module; it is never served by Slowave.
"""

_APP_JS = ""
