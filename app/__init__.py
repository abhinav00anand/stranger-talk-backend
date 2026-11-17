"""Stranger Talk (ChatFreely) backend application package.

Includes Windows Python 3.14 safeguard for ujson C-extension DLL loading failure.
"""

import sys

# Windows Python 3.14 safeguard: fastapi.responses optionally imports ujson.
# If ujson C-extension fails to load DLLs on Windows Python 3.14, setting
# sys.modules["ujson"] = None ensures FastAPI falls back cleanly to standard json / orjson.
sys.modules.setdefault("ujson", None)
