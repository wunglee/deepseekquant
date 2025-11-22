import os
import sys
import importlib
# Ensure project root is on sys.path for absolute imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# Preload real infrastructure package to avoid name shadowing by tests/infrastructure/
try:
    infra = importlib.import_module('infrastructure')
    sys.modules['infrastructure'] = infra
except Exception:
    pass
