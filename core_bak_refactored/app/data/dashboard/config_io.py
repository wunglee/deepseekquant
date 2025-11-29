import json
from typing import Dict


def export_config(config: Dict, filepath: str) -> bool:
    try:
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception:
        return False


def import_config(filepath: str) -> Dict:
    with open(filepath, 'r') as f:
        return json.load(f)
