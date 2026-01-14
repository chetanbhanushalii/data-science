import json
from pathlib import Path
from typing import Dict,Any

CONFIG_PATH = Path(__file__).resolve().parents[1]/"config.json"

def load_config() -> Dict[str,Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"config.json not found at location:{CONFIG_PATH}")
    with CONFIG_PATH.open("r",encoding="utf-8") as f:
        return json.load(f)

# print(load_config())