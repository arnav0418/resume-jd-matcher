import json
from pathlib import Path
from typing import Dict, List, Optional


def load_jds(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_jd_by_id(jds: List[Dict], jd_id: str) -> Optional[Dict]:
    for jd in jds:
        if jd.get("id") == jd_id:
            return jd
    return None
