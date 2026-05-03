from __future__ import annotations
import re
from typing import Optional
from core.schemas import BabyMetadata, MotherMetadata


def _parse_fields(hea_path: str) -> dict[str, str]:
    """Read all #Key   Value lines from a .hea file into a dict."""
    fields: dict[str, str] = {}
    try:
        with open(hea_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line.startswith("#"):
                    continue
                content = line[1:].strip()  # strip leading # and whitespace
                # Skip comment/section header lines (start with - or contain no space)
                if not content or content.startswith("-"):
                    continue
                # Split on whitespace from the right: key is everything before last token,
                # value is the last token. This handles both single and multi-space separators.
                parts = content.rsplit(None, 1)
                if len(parts) == 2:
                    fields[parts[0].strip()] = parts[1].strip()
    except FileNotFoundError:
        pass
    return fields


def _int(fields: dict[str, str], key: str) -> Optional[int]:
    v = fields.get(key)
    if v is None:
        return None
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return None


def _bool(fields: dict[str, str], key: str) -> Optional[bool]:
    v = _int(fields, key)
    return bool(v) if v is not None else None


_SEX_MAP = {"1": "Male", "2": "Female"}


def parse_hea_file(hea_path: str, record_id: str) -> tuple[BabyMetadata, MotherMetadata]:
    """
    Parse a WFDB .hea file and return structured baby and mother metadata.
    Returns objects with all-None fields if the file is missing or unreadable.
    """
    fields = _parse_fields(hea_path)

    sex_raw = fields.get("Sex")
    baby = BabyMetadata(
        baby_id=record_id,
        gestational_weeks=_int(fields, "Gest. weeks"),
        weight_g=_int(fields, "Weight(g)"),
        sex=_SEX_MAP.get(sex_raw) if sex_raw else None,
        apgar1=_int(fields, "Apgar1"),
        apgar5=_int(fields, "Apgar5"),
    )

    mother = MotherMetadata(
        mother_age=_int(fields, "Age"),
        gravidity=_int(fields, "Gravidity"),
        parity=_int(fields, "Parity"),
        diabetes=_bool(fields, "Diabetes"),
        hypertension=_bool(fields, "Hypertension"),
        preeclampsia=_bool(fields, "Preeclampsia"),
    )

    return baby, mother
