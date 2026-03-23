from __future__ import annotations

import re


VARIANT_MODE_SUFFIX_RE = re.compile(
    r"_(ASD(?:FR|HR|NG)[A-Za-z0-9]*|AVIRIS[A-Za-z0-9]*|BECK[A-Za-z0-9]*|NIC4[A-Za-z0-9]*)_(AREF|RREF|RTGC|TRAN)$",
    flags=re.IGNORECASE,
)


def normalize_display_name(raw_name: str) -> str:
    s = raw_name.strip()
    if s.lower().startswith("splib07b_"):
        s = s[9:]
    s = VARIANT_MODE_SUFFIX_RE.sub("", s)
    return s.replace("_", " ").strip() or raw_name


def parse_variant_mode(raw_name: str) -> tuple[str | None, str | None]:
    m = VARIANT_MODE_SUFFIX_RE.search(raw_name)
    if not m:
        return None, None
    return m.group(1).upper(), m.group(2).upper()
