#!/usr/bin/env python3
"""Single source of truth for the Greece photo set.

All three pipeline stages (caption / gallery / narrate) enumerate photos
through here so they agree on which folders are included, the trip order, and
the per-photo group label. Output files are keyed by the photo *stem*, which is
unique across all folders (verified: the main set is "Greece 2026 - N of 221",
the others are DSC_####).
"""
import re
from pathlib import Path

PICTURES = Path("/home/mjbernaski/Pictures")

# (folder, human label for the gallery section divider), in display order.
PHOTO_SOURCES = [
    ("For Slide Show", "Athens, the Peloponnese & Rhodes"),
]


def _sortkey(name: str):
    """Natural order within a folder: 'N of 221' number, else DSC number, else
    the first integer in the name, else the name itself."""
    m = re.search(r"(\d+)\s+of", name)
    if m:
        return (0, int(m.group(1)))
    m = re.search(r"DSC[_-]?(\d+)", name, re.IGNORECASE)
    if m:
        return (1, int(m.group(1)))
    m = re.search(r"(\d+)", name)
    if m:
        return (2, int(m.group(1)))
    return (3, name)


def list_photos():
    """Every real JPEG across all sources, in display order.

    Returns a list of dicts: {path, stem, group} where group is the section
    label. Skips macOS ._ AppleDouble sidecars and missing folders.
    """
    out = []
    for folder, label in PHOTO_SOURCES:
        d = PICTURES / folder
        if not d.is_dir():
            continue
        photos = [
            p for p in d.iterdir()
            if p.is_file()
            and p.suffix.lower() in (".jpeg", ".jpg")
            and not p.name.startswith("._")
        ]
        for p in sorted(photos, key=lambda p: _sortkey(p.name)):
            out.append({"path": p, "stem": p.stem, "group": label})
    return out
