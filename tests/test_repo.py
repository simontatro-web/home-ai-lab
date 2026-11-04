import re
import struct
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = [p for p in ROOT.rglob("*.md") if ".git" not in p.parts]
LINK = re.compile(r"""(?:\]\(|src=")([^)"#\s]+)""")


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_local_links_and_images_exist(doc):
    missing = [t for t in LINK.findall(doc.read_text(encoding="utf-8"))
               if not t.startswith(("http://", "https://", "mailto:")) and not (doc.parent / t).exists()]
    assert not missing


def test_tour_video_duration():
    data = (ROOT / "docs/video/tour.mp4").read_bytes()
    i = data.index(b"mvhd") + 4
    if data[i] == 0:
        timescale, duration = struct.unpack(">II", data[i + 12:i + 20])
    else:
        timescale, duration = struct.unpack(">IQ", data[i + 20:i + 32])
    assert 4.4 <= duration / timescale <= 4.6
