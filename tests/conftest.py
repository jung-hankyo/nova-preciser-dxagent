import pytest

from dxagent.config import get_settings
from dxagent.runtime import make_store


@pytest.fixture
def settings(tmp_path):
    s = get_settings()
    s.contrast_cards = False
    s.live_fallback = False
    s.local_fallback = False
    s.kb_dir = tmp_path / "kb"  # never the developer's compiled kb/
    s.cache_dir = tmp_path / "cache"
    s.sources_dir = tmp_path / "sources"
    return s


@pytest.fixture
def store(settings):
    """The demo fixture cards (retired v0.3 seeds): plumbing tests only, not knowledge."""
    return make_store(settings, fixtures=True)


def hyp(store, cid, prior):
    c = store.get(cid)
    return {"disease_id": c.id, "name": c.name, "prior": prior, "cant_miss": c.cant_miss}


def finding(code, present=True, conf=0.95, source="history", turn=1, cc=False):
    return {"code": code, "label": code, "present": present, "confidence": conf, "source": source, "evidence": "", "turn": turn, "chief_complaint": cc}


def result(test_id, outcome, conf=0.97, turn=1):
    return {"test_id": test_id, "outcome": outcome, "confidence": conf, "turn": turn, "evidence": ""}
