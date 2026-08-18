import torch
import pytest
from lema._tensorstore import TensorStore, Stream, StreamKind


def test_register_and_ensure():
    store = TensorStore()
    s = Stream(StreamKind.WEIGHTS, 0, "q_proj.weight", (4, 4), torch.float32)
    store.register(s)
    got = store.ensure((s.kind, s.layer_id, s.key))
    assert got[(s.kind, s.layer_id, s.key)].shape == (4, 4)


def test_stream_identity_uniqueness():
    store = TensorStore()
    s1 = Stream(StreamKind.WEIGHTS, 0, "q_proj.weight", (4, 4), torch.float32)
    s2 = Stream(StreamKind.WEIGHTS, 0, "q_proj.weight", (4, 4), torch.float32)
    store.register(s1)
    with pytest.raises(ValueError):
        store.register(s2)


def test_lru_eviction():
    store = TensorStore(capacity=2)
    for i in range(3):
        store.register(Stream(StreamKind.WEIGHTS, i, "w", (2,), torch.float32,
                              source=lambda: torch.ones(2)))
        store.ensure((StreamKind.WEIGHTS, i, "w"))
    assert store.slots()[StreamKind.WEIGHTS] == 2
    with pytest.raises(KeyError):
        store.ensure((StreamKind.WEIGHTS, 0, "w"), _no_evict=True)


def test_source_lazy_load():
    calls = []
    def loader():
        calls.append(1)
        return torch.arange(6).reshape(2, 3)
    store = TensorStore()
    store.register(Stream(StreamKind.WEIGHTS, 1, "w", (2, 3), torch.int64, source=loader))
    t = store.ensure((StreamKind.WEIGHTS, 1, "w"))[(StreamKind.WEIGHTS, 1, "w")]
    assert torch.equal(t, torch.arange(6).reshape(2, 3))
    assert len(calls) == 1
