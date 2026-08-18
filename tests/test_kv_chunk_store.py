import torch
import pytest
from lema._tensorstore import KVChunkStore


def test_stash_load_roundtrip():
    s = KVChunkStore(kv_chunk_size=8)
    k = torch.randn(2, 4, 8, 16)
    v = torch.randn(2, 4, 8, 16)
    s.stash(3, 0, k, v)
    k2, v2 = s.load(3, 0)
    assert torch.equal(k, k2) and torch.equal(v, v2)


def test_chunk_boundary_rolls():
    s = KVChunkStore(kv_chunk_size=4)
    for i in range(5):
        s.append(0, torch.randn(2, 4, 1, 16), torch.randn(2, 4, 1, 16))
    assert s.num_chunks(0) == 2


def test_ram_disk_fallback(tmp_path):
    s = KVChunkStore(kv_chunk_size=4, max_ram_gb=0.0, disk_dir=str(tmp_path))
    k = torch.randn(2, 4, 4, 16)
    v = torch.randn(2, 4, 4, 16)
    s.stash(0, 0, k, v)
    k2, v2 = s.load(0, 0)
    assert torch.equal(k, k2) and torch.equal(v, v2)


def test_append_grows_current_chunk():
    s = KVChunkStore(kv_chunk_size=4)
    for i in range(3):
        s.append(1, torch.randn(2, 4, 1, 16), torch.randn(2, 4, 1, 16))
    assert s.current_size(1) == 3
    assert s.num_chunks(1) == 1
