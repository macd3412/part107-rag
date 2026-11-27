"""Offline unit tests for HybridRetriever RRF fusion using dummy vectors.

No live API calls: document and query embeddings are hand-built numpy arrays and
the network embed step is monkeypatched.
"""

import numpy as np

from src.rag.retriever import HybridRetriever, reciprocal_rank_fusion, tokenize


def test_tokenize_lowercases_and_keeps_dotted_tokens():
    assert tokenize("Part 107.51 Night-Ops!") == ["part", "107.51", "night", "ops"]


def test_rrf_rewards_items_ranked_high_by_both_lists():
    # Index 2 is top of list A and second of list B -> best fused score.
    list_a = [2, 0, 1]
    list_b = [1, 2, 0]
    fused = reciprocal_rank_fusion([list_a, list_b], top_k=3)
    assert fused[0] == 2


def test_rrf_respects_top_k():
    fused = reciprocal_rank_fusion([[0, 1, 2, 3, 4]], top_k=2)
    assert fused == [0, 1]


def test_rrf_sums_scores_across_lists():
    # Item 5 appears only once; item 0 appears first in both lists.
    fused = reciprocal_rank_fusion([[0, 5], [0, 9]], top_k=3)
    assert fused[0] == 0


def _dummy_retriever():
    chunks = [
        {"text": "maximum groundspeed altitude operating limitations", "id": "a"},
        {"text": "night lighting anti-collision requirements", "id": "b"},
        {"text": "remote identification broadcast module", "id": "c"},
    ]
    # Orthonormal-ish document vectors, already unit-normalized.
    embeddings = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return HybridRetriever(chunks=chunks, embeddings=embeddings, api_key="test-key")


def test_search_fuses_dense_and_lexical(monkeypatch):
    retriever = _dummy_retriever()
    # Query vector points at document 1 ("night") for the dense signal.
    monkeypatch.setattr(
        retriever, "embed_query", lambda q: np.array([0.0, 1.0, 0.0], dtype=np.float32)
    )

    results = retriever.search("night lighting", top_k=3)
    assert len(results) == 3
    # Document b wins: dense vector and BM25 lexical match both favor it.
    assert results[0]["id"] == "b"


def test_search_top_k_limits_results(monkeypatch):
    retriever = _dummy_retriever()
    monkeypatch.setattr(
        retriever, "embed_query", lambda q: np.array([1.0, 0.0, 0.0], dtype=np.float32)
    )
    results = retriever.search("groundspeed", top_k=1)
    assert len(results) == 1
    assert results[0]["id"] == "a"


def test_constructor_does_not_call_network():
    # Building the retriever with injected data must not require an API key.
    retriever = HybridRetriever(
        chunks=[{"text": "hello world", "id": "x"}],
        embeddings=np.array([[1.0]], dtype=np.float32),
    )
    assert retriever.chunks[0]["id"] == "x"
