import pytest

from core.concepts import Concept, ConceptSet, build_concept_graph


def test_concept_model_validates_source_reference() -> None:
    concept = Concept(
        name="Retrieval",
        definition="Finding relevant lecture context.",
        first_mention_timestamp="00:12",
        related_concepts=["Embeddings"],
    )
    assert concept.first_mention_timestamp == "00:12"


def test_graph_adds_cooccurrence_edges() -> None:
    pytest.importorskip("networkx")
    concepts = ConceptSet(
        concepts=[
            Concept(name="Retrieval", definition="Finding context.", first_mention_timestamp="00:12"),
            Concept(name="Embeddings", definition="Vector representations.", first_mention_timestamp="00:30"),
        ]
    )
    chunks = [type("Chunk", (), {"text": "Retrieval uses Embeddings to find context."})()]
    graph = build_concept_graph(concepts, chunks)
    assert graph.has_edge("Retrieval", "Embeddings")
    assert graph["Retrieval"]["Embeddings"]["relation"] in {"co-occurrence", "co-occurrence + similarity"}
