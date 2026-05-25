"""
graph_rag_builder.py
─────────────────────
Builds a NetworkX knowledge graph from extracted JSONL records and persists it.
Nodes: Concept, Misconception, PISACompetency, Question
Edges: HAS_MISCONCEPTION, TESTS_COMPETENCY, RELATED_CONCEPT, PREREQUISITE_OF, APPEARS_IN_QUESTION

The graph augments vector retrieval: when a student asks about a concept, we
retrieve its subgraph neighbours to pull in prerequisite concepts and common
misconceptions that the vector search alone might miss.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import yaml

logger = logging.getLogger(__name__)

# Node types
NT_CONCEPT = "Concept"
NT_MISCONCEPTION = "Misconception"
NT_COMPETENCY = "PISACompetency"
NT_QUESTION = "Question"

# Edge types
ET_HAS_MISCONCEPTION = "HAS_MISCONCEPTION"
ET_TESTS_COMPETENCY = "TESTS_COMPETENCY"
ET_RELATED_CONCEPT = "RELATED_CONCEPT"
ET_PREREQUISITE = "PREREQUISITE_OF"
ET_APPEARS_IN = "APPEARS_IN_QUESTION"


def _norm(text: str) -> str:
    """Normalise a string to a canonical node key."""
    return text.strip().lower().replace("  ", " ")


class KnowledgeGraphBuilder:
    """
    Builds and persists the PISA knowledge graph.

    Usage
    -----
    builder = KnowledgeGraphBuilder(cfg)
    builder.ingest_jsonl("data/processed_jsonl/knowledge_base.jsonl")
    builder.save()

    # Later:
    G = KnowledgeGraphBuilder.load(cfg)
    subgraph = KnowledgeGraphBuilder.get_subgraph(G, seed_concept, max_hops=2)
    """

    def __init__(self, config_path: Path):
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.graph_path = Path(cfg["graph"]["storage_path"])
        self.max_hops: int = cfg["graph"]["max_hops"]
        self.max_subgraph_nodes: int = cfg["graph"]["max_subgraph_nodes"]
        self.G: nx.DiGraph = nx.DiGraph()

    # ── graph construction ────────────────────────────────────────────────────

    def _add_node(self, node_id: str, node_type: str, **attrs):
        if not self.G.has_node(node_id):
            self.G.add_node(node_id, type=node_type, **attrs)

    def _add_edge(self, src: str, dst: str, rel: str, **attrs):
        self.G.add_edge(src, dst, relation=rel, **attrs)

    def ingest_record(self, record: Dict[str, Any]):
        """Add one knowledge base record to the graph."""
        concept_key = _norm(record["correct_concept"])
        question_id = f"q:{_norm(record['question_summary'][:60])}"
        competency_key = _norm(record.get("pisa_competency", "unknown"))
        topic_key = _norm(record.get("topic", "general"))

        # ── Nodes ──────────────────────────────────────────────────────────
        self._add_node(
            concept_key, NT_CONCEPT,
            label=record["correct_concept"],
            topic=record.get("topic", ""),
            cognitive_demand=record.get("cognitive_demand", "Medium"),
        )
        self._add_node(question_id, NT_QUESTION,
                       label=record["question_summary"][:120],
                       source_file=record.get("source_file", ""),
                       page_range=record.get("page_range", ""))
        self._add_node(competency_key, NT_COMPETENCY,
                       label=record.get("pisa_competency", ""))
        self._add_node(topic_key, NT_CONCEPT,
                       label=record.get("topic", topic_key))

        # ── Misconceptions ────────────────────────────────────────────────
        misc_raw = record.get("common_misconceptions", "")
        for misc in misc_raw.split(";"):
            misc = misc.strip()
            if len(misc) > 5:
                misc_key = _norm(misc)
                self._add_node(misc_key, NT_MISCONCEPTION, label=misc)
                self._add_edge(concept_key, misc_key, ET_HAS_MISCONCEPTION)

        # ── Key vocabulary → related concepts ─────────────────────────────
        for vocab in record.get("key_vocabulary", []):
            vocab_key = _norm(vocab)
            if vocab_key != concept_key:
                self._add_node(vocab_key, NT_CONCEPT, label=vocab)
                self._add_edge(concept_key, vocab_key, ET_RELATED_CONCEPT)

        # ── Edges ──────────────────────────────────────────────────────────
        self._add_edge(question_id, concept_key, ET_APPEARS_IN)
        self._add_edge(concept_key, competency_key, ET_TESTS_COMPETENCY)
        self._add_edge(topic_key, concept_key, ET_RELATED_CONCEPT)

    def ingest_jsonl(self, jsonl_path: Path):
        """Ingest all records from the knowledge base JSONL file."""
        if not jsonl_path.exists():
            raise FileNotFoundError(f"JSONL not found: {jsonl_path}")
        count = 0
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    self.ingest_record(record)
                    count += 1
                except json.JSONDecodeError as e:
                    logger.warning(f"[Graph] Skipping malformed JSONL line: {e}")
        logger.info(
            f"[Graph] Ingested {count} records → "
            f"{self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges"
        )

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self):
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.graph_path, "wb") as f:
            pickle.dump(self.G, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"[Graph] Saved to {self.graph_path}")

    @staticmethod
    def load(config_path: Path) -> nx.DiGraph:
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        graph_path = Path(cfg["graph"]["storage_path"])
        if not graph_path.exists():
            logger.warning("[Graph] No graph file found — returning empty graph")
            return nx.DiGraph()
        with open(graph_path, "rb") as f:
            G = pickle.load(f)
        logger.info(
            f"[Graph] Loaded {G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges from {graph_path}"
        )
        return G

    # ── retrieval ─────────────────────────────────────────────────────────────

    @staticmethod
    def get_subgraph_context(
        G: nx.DiGraph,
        seed_query: str,
        max_hops: int = 2,
        max_nodes: int = 20,
    ) -> Dict[str, Any]:
        """
        Given a free-text query, find the best-matching seed node and return
        a subgraph context dict containing related concepts, misconceptions, etc.

        Returns
        -------
        dict with keys: concepts, misconceptions, competencies, graph_summary
        """
        if G.number_of_nodes() == 0:
            return {"concepts": [], "misconceptions": [], "competencies": [], "graph_summary": ""}

        query_norm = _norm(seed_query)

        # Fuzzy seed: find nodes whose label contains query words
        query_words: Set[str] = set(query_norm.split())
        scored: List[Tuple[float, str]] = []
        for node, data in G.nodes(data=True):
            label = _norm(data.get("label", node))
            overlap = len(query_words & set(label.split()))
            if overlap > 0:
                scored.append((overlap / max(len(query_words), 1), node))

        if not scored:
            return {"concepts": [], "misconceptions": [], "competencies": [], "graph_summary": ""}

        scored.sort(reverse=True)
        seed_node = scored[0][1]

        # BFS up to max_hops
        visited: Set[str] = {seed_node}
        frontier = {seed_node}
        for _ in range(max_hops):
            next_frontier: Set[str] = set()
            for n in frontier:
                next_frontier |= set(G.successors(n)) | set(G.predecessors(n))
            next_frontier -= visited
            visited |= next_frontier
            frontier = next_frontier
            if len(visited) >= max_nodes:
                break

        visited = set(list(visited)[:max_nodes])
        subg = G.subgraph(visited)

        concepts, misconceptions, competencies = [], [], []
        for node, data in subg.nodes(data=True):
            ntype = data.get("type", "")
            label = data.get("label", node)
            if ntype == NT_CONCEPT:
                concepts.append(label)
            elif ntype == NT_MISCONCEPTION:
                misconceptions.append(label)
            elif ntype == NT_COMPETENCY:
                competencies.append(label)

        graph_summary = (
            f"Concepts: {'; '.join(concepts[:5])} | "
            f"Common misconceptions: {'; '.join(misconceptions[:4])} | "
            f"PISA competency: {'; '.join(competencies[:2])}"
        )

        return {
            "concepts": concepts,
            "misconceptions": misconceptions,
            "competencies": competencies,
            "graph_summary": graph_summary,
        }
