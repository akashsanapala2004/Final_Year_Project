"""Computer Science Ontology (CSO.3.3.owl) Parser, Classifier, and Concept Identifier.

Utilizes CSO.3.3.owl to:
1. Identify scientific concepts from research text against 14,000+ ontological topics and synonyms.
2. Classify concepts into the 4-way taxonomy: Mathematics, Algorithms, Theories, Methods.
3. Resolve ontological super-topics, sub-topics, equivalents, and URIs.
Zero external dependencies (pure Python standard library xml.etree.ElementTree).
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from src.models.schemas import ConceptCategory

CSO_BASE_URI = "https://cso.kmi.open.ac.uk/topics/"

# High-level ancestor sets for 4-way classification
MATH_ANCESTORS = {
    "mathematics",
    "applied_mathematics",
    "discrete_mathematics",
    "linear_algebra",
    "algebra",
    "geometry",
    "calculus",
    "statistics",
    "probability",
    "numerical_analysis",
    "combinatorics",
}

ALGO_ANCESTORS = {
    "algorithms",
    "graph_algorithms",
    "approximation_algorithms",
    "heuristic_algorithms",
    "optimization_algorithms",
    "search_algorithms",
    "sorting_algorithms",
    "machine_learning_algorithms",
    "evolutionary_algorithms",
    "genetic_algorithms",
    "clustering_algorithms",
}

THEORY_ANCESTORS = {
    "theoretical_computer_science",
    "computational_complexity",
    "information_theory",
    "automata_theory",
    "game_theory",
    "formal_methods",
    "coding_theory",
    "control_theory",
    "network_theory",
    "communication",
}


class CSOTopic:
    """Represents a scientific topic in the Computer Science Ontology."""

    def __init__(
        self,
        topic_id: str,
        label: str,
        uri: str,
        super_topics: List[str],
        sub_topics: List[str],
        equivalents: List[str],
        category: ConceptCategory,
    ) -> None:
        self.topic_id = topic_id
        self.label = label
        self.uri = uri
        self.super_topics = super_topics
        self.sub_topics = sub_topics
        self.equivalents = equivalents
        self.category = category

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic_id": self.topic_id,
            "label": self.label,
            "uri": self.uri,
            "category": self.category.value,
            "super_topics": self.super_topics,
            "sub_topics": self.sub_topics,
            "equivalents": self.equivalents,
        }


class CSOOntology:
    """Singleton parser and lookup engine for CSO.3.3.owl."""

    _instance: Optional[CSOOntology] = None

    def __init__(self, owl_path: Optional[Union[str, Path]] = None) -> None:
        self.owl_path = self._resolve_owl_path(owl_path)
        self.topics: Dict[str, CSOTopic] = {}
        self.label_to_topic_id: Dict[str, str] = {}
        self.sub_to_parents: Dict[str, List[str]] = {}
        self._loaded = False
        if self.owl_path and self.owl_path.is_file():
            self._load()

    @classmethod
    def get_instance(cls, owl_path: Optional[Union[str, Path]] = None) -> CSOOntology:
        if cls._instance is None:
            cls._instance = cls(owl_path)
        return cls._instance

    def _resolve_owl_path(self, owl_path: Optional[Union[str, Path]]) -> Optional[Path]:
        if owl_path and Path(owl_path).is_file():
            return Path(owl_path)
        candidates = [
            Path("CSO.3.3.owl"),
            Path("data/CSO.3.3.owl"),
            Path("/home/akash/4ject/CSO.3.3.owl"),
            Path("../CSO.3.3.owl"),
        ]
        for c in candidates:
            if c.is_file():
                return c.resolve()
        return None

    def _load(self) -> None:
        """Fast iterative parse of CSO.3.3.owl using iterparse."""
        if not self.owl_path or not self.owl_path.is_file():
            return

        raw_topics: Dict[str, Dict[str, Any]] = {}

        for event, elem in ET.iterparse(str(self.owl_path), events=("end",)):
            if elem.tag.endswith("Description"):
                about = elem.attrib.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about", "")
                if "topics/" in about:
                    topic_id = about.split("topics/")[-1]
                    label = ""
                    children: List[str] = []
                    equivalents: List[str] = []

                    for child in elem:
                        tag = child.tag
                        if tag.endswith("label") and child.text and not label:
                            label = child.text.strip()
                        elif tag.endswith("superTopicOf"):
                            res = child.attrib.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource", "")
                            if "topics/" in res:
                                c_id = res.split("topics/")[-1]
                                children.append(c_id)
                                self.sub_to_parents.setdefault(c_id, []).append(topic_id)
                        elif tag.endswith("preferentialEquivalent") or tag.endswith("relatedEquivalent"):
                            res = child.attrib.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource", "")
                            if "topics/" in res:
                                equivalents.append(res.split("topics/")[-1])

                    if not label:
                        label = topic_id.replace("_", " ").replace("%28", "(").replace("%29", ")").title()

                    raw_topics[topic_id] = {
                        "topic_id": topic_id,
                        "label": label,
                        "uri": about,
                        "children": children,
                        "equivalents": equivalents,
                    }
                elem.clear()

        # Build CSOTopic instances and classify categories
        for tid, data in raw_topics.items():
            lbl = data["label"]
            parents = self.sub_to_parents.get(tid, [])
            cat = self.classify(tid, lbl, parents)

            topic_obj = CSOTopic(
                topic_id=tid,
                label=lbl,
                uri=data["uri"],
                super_topics=parents,
                sub_topics=data["children"],
                equivalents=data["equivalents"],
                category=cat,
            )
            self.topics[tid] = topic_obj

            # Map normalized labels to topic_id
            self.label_to_topic_id[lbl.lower()] = tid
            # Also clean punctuation
            clean_lbl = re.sub(r"[^a-zA-Z0-9\s]", " ", lbl.lower())
            clean_lbl_norm = " ".join(clean_lbl.split())
            if clean_lbl_norm:
                self.label_to_topic_id[clean_lbl_norm] = tid

            # Also index topic ID cleaned
            tid_clean = tid.replace("_", " ").lower()
            self.label_to_topic_id[tid_clean] = tid

        self._loaded = True

    def classify(self, topic_id: str, label: str, parents: Optional[List[str]] = None) -> ConceptCategory:
        """Classify a CSO topic into Mathematics, Algorithms, Theories, or Methods."""
        t_low = (topic_id + " " + label).lower()
        parents = parents or self.sub_to_parents.get(topic_id, [])

        # 1. Methods check
        if any(w in t_low for w in [
            "cancellation", "reduction", "reasoning", "multiplexing", "modulation",
            "scheme", "allocation", "technique", "method", "filtering", "estimation",
            "checking", "control", "access control", "system", "architecture", "protocol",
            "case-based", "watermarking", "segmentation", "enhancement"
        ]):
            return ConceptCategory.METHODS

        # 2. Algorithms check
        if any(w in t_low for w in [
            "algorithm", "heuristic", "traversal", "search", "descent", "sort",
            "decomposition", "solver", "scan", "test", "glrt", "dynamic programming",
            "k-means", "viterbi", "graham", "simplex", "backtracking", "genetic"
        ]):
            return ConceptCategory.ALGORITHMS

        # 3. Mathematics check
        if any(w in t_low for w in [
            "algebra", "subspace", "matrix", "vector", "distribution", "calculus",
            "norm", "integral", "equation", "projection", "eigen", "fourier",
            "kronecker", "probability", "statistics", "optimization", "convex",
            "differential", "stochastic", "space", "transform"
        ]):
            return ConceptCategory.MATHEMATICS

        # 4. Theories check
        if any(w in t_low for w in [
            "theory", "complexity", "principle", "theorem", "law", "game",
            "property", "security", "framework", "radio", "sensing", "automata",
            "computability", "information theory", "semantics", "logic", "formal"
        ]):
            return ConceptCategory.THEORIES

        # Check ancestor hierarchy
        for p in parents:
            p_low = p.lower()
            if any(m in p_low for m in MATH_ANCESTORS):
                return ConceptCategory.MATHEMATICS
            if any(a in p_low for a in ALGO_ANCESTORS):
                return ConceptCategory.ALGORITHMS
            if any(th in p_low for th in THEORY_ANCESTORS):
                return ConceptCategory.THEORIES

        return ConceptCategory.METHODS

    def identify(self, text: str) -> Optional[CSOTopic]:
        """Identify if a concept name matches a CSO topic."""
        clean = text.strip().lower()
        tid = self.label_to_topic_id.get(clean)
        if not tid:
            # Try removing special characters
            clean_punct = " ".join(re.sub(r"[^a-zA-Z0-9\s]", " ", clean).split())
            tid = self.label_to_topic_id.get(clean_punct)

        if tid and tid in self.topics:
            return self.topics[tid]

        # Fuzzy check in labels
        for lbl, t_id in self.label_to_topic_id.items():
            if len(lbl) > 4 and (lbl == clean or clean in lbl or lbl in clean):
                return self.topics.get(t_id)

        return None

    def extract_from_text(self, text: str, max_ngram: int = 5) -> List[CSOTopic]:
        """Extract matching CSO topics from text using sliding n-gram matching."""
        if not text:
            return []

        words = re.findall(r"[a-zA-Z0-9\-]+", text.lower())
        matched_topics: Dict[str, Tuple[CSOTopic, int]] = {}  # topic_id -> (topic, ngram_length)
        occupied_indices: Set[int] = set()

        # Traverse from longest n-grams (5) to shortest (2 or 1) to match maximal phrases first
        for n in range(max_ngram, 1, -1):
            for i in range(len(words) - n + 1):
                indices = set(range(i, i + n))
                if indices.intersection(occupied_indices):
                    continue

                ngram = " ".join(words[i:i + n])
                tid = self.label_to_topic_id.get(ngram)
                if tid and tid in self.topics:
                    topic = self.topics[tid]
                    matched_topics[tid] = (topic, n)
                    occupied_indices.update(indices)

        # Single word check for unambiguous scientific terms (length > 4)
        for i, w in enumerate(words):
            if i in occupied_indices or len(w) < 4:
                continue
            tid = self.label_to_topic_id.get(w)
            if tid and tid in self.topics:
                # Filter out generic English stop-words that appear in CSO (e.g. 'paper', 'user')
                if w not in {"paper", "users", "system", "method", "level", "point", "model", "order", "state", "case", "process", "result"}:
                    matched_topics[tid] = (self.topics[tid], 1)
                    occupied_indices.add(i)

        return [t for t, _ in matched_topics.values()]


# Module-level convenience functions
def get_cso_ontology(owl_path: Optional[Union[str, Path]] = None) -> CSOOntology:
    return CSOOntology.get_instance(owl_path)


def identify_cso_concept(text: str) -> Optional[CSOTopic]:
    return get_cso_ontology().identify(text)


def classify_cso_concept(text: str) -> ConceptCategory:
    topic = identify_cso_concept(text)
    if topic:
        return topic.category
    return get_cso_ontology().classify(text, text)
