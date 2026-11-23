"""Data models and schemas for Scientific Prerequisite Concept Extraction & Knowledge Graph System."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class ConceptCategory(str, Enum):
    """Four-way taxonomic categories for scientific concepts."""

    MATHEMATICS = "Mathematics"
    ALGORITHMS = "Algorithms"
    THEORIES = "Theories"
    METHODS = "Methods"

    @classmethod
    def from_str(cls, val: Union[str, ConceptCategory]) -> ConceptCategory:
        """Parse string to ConceptCategory case-insensitively."""
        if isinstance(val, cls):
            return val
        val_clean = str(val).strip().capitalize()
        for member in cls:
            if member.value.lower() == val_clean.lower():
                return member
        # Fallback to Methods or closest
        return cls.METHODS


def _to_serializable(val: Any) -> Any:
    """Helper to convert dataclasses, enums, lists, and dicts to plain Python primitives."""
    if isinstance(val, Enum):
        return val.value
    if is_dataclass(val):
        return {k: _to_serializable(v) for k, v in asdict(val).items()}
    if isinstance(val, list):
        return [_to_serializable(item) for item in val]
    if isinstance(val, dict):
        return {str(k): _to_serializable(v) for k, v in val.items()}
    return val


@dataclass
class Formula:
    """Represents a mathematical equation or formula extracted from TEI-XML."""

    formula_id: str
    raw_text: str
    latex: str = ""
    section_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return _to_serializable(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Formula:
        return cls(
            formula_id=str(data.get("formula_id", "")),
            raw_text=str(data.get("raw_text", "")),
            latex=str(data.get("latex", "")),
            section_id=str(data.get("section_id", "")),
        )


@dataclass
class Section:
    """Represents a section in an academic paper."""

    section_id: str
    heading: str
    category: str
    paragraphs: List[str] = field(default_factory=list)
    formulas: List[Formula] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _to_serializable(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Section:
        formulas = [Formula.from_dict(f) for f in data.get("formulas", [])]
        return cls(
            section_id=str(data.get("section_id", "")),
            heading=str(data.get("heading", "")),
            category=str(data.get("category", "")),
            paragraphs=list(data.get("paragraphs", [])),
            formulas=formulas,
            citations=list(data.get("citations", [])),
        )


@dataclass
class DocumentIR:
    """Intermediate Representation of an academic research paper."""

    paper_id: str
    title: str
    abstract: str
    sections: List[Section] = field(default_factory=list)
    preliminaries: List[str] = field(default_factory=list)
    citations: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _to_serializable(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DocumentIR:
        sections = [Section.from_dict(s) for s in data.get("sections", [])]
        return cls(
            paper_id=str(data.get("paper_id", "")),
            title=str(data.get("title", "")),
            abstract=str(data.get("abstract", "")),
            sections=sections,
            preliminaries=list(data.get("preliminaries", [])),
            citations=dict(data.get("citations", {})),
        )


@dataclass
class ConceptEntity:
    """Scientific concept entity extracted from the paper."""

    concept_id: str
    name: str
    category: ConceptCategory
    confidence: float = 1.0
    source_sections: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    is_target_paper_contribution: bool = False
    cso_topic_id: Optional[str] = None
    cso_uri: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = _to_serializable(self)
        if isinstance(self.category, ConceptCategory):
            d["category"] = self.category.value
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ConceptEntity:
        return cls(
            concept_id=str(data.get("concept_id", "")),
            name=str(data.get("name", "")),
            category=ConceptCategory.from_str(data.get("category", ConceptCategory.METHODS)),
            confidence=float(data.get("confidence", 1.0)),
            source_sections=list(data.get("source_sections", [])),
            aliases=list(data.get("aliases", [])),
            is_target_paper_contribution=bool(data.get("is_target_paper_contribution", False)),
            cso_topic_id=data.get("cso_topic_id"),
            cso_uri=data.get("cso_uri"),
        )


@dataclass
class PrerequisiteEdge:
    """Directed edge representing A is a prerequisite for B (source_id -> target_id)."""

    source_id: str
    source_name: str
    target_id: str
    target_name: str
    relation_type: str = "requires_prerequisite"
    confidence: float = 1.0
    asymmetry_score: float = 0.0
    epistemic_rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return _to_serializable(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PrerequisiteEdge:
        return cls(
            source_id=str(data.get("source_id", "")),
            source_name=str(data.get("source_name", "")),
            target_id=str(data.get("target_id", "")),
            target_name=str(data.get("target_name", "")),
            relation_type=str(data.get("relation_type", "requires_prerequisite")),
            confidence=float(data.get("confidence", 1.0)),
            asymmetry_score=float(data.get("asymmetry_score", 0.0)),
            epistemic_rationale=str(data.get("epistemic_rationale", "")),
        )


@dataclass
class GraphMetrics:
    """Topological metrics for the Prerequisite Knowledge Graph."""

    num_nodes: int
    num_edges: int
    is_dag: bool
    cycles_detected_and_resolved: int = 0
    density: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return _to_serializable(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GraphMetrics:
        return cls(
            num_nodes=int(data.get("num_nodes", 0)),
            num_edges=int(data.get("num_edges", 0)),
            is_dag=bool(data.get("is_dag", False)),
            cycles_detected_and_resolved=int(data.get("cycles_detected_and_resolved", 0)),
            density=float(data.get("density", 0.0)),
        )


@dataclass
class PrerequisiteDAG:
    """Directed Acyclic Graph of Prerequisite Concepts."""

    paper_id: str
    metrics: GraphMetrics
    nodes: List[ConceptEntity] = field(default_factory=list)
    edges: List[PrerequisiteEdge] = field(default_factory=list)
    adjacency_list: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _to_serializable(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PrerequisiteDAG:
        nodes = [ConceptEntity.from_dict(n) for n in data.get("nodes", [])]
        edges = [PrerequisiteEdge.from_dict(e) for e in data.get("edges", [])]
        metrics = GraphMetrics.from_dict(data.get("metrics", {}))
        return cls(
            paper_id=str(data.get("paper_id", "")),
            metrics=metrics,
            nodes=nodes,
            edges=edges,
            adjacency_list=dict(data.get("adjacency_list", {})),
        )


@dataclass
class CurriculumConcept:
    """Concept entry inside a pedagogical curriculum tier."""

    concept_id: str
    name: str
    category: str
    confidence: float = 1.0
    why_needed: str = ""
    prerequisites: List[str] = field(default_factory=list)
    is_target_paper_contribution: bool = False
    cso_topic_id: Optional[str] = None
    cso_uri: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return _to_serializable(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CurriculumConcept:
        return cls(
            concept_id=str(data.get("concept_id", "")),
            name=str(data.get("name", "")),
            category=str(data.get("category", "")),
            confidence=float(data.get("confidence", 1.0)),
            why_needed=str(data.get("why_needed", "")),
            prerequisites=list(data.get("prerequisites", [])),
            is_target_paper_contribution=bool(data.get("is_target_paper_contribution", False)),
            cso_topic_id=data.get("cso_topic_id"),
            cso_uri=data.get("cso_uri"),
        )


@dataclass
class CurriculumTier:
    """A pedagogical tier in the learning sequence (Tiers 1 to 4)."""

    tier: int
    tier_name: str
    concepts: List[CurriculumConcept] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _to_serializable(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CurriculumTier:
        concepts = [CurriculumConcept.from_dict(c) for c in data.get("concepts", [])]
        return cls(
            tier=int(data.get("tier", 1)),
            tier_name=str(data.get("tier_name", "")),
            concepts=concepts,
        )


@dataclass
class CurriculumPlan:
    """Full pedagogical curriculum plan for an academic paper."""

    paper_id: str
    paper_title: str
    curriculum: List[CurriculumTier] = field(default_factory=list)
    total_prerequisites: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return _to_serializable(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CurriculumPlan:
        curriculum = [CurriculumTier.from_dict(t) for t in data.get("curriculum", [])]
        return cls(
            paper_id=str(data.get("paper_id", "")),
            paper_title=str(data.get("paper_title", "")),
            curriculum=curriculum,
            total_prerequisites=int(data.get("total_prerequisites", 0)),
        )


@dataclass
class EvidenceRecord:
    """Verifiable textual and mathematical grounding record for a prerequisite relation."""

    prerequisite_concept: str
    dependent_concept: str
    evidence_type: str
    section: str
    quote: str
    formula_reference: Optional[str] = None
    cited_foundational_paper: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return _to_serializable(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EvidenceRecord:
        return cls(
            prerequisite_concept=str(data.get("prerequisite_concept", "")),
            dependent_concept=str(data.get("dependent_concept", "")),
            evidence_type=str(data.get("evidence_type", "")),
            section=str(data.get("section", "")),
            quote=str(data.get("quote", "")),
            formula_reference=data.get("formula_reference"),
            cited_foundational_paper=data.get("cited_foundational_paper"),
        )
