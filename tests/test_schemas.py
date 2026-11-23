"""Unit tests for schemas and data models (tests/test_schemas.py).

Verifies serialization, deserialization, and integrity of typed dataclasses:
- ConceptCategory, Formula, Section, DocumentIR
- ConceptEntity (including CSO ontology attributes)
- PrerequisiteEdge, GraphMetrics, PrerequisiteDAG
- CurriculumConcept, CurriculumTier, CurriculumPlan
- EvidenceRecord
"""

from __future__ import annotations

import json
import unittest

from src.models.schemas import (
    ConceptCategory,
    ConceptEntity,
    CurriculumConcept,
    CurriculumPlan,
    CurriculumTier,
    DocumentIR,
    EvidenceRecord,
    Formula,
    GraphMetrics,
    PrerequisiteDAG,
    PrerequisiteEdge,
    Section,
)


class TestSchemas(unittest.TestCase):
    """Test suite for all schema dataclasses and JSON serialization."""

    def test_concept_category(self):
        """Verify 4-way taxonomic enum parsing and fallbacks."""
        self.assertEqual(ConceptCategory.from_str("mathematics"), ConceptCategory.MATHEMATICS)
        self.assertEqual(ConceptCategory.from_str("ALGORITHMS"), ConceptCategory.ALGORITHMS)
        self.assertEqual(ConceptCategory.from_str("Theories"), ConceptCategory.THEORIES)
        self.assertEqual(ConceptCategory.from_str("methods"), ConceptCategory.METHODS)
        self.assertEqual(ConceptCategory.from_str("nonexistent"), ConceptCategory.METHODS)

    def test_formula_serialization(self):
        """Verify Formula serialization, latex parsing, and deserialization."""
        f = Formula(formula_id="eq1", raw_text="y = Hx + n", latex="y = Hx + n", section_id="sec2")
        d = f.to_dict()
        self.assertEqual(d["formula_id"], "eq1")
        self.assertEqual(d["latex"], "y = Hx + n")

        j = f.to_json()
        self.assertIn("eq1", j)
        f_restored = Formula.from_dict(json.loads(j))
        self.assertEqual(f, f_restored)

    def test_section_and_document_ir(self):
        """Verify Section and DocumentIR hierarchy and bibliographic references."""
        formula = Formula(formula_id="f1", raw_text="E = mc^2", latex="E=mc^2", section_id="sec1")
        sec = Section(
            section_id="sec1",
            heading="1. Introduction",
            category="Introduction",
            paragraphs=["Paragraph discussing foundational methods."],
            formulas=[formula],
            citations=["b0"],
        )
        doc = DocumentIR(
            paper_id="paper_001",
            title="Advanced Signal Processing",
            abstract="Abstract overview of signal detection.",
            sections=[sec],
            preliminaries=["Linear Algebra"],
            citations={"b0": {"title": "Foundations", "authors": ["Shannon"], "date": "1948"}},
        )

        d = doc.to_dict()
        self.assertEqual(d["paper_id"], "paper_001")
        self.assertEqual(len(d["sections"]), 1)
        self.assertEqual(d["sections"][0]["formulas"][0]["formula_id"], "f1")

        doc_restored = DocumentIR.from_dict(json.loads(doc.to_json()))
        self.assertEqual(doc_restored.paper_id, "paper_001")
        self.assertEqual(doc_restored.sections[0].heading, "1. Introduction")
        self.assertEqual(len(doc_restored.sections[0].formulas), 1)

    def test_concept_entity_with_cso_attributes(self):
        """Verify ConceptEntity serialization including CSO ontology identity."""
        concept = ConceptEntity(
            concept_id="c_op",
            name="Oblique Projection",
            category=ConceptCategory.MATHEMATICS,
            confidence=0.95,
            source_sections=["sec1", "sec2"],
            aliases=["OP", "Oblique Projector"],
            is_target_paper_contribution=False,
            cso_topic_id="projection",
            cso_uri="https://cso.kmi.open.ac.uk/topics/projection",
        )
        d = concept.to_dict()
        self.assertEqual(d["category"], "Mathematics")
        self.assertEqual(d["cso_topic_id"], "projection")
        self.assertEqual(d["cso_uri"], "https://cso.kmi.open.ac.uk/topics/projection")

        restored = ConceptEntity.from_dict(d)
        self.assertEqual(restored.category, ConceptCategory.MATHEMATICS)
        self.assertEqual(restored.cso_topic_id, "projection")
        self.assertEqual(restored.name, "Oblique Projection")

    def test_prerequisite_edge_and_dag(self):
        """Verify PrerequisiteEdge, GraphMetrics, and PrerequisiteDAG integrity."""
        edge = PrerequisiteEdge(
            source_id="c1",
            source_name="Linear Algebra",
            target_id="c2",
            target_name="Oblique Projection",
            relation_type="requires_prerequisite",
            confidence=0.98,
            asymmetry_score=0.85,
            epistemic_rationale="Linear algebra is required to define subspaces.",
        )
        metrics = GraphMetrics(
            num_nodes=2,
            num_edges=1,
            is_dag=True,
            cycles_detected_and_resolved=0,
            density=0.5,
        )
        c1 = ConceptEntity(concept_id="c1", name="Linear Algebra", category=ConceptCategory.MATHEMATICS)
        c2 = ConceptEntity(concept_id="c2", name="Oblique Projection", category=ConceptCategory.MATHEMATICS)
        dag = PrerequisiteDAG(
            paper_id="paper_001",
            metrics=metrics,
            nodes=[c1, c2],
            edges=[edge],
            adjacency_list={"c1": ["c2"], "c2": []},
        )

        d = dag.to_dict()
        self.assertTrue(d["metrics"]["is_dag"])
        self.assertEqual(len(d["edges"]), 1)

        dag_restored = PrerequisiteDAG.from_dict(json.loads(dag.to_json()))
        self.assertTrue(dag_restored.metrics.is_dag)
        self.assertEqual(dag_restored.edges[0].source_name, "Linear Algebra")

    def test_curriculum_plan_and_tiers(self):
        """Verify CurriculumConcept, CurriculumTier, and CurriculumPlan serialization."""
        cc1 = CurriculumConcept(
            concept_id="c1",
            name="Subspaces",
            category="Mathematics",
            confidence=0.9,
            why_needed="Underpins projection theory",
            prerequisites=[],
            is_target_paper_contribution=False,
            cso_topic_id="subspaces",
            cso_uri="https://cso.kmi.open.ac.uk/topics/subspaces",
        )
        tier1 = CurriculumTier(
            tier=1,
            tier_name="Tier 1: Foundational Mathematical & Signal Preliminaries",
            concepts=[cc1],
        )
        plan = CurriculumPlan(
            paper_id="paper_001",
            paper_title="Sample Research Paper",
            curriculum=[tier1],
            total_prerequisites=1,
        )

        d = plan.to_dict()
        self.assertEqual(len(d["curriculum"]), 1)
        self.assertEqual(d["curriculum"][0]["concepts"][0]["cso_topic_id"], "subspaces")

        plan_restored = CurriculumPlan.from_dict(json.loads(plan.to_json()))
        self.assertEqual(plan_restored.paper_title, "Sample Research Paper")
        self.assertEqual(plan_restored.curriculum[0].concepts[0].name, "Subspaces")

    def test_evidence_record(self):
        """Verify EvidenceRecord verbatim quotes and citation groundings."""
        ev = EvidenceRecord(
            prerequisite_concept="Oblique Projection",
            dependent_concept="Jamming Cancellation",
            evidence_type="formula_and_text",
            section="Section 3: Methodology",
            quote="Jamming cancellation is achieved via oblique projection operator E_sr.",
            formula_reference="Formula (3)",
            cited_foundational_paper="Behrens and Scharf, 1994",
        )
        d = ev.to_dict()
        self.assertEqual(d["formula_reference"], "Formula (3)")
        self.assertEqual(d["evidence_type"], "formula_and_text")

        ev_restored = EvidenceRecord.from_dict(d)
        self.assertEqual(ev_restored.quote, ev.quote)
        self.assertEqual(ev_restored.cited_foundational_paper, ev.cited_foundational_paper)


if __name__ == "__main__":
    unittest.main()
