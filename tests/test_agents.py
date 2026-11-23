"""Comprehensive Refactored Test Suite for Multi-Agent Pipeline (tests/test_agents.py).

Covers all 6 agents and foundational algorithms:
- TestPaperParserAgent: GROBID TEI-XML, formulas, citations, copyright boilerplate rejection
- TestCSOOntology: Computer Science Ontology (CSO.3.3.owl) identification & classification
- TestConceptExtractorAgent: 4-way taxonomy, formula operators, dynamic acronyms, contribution rules
- TestPrerequisiteRelationAgent: Asymmetric RefD, iPRL taxonomic delta, lexical subsumption, contribution bias
- TestGraphReasoningAgent: DFS cycle detection, Greedy FAS cycle elimination, Transitive Reduction
- TestCurriculumPlannerAgent: Kahn's topological sort with category priority, DAG Tier Monotonicity
- TestEvidenceGroundingAgent: Verbatim quotes, formula references, citation linking, pipe sanitization
- TestPipelineOrchestrator: Complete end-to-end multi-agent orchestration and report generation
"""

from __future__ import annotations

import unittest
from pathlib import Path

from src.agents.concept_extractor_agent import ConceptExtractorAgent
from src.agents.curriculum_planner_agent import CurriculumPlannerAgent
from src.agents.evidence_grounding_agent import EvidenceGroundingAgent
from src.agents.graph_reasoning_agent import GraphReasoningAgent
from src.agents.orchestrator import PipelineOrchestrator
from src.agents.paper_parser_agent import PaperParserAgent, is_copyright_boilerplate
from src.agents.prerequisite_relation_agent import PrerequisiteRelationAgent
from src.models.schemas import (
    ConceptCategory,
    ConceptEntity,
    DocumentIR,
    Formula,
    GraphMetrics,
    PrerequisiteDAG,
    PrerequisiteEdge,
    Section,
)
from src.utils.cso_classifier import CSOOntology, classify_cso_concept, get_cso_ontology, identify_cso_concept


class TestPaperParserAgent(unittest.TestCase):
    """Stage 1: Document & Formula Parsing Tests."""

    def setUp(self):
        self.parser = PaperParserAgent(data_dir="data")

    def test_copyright_boilerplate_rejection(self):
        """Anti-Pattern #4: Rejection of copyright disclaimers in titles."""
        copyright_samples = [
            "Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee.",
            "Copyright (c) 2021 IEEE Transactions on Signal Processing. All rights reserved.",
            "ACM Reference Format: Author et al. Proceedings of the 2020 Conference.",
            "This paper is published under Creative Commons Attribution License.",
        ]
        for sample in copyright_samples:
            self.assertTrue(is_copyright_boilerplate(sample), f"Failed to reject copyright text: {sample}")

        valid_titles = [
            "Cognitive Radio Spectrum Sensing and Jamming Cancellation via Oblique Projection and GLRT",
            "Hypergraph Reduction via Hypertree Decomposition and Case-Based Reasoning",
            "Physical Layer Security in Multicarrier OFDM Systems using Convex Optimization",
        ]
        for title in valid_titles:
            self.assertFalse(is_copyright_boilerplate(title), f"False positive on valid title: {title}")

    def test_parse_tei_xml_metadata_and_sections(self):
        """Verify TEI-XML title extraction (fallback past copyright note), sections, and abstract."""
        doc = self.parser.parse("paper_opjc")
        self.assertEqual(doc.paper_id, "paper_opjc")
        self.assertIn("Cognitive Radio Spectrum Sensing", doc.title)
        self.assertNotIn("Permission to make digital", doc.title)
        self.assertTrue(len(doc.abstract) > 30)
        self.assertTrue(len(doc.sections) >= 4)

    def test_formula_parsing_and_association(self):
        """Verify explicit extraction and section linking of <tei:formula> elements."""
        doc = self.parser.parse("paper_opjc")
        all_formulas = [f for s in doc.sections for f in s.formulas]
        self.assertTrue(len(all_formulas) >= 3, "Formulas were not properly parsed from TEI-XML!")

        # Verify formula IDs, raw text, and latex attributes
        formula_ids = {f.formula_id for f in all_formulas}
        self.assertIn("formula_1", formula_ids)
        self.assertIn("formula_2", formula_ids)

        # Check operator content in formulas
        f2 = next(f for f in all_formulas if f.formula_id == "formula_2")
        self.assertIn("E_{sr}", f2.latex)

    def test_bibliography_extraction(self):
        """Verify bibliography listBibl extraction and in-text citation linking."""
        doc = self.parser.parse("paper_opjc")
        self.assertIn("b0", doc.citations)
        self.assertIn("b1", doc.citations)
        self.assertEqual(doc.citations["b0"]["authors"][0], "J. Mitola")
        self.assertEqual(doc.citations["b1"]["authors"][0], "S. M. Kay")


class TestCSOOntology(unittest.TestCase):
    """Computer Science Ontology (CSO.3.3.owl) Verification Tests."""

    @classmethod
    def setUpClass(cls):
        cls.cso = get_cso_ontology()

    def test_ontology_cardinality(self):
        """Verify CSO ontology loaded with >14,000 scientific concepts."""
        self.assertGreater(len(self.cso.topics), 14000)
        self.assertGreater(len(self.cso.label_to_topic_id), 14000)

    def test_concept_identification(self):
        """Verify accurate topic identification and URI resolution."""
        topic_glrt = self.cso.identify("GLRT")
        self.assertIsNotNone(topic_glrt)
        self.assertIn("glrt", topic_glrt.topic_id.lower())
        self.assertEqual(topic_glrt.category, ConceptCategory.ALGORITHMS)
        self.assertTrue(topic_glrt.uri.startswith("https://cso.kmi.open.ac.uk/topics/"))

        topic_fourier = self.cso.identify("Fourier Transform")
        self.assertIsNotNone(topic_fourier)
        self.assertEqual(topic_fourier.category, ConceptCategory.MATHEMATICS)

        topic_sec = self.cso.identify("Physical Layer Security")
        self.assertIsNotNone(topic_sec)
        self.assertEqual(topic_sec.category, ConceptCategory.THEORIES)

    def test_ontology_classification(self):
        """Verify 4-way classification across domain areas."""
        self.assertEqual(classify_cso_concept("discrete fourier transform"), ConceptCategory.MATHEMATICS)
        self.assertEqual(classify_cso_concept("genetic algorithms"), ConceptCategory.ALGORITHMS)
        self.assertEqual(classify_cso_concept("information theory"), ConceptCategory.THEORIES)
        self.assertEqual(classify_cso_concept("case-based reasoning"), ConceptCategory.METHODS)

    def test_n_gram_text_extraction(self):
        """Verify sliding window n-gram extraction from continuous text."""
        sample = "We apply discrete fourier transform and genetic algorithms to optimize cognitive radio networks."
        extracted = self.cso.extract_from_text(sample)
        labels = {t.label.lower() for t in extracted}
        self.assertTrue(any("fourier" in l for l in labels))
        self.assertTrue(any("genetic" in l for l in labels))
        self.assertTrue(any("cognitive radio" in l for l in labels))


class TestConceptExtractorAgent(unittest.TestCase):
    """Stage 2: 4-Way Taxonomy Concept Extraction Tests."""

    def setUp(self):
        self.parser = PaperParserAgent(data_dir="data")
        self.extractor = ConceptExtractorAgent()

    def test_4way_taxonomy_extraction(self):
        """Verify extraction across Mathematics, Algorithms, Theories, and Methods."""
        doc = self.parser.parse("paper_opjc")
        concepts = self.extractor.extract_concepts(doc)
        concept_names = {c.name for c in concepts}
        category_map = {c.name: c.category for c in concepts}

        # Mathematics
        self.assertIn("Linear Algebra", concept_names)
        self.assertEqual(category_map["Linear Algebra"], ConceptCategory.MATHEMATICS)
        self.assertIn("Subspaces", concept_names)
        self.assertEqual(category_map["Subspaces"], ConceptCategory.MATHEMATICS)
        self.assertIn("Oblique Projection", concept_names)

        # Algorithms
        self.assertIn("GLRT", concept_names)
        self.assertEqual(category_map["GLRT"], ConceptCategory.ALGORITHMS)

        # Theories
        self.assertIn("Cognitive Radio", concept_names)
        self.assertEqual(category_map["Cognitive Radio"], ConceptCategory.THEORIES)
        self.assertIn("Spectrum Sensing", concept_names)

        # Methods
        self.assertIn("Jamming Cancellation", concept_names)
        self.assertEqual(category_map["Jamming Cancellation"], ConceptCategory.METHODS)

    def test_formula_operator_extraction(self):
        """Anti-Pattern #3: Mathematical formulas contain crucial foundational operators."""
        doc = self.parser.parse("paper_opjc")
        concepts = self.extractor.extract_concepts(doc)
        concept_by_name = {c.name: c for c in concepts}

        self.assertIn("Oblique Projection", concept_by_name)
        self.assertIn("Covariance Matrices", concept_by_name)
        # Check source section association
        self.assertTrue(len(concept_by_name["Oblique Projection"].source_sections) > 0)

    def test_contribution_rule(self):
        """Strict Rule: Foundational math and broad domain theories are NEVER marked as paper contributions."""
        doc = self.parser.parse("paper_opjc")
        concepts = self.extractor.extract_concepts(doc)
        for c in concepts:
            if c.category in (ConceptCategory.MATHEMATICS, ConceptCategory.THEORIES):
                self.assertFalse(
                    c.is_target_paper_contribution,
                    f"Violation: {c.name} in {c.category.value} was marked as target paper contribution!"
                )

    def test_cso_ontology_enrichment(self):
        """Verify extracted concepts are enriched with CSO Topic IDs and URIs."""
        doc = self.parser.parse("paper_opjc")
        concepts = self.extractor.extract_concepts(doc)
        cso_enriched = [c for c in concepts if c.cso_topic_id is not None]
        self.assertTrue(len(cso_enriched) >= 2, "Extracted concepts were not enriched with CSO ontology data!")
        glrt_concept = next(c for c in concepts if c.name == "GLRT")
        self.assertIsNotNone(glrt_concept.cso_topic_id)
        self.assertTrue(glrt_concept.cso_uri.startswith("https://cso.kmi.open.ac.uk/"))


class TestPrerequisiteRelationAgent(unittest.TestCase):
    """Stage 3: Asymmetric Prerequisite Learning Tests."""

    def setUp(self):
        self.agent = PrerequisiteRelationAgent()

    def test_contribution_bias_and_strict_direction(self):
        """Strict Rule: Target paper contributions can NEVER precede foundational external prerequisites."""
        c_math = ConceptEntity(concept_id="c_math", name="Linear Algebra", category=ConceptCategory.MATHEMATICS)
        c_contrib = ConceptEntity(concept_id="c_contrib", name="Jamming Cancellation", category=ConceptCategory.METHODS, is_target_paper_contribution=True)
        doc = DocumentIR(
            paper_id="test",
            title="Test Paper",
            abstract="",
            sections=[
                Section(
                    section_id="sec1",
                    heading="Methodology",
                    category="Methodology",
                    paragraphs=["Linear Algebra is required for Jamming Cancellation."],
                )
            ],
        )

        edges = self.agent.learn_prerequisites([c_math, c_contrib], doc)

        # Valid edge: Math -> Contribution
        self.assertTrue(any(e.source_name == "Linear Algebra" and e.target_name == "Jamming Cancellation" for e in edges))
        # Forbidden edge: Contribution -> Math MUST NEVER EXIST
        self.assertFalse(any(e.source_name == "Jamming Cancellation" and e.target_name == "Linear Algebra" for e in edges))

    def test_lexical_subsumption(self):
        """Pan et al. (2019): Lexical subsumption A in B implies A -> B."""
        c_base = ConceptEntity(concept_id="c_op", name="Oblique Projection", category=ConceptCategory.MATHEMATICS)
        c_comp = ConceptEntity(concept_id="c_opjc", name="Oblique Projection-Based Jamming Cancellation", category=ConceptCategory.METHODS, is_target_paper_contribution=True)
        doc = DocumentIR(paper_id="test", title="Test", abstract="", sections=[])

        edges = self.agent.learn_prerequisites([c_base, c_comp], doc)
        self.assertTrue(any(e.source_name == "Oblique Projection" and e.target_name == "Oblique Projection-Based Jamming Cancellation" for e in edges))

    def test_epistemic_rationale_generation(self):
        """Verify explainable epistemic rationales are synthesized for prerequisite edges."""
        c1 = ConceptEntity(concept_id="c1", name="Subspaces", category=ConceptCategory.MATHEMATICS)
        c2 = ConceptEntity(concept_id="c2", name="Oblique Projection", category=ConceptCategory.MATHEMATICS)
        doc = DocumentIR(paper_id="test", title="Test", abstract="", sections=[])

        edges = self.agent.learn_prerequisites([c1, c2], doc)
        self.assertTrue(len(edges) > 0)
        self.assertTrue(len(edges[0].epistemic_rationale) > 10)


class TestGraphReasoningAgent(unittest.TestCase):
    """Stage 4: Prerequisite KG, Greedy FAS, and Transitive Reduction Tests."""

    def setUp(self):
        self.agent = GraphReasoningAgent()

    def test_greedy_fas_cycle_elimination(self):
        """Verify directed cycle detection (3-color DFS) and elimination of lowest confidence edge."""
        nodes = [
            ConceptEntity(concept_id="A", name="Concept A", category=ConceptCategory.MATHEMATICS),
            ConceptEntity(concept_id="B", name="Concept B", category=ConceptCategory.ALGORITHMS),
            ConceptEntity(concept_id="C", name="Concept C", category=ConceptCategory.METHODS),
        ]
        edges = [
            PrerequisiteEdge(source_id="A", source_name="Concept A", target_id="B", target_name="Concept B", confidence=0.95),
            PrerequisiteEdge(source_id="B", source_name="Concept B", target_id="C", target_name="Concept C", confidence=0.85),
            PrerequisiteEdge(source_id="C", source_name="Concept C", target_id="A", target_name="Concept A", confidence=0.40),  # Weakest edge in cycle
        ]

        dag = self.agent.build_dag("test_cycle", nodes, edges)
        self.assertTrue(dag.metrics.is_dag, "Greedy FAS failed to produce a strict DAG!")
        self.assertEqual(dag.metrics.cycles_detected_and_resolved, 1)

        # C -> A should be excised; A -> B and B -> C remain intact
        active_pairs = {(e.source_id, e.target_id) for e in dag.edges}
        self.assertNotIn(("C", "A"), active_pairs)
        self.assertIn(("A", "B"), active_pairs)
        self.assertIn(("B", "C"), active_pairs)

    def test_transitive_reduction(self):
        """Verify removal of redundant bypass edges (u -> v when u -> ... -> v exists)."""
        nodes = [
            ConceptEntity(concept_id="A", name="Concept A", category=ConceptCategory.MATHEMATICS),
            ConceptEntity(concept_id="B", name="Concept B", category=ConceptCategory.ALGORITHMS),
            ConceptEntity(concept_id="C", name="Concept C", category=ConceptCategory.METHODS),
        ]
        edges = [
            PrerequisiteEdge(source_id="A", source_name="Concept A", target_id="B", target_name="Concept B", confidence=0.9),
            PrerequisiteEdge(source_id="B", source_name="Concept B", target_id="C", target_name="Concept C", confidence=0.8),
            PrerequisiteEdge(source_id="A", source_name="Concept A", target_id="C", target_name="Concept C", confidence=0.85),  # Bypass edge
        ]

        dag = self.agent.build_dag("test_transitive", nodes, edges)
        active_pairs = {(e.source_id, e.target_id) for e in dag.edges}
        self.assertNotIn(("A", "C"), active_pairs, "Transitive reduction failed to remove redundant bypass edge!")
        self.assertIn(("A", "B"), active_pairs)
        self.assertIn(("B", "C"), active_pairs)


class TestCurriculumPlannerAgent(unittest.TestCase):
    """Stage 5: Pedagogical Curriculum Planner & DAG Monotonicity Tests."""

    def setUp(self):
        self.planner = CurriculumPlannerAgent()

    def test_dag_topological_tier_monotonicity(self):
        """Anti-Pattern #5: DAG Topological Tier Monotonicity: for every edge u -> v, tier(v) >= tier(u)."""
        c1 = ConceptEntity(concept_id="c1", name="Subspaces", category=ConceptCategory.MATHEMATICS)
        c2 = ConceptEntity(concept_id="c2", name="Oblique Projection", category=ConceptCategory.MATHEMATICS)
        c3 = ConceptEntity(concept_id="c3", name="GLRT", category=ConceptCategory.ALGORITHMS)
        c4 = ConceptEntity(concept_id="c4", name="Jamming Cancellation", category=ConceptCategory.METHODS, is_target_paper_contribution=True)

        nodes = [c1, c2, c3, c4]
        edges = [
            PrerequisiteEdge(source_id="c1", source_name="Subspaces", target_id="c2", target_name="Oblique Projection"),
            PrerequisiteEdge(source_id="c2", source_name="Oblique Projection", target_id="c4", target_name="Jamming Cancellation"),
            PrerequisiteEdge(source_id="c1", source_name="Subspaces", target_id="c3", target_name="GLRT"),
        ]

        dag = PrerequisiteDAG(
            paper_id="test",
            metrics=GraphMetrics(num_nodes=4, num_edges=3, is_dag=True),
            nodes=nodes,
            edges=edges,
            adjacency_list={"c1": ["c2", "c3"], "c2": ["c4"], "c3": [], "c4": []},
        )

        plan = self.planner.plan_curriculum(dag, "Test Paper")

        # Map each concept to assigned tier (1 to 4)
        concept_tier_map = {}
        for tier in plan.curriculum:
            for c in tier.concepts:
                concept_tier_map[c.concept_id] = tier.tier

        # Strict Monotonicity Check across all edges
        for edge in dag.edges:
            src_tier = concept_tier_map[edge.source_id]
            tgt_tier = concept_tier_map[edge.target_id]
            self.assertGreaterEqual(
                tgt_tier,
                src_tier,
                f"Monotonicity Violated: {edge.source_name} (Tier {src_tier}) -> {edge.target_name} (Tier {tgt_tier})",
            )


class TestEvidenceGroundingAgent(unittest.TestCase):
    """Stage 6: Grounded Evidence Generation Tests."""

    def setUp(self):
        self.grounding_agent = EvidenceGroundingAgent()

    def test_pipe_sanitization_and_formula_linking(self):
        """Verify pipe '|' sanitization in section headings to prevent broken Markdown tables."""
        doc = DocumentIR(
            paper_id="test",
            title="Test Paper",
            abstract="",
            sections=[
                Section(
                    section_id="sec1",
                    heading="System Model | Subspace Formulation | Signal Subspaces",  # Contains pipes
                    category="Preliminaries",
                    paragraphs=["The Oblique Projection operator removes interference for Jamming Cancellation."],
                    formulas=[Formula(formula_id="f1", raw_text="E_sr = S(S^H P S)^-1", latex="E_sr", section_id="sec1")],
                )
            ],
        )
        edge = PrerequisiteEdge(
            source_id="op",
            source_name="Oblique Projection",
            target_id="jc",
            target_name="Jamming Cancellation",
        )

        records = self.grounding_agent.ground_prerequisites([edge], doc)
        self.assertEqual(len(records), 1)
        rec = records[0]

        # Verify pipe '|' was sanitized
        self.assertNotIn("|", rec.section)
        self.assertIn("-", rec.section)
        self.assertIn("Oblique Projection operator removes interference", rec.quote)


class TestPipelineOrchestrator(unittest.TestCase):
    """Stage 7: End-to-End Orchestrator Pipeline Tests."""

    def setUp(self):
        self.orchestrator = PipelineOrchestrator(data_dir="data", output_dir="output")

    def test_end_to_end_pipeline(self):
        """Execute full 6-stage pipeline and verify artifacts."""
        res = self.orchestrator.run_pipeline("paper_opjc", save_artifacts=True)
        self.assertEqual(res["paper_id"], "paper_opjc")
        self.assertTrue(len(res["concepts"]) >= 4)
        self.assertTrue(res["dag"].metrics.is_dag)
        self.assertEqual(len(res["curriculum"].curriculum), 4)

        paths = res["paths"]
        self.assertTrue(paths["json"].is_file())
        self.assertTrue(paths["markdown"].is_file())
        self.assertTrue(paths["mermaid"].is_file())


if __name__ == "__main__":
    unittest.main()
