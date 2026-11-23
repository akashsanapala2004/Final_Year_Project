"""Comprehensive Refactored Test Suite for CLI Tools and Utilities (tests/test_tools_and_cli.py).

Covers:
- TestGraphViz: Mermaid diagram export and styling
- TestPrereqClassifier: Pairwise logistic regression trainer, feature vectors, checkpointing
- TestCLIExecution:
  - run.py CLI: --list, --input, --all, error handling
  - query.py CLI: --category, --concept, --details, --cso
  - batch_train.py CLI: parallel multi-core execution, automatic resume, catalog compilation
"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

from src.models.schemas import (
    ConceptCategory,
    ConceptEntity,
    GraphMetrics,
    PrerequisiteDAG,
    PrerequisiteEdge,
)
from src.models.train_prereq_classifier import PrereqPairClassifier, train_from_corpus
from src.utils.graph_viz import MermaidGraphExporter


class TestGraphViz(unittest.TestCase):
    """Mermaid Flowchart Generator and Visualizer Tests."""

    def test_mermaid_export(self):
        """Verify Mermaid syntax generation, color classes, and node sanitization."""
        c1 = ConceptEntity(concept_id="c_math", name="Linear Algebra", category=ConceptCategory.MATHEMATICS)
        c2 = ConceptEntity(concept_id="c_algo", name="GLRT", category=ConceptCategory.ALGORITHMS)
        edge = PrerequisiteEdge(
            source_id="c_math",
            source_name="Linear Algebra",
            target_id="c_algo",
            target_name="GLRT",
            asymmetry_score=0.85,
        )

        dag = PrerequisiteDAG(
            paper_id="test_dag",
            metrics=GraphMetrics(num_nodes=2, num_edges=1, is_dag=True),
            nodes=[c1, c2],
            edges=[edge],
            adjacency_list={"c_math": ["c_algo"], "c_algo": []},
        )

        exporter = MermaidGraphExporter()
        mmd = exporter.export(dag)

        self.assertTrue(mmd.startswith("flowchart TD"))
        self.assertIn("node_c_math", mmd)
        self.assertIn("node_c_algo", mmd)
        self.assertIn("classDef tier1", mmd)
        self.assertIn("-->", mmd)

    def test_mermaid_save(self):
        """Verify saving Mermaid string to physical file."""
        exporter = MermaidGraphExporter()
        dag = PrerequisiteDAG(
            paper_id="test_save",
            metrics=GraphMetrics(num_nodes=0, num_edges=0, is_dag=True),
            nodes=[],
            edges=[],
            adjacency_list={},
        )
        test_file = Path("output/temp_test_graph.mmd")
        self.addCleanup(lambda: test_file.unlink(missing_ok=True))

        saved_path = exporter.save(dag, None, test_file)
        self.assertTrue(saved_path.is_file())


class TestPrereqClassifier(unittest.TestCase):
    """Pairwise Prerequisite Classifier Trainer Tests."""

    def test_feature_extraction(self):
        """Verify feature vector normalization."""
        clf = PrereqPairClassifier()
        feats = clf.extract_features(
            src_tier=1.0,
            tgt_tier=4.0,
            src_is_contrib=False,
            tgt_is_contrib=True,
            src_pos=0,
            tgt_pos=3,
            src_name="Linear Algebra",
            tgt_name="Jamming Cancellation",
            ref_d=0.8,
            co_occurrence=4,
        )
        self.assertEqual(len(feats), 6)
        self.assertEqual(feats[0], 3.0)  # tax_delta: 4.0 - 1.0
        self.assertEqual(feats[1], 1.0)  # contrib_bias: +1.0

    def test_classifier_training_and_persistence(self):
        """Verify stochastic gradient descent training, metrics, and JSON serialization."""
        dataset = [
            {"src_tier": 1.0, "tgt_tier": 4.0, "src_is_contrib": False, "tgt_is_contrib": True, "src_pos": 0, "tgt_pos": 3, "src_name": "Linear Algebra", "tgt_name": "Jamming Cancellation", "ref_d": 0.8, "co_occurrence": 4, "label": 1},
            {"src_tier": 1.0, "tgt_tier": 2.0, "src_is_contrib": False, "tgt_is_contrib": False, "src_pos": 0, "tgt_pos": 1, "src_name": "Subspaces", "tgt_name": "Oblique Projection", "ref_d": 0.7, "co_occurrence": 3, "label": 1},
            {"src_tier": 4.0, "tgt_tier": 1.0, "src_is_contrib": True, "tgt_is_contrib": False, "src_pos": 3, "tgt_pos": 0, "src_name": "Jamming Cancellation", "tgt_name": "Subspaces", "ref_d": -0.8, "co_occurrence": 4, "label": 0},
            {"src_tier": 3.0, "tgt_tier": 1.0, "src_is_contrib": False, "tgt_is_contrib": False, "src_pos": 2, "tgt_pos": 0, "src_name": "Spectrum Sensing", "tgt_name": "Linear Algebra", "ref_d": -0.5, "co_occurrence": 1, "label": 0},
        ]

        checkpoint_file = Path("models/checkpoints/test_checkpoint.json")
        self.addCleanup(lambda: checkpoint_file.unlink(missing_ok=True))

        clf = train_from_corpus(dataset, checkpoint_file)
        self.assertTrue(checkpoint_file.is_file())
        self.assertGreaterEqual(clf.metrics["accuracy"], 0.75)

        # Load trained weights
        loaded_clf = PrereqPairClassifier(checkpoint_file)
        self.assertEqual(len(loaded_clf.weights), 6)

        # Predict probability
        feats_pos = loaded_clf.extract_features(1.0, 4.0, False, True, 0, 3, "Linear Algebra", "Jamming Cancellation", 0.8, 4)
        prob_pos = loaded_clf.predict_probability(feats_pos)
        self.assertGreater(prob_pos, 0.5)

        feats_neg = loaded_clf.extract_features(4.0, 1.0, True, False, 3, 0, "Jamming Cancellation", "Linear Algebra", -0.8, 4)
        prob_neg = loaded_clf.predict_probability(feats_neg)
        self.assertLess(prob_neg, 0.5)


class TestCLIExecution(unittest.TestCase):
    """End-to-End CLI Invocation Tests."""

    @classmethod
    def setUpClass(cls):
        # Pre-process all papers so query catalog tests have populated data
        subprocess.run([sys.executable, "run.py", "--all"], capture_output=True, text=True)

    def test_cli_convert_pdfs(self):
        """Verify python3 run.py --convert-pdfs converts raw PDFs in data/raw_pdfs."""
        from src.utils.pdf_grobid_converter import generate_valid_pdf

        temp_pdf = Path("data/raw_pdfs/cli_test_paper.pdf")
        self.addCleanup(lambda: temp_pdf.unlink(missing_ok=True))
        temp_tei = Path("data/grobid_output/cli_test_paper.tei.xml")
        self.addCleanup(lambda: temp_tei.unlink(missing_ok=True))

        generate_valid_pdf(
            title="Convex Optimization and Lagrangian Duality",
            sections=[("1. Introduction", ["Convex optimization guarantees unique global minima."])],
            output_path=temp_pdf,
        )

        res = subprocess.run([sys.executable, "run.py", "--convert-pdfs"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("CONVERTING RAW PDFS", res.stdout)
        self.assertTrue(temp_tei.is_file())

    def test_cli_run_list(self):
        """Verify python3 run.py --list reports paper library status."""
        res = subprocess.run([sys.executable, "run.py", "--list"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("AVAILABLE PAPERS IN LIBRARY", res.stdout)
        self.assertIn("paper_opjc", res.stdout)
        self.assertIn("Processed", res.stdout)

    def test_cli_run_single_input_success(self):
        """Verify python3 run.py --input <id> executes 6-stage pipeline."""
        res = subprocess.run([sys.executable, "run.py", "--input", "paper_opjc"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("SCIENTIFIC PREREQUISITE CURRICULUM PIPELINE", res.stdout)
        self.assertIn("Cognitive Radio", res.stdout)
        self.assertIn("Strict DAG = True", res.stdout)
        self.assertIn("ARTIFACTS SAVED", res.stdout)

    def test_cli_run_single_input_nonexistent(self):
        """Verify python3 run.py --input <nonexistent> exits with returncode 1."""
        res = subprocess.run([sys.executable, "run.py", "--input", "invalid_nonexistent_paper"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 1)

    def test_cli_run_all(self):
        """Verify python3 run.py --all processes entire batch."""
        res = subprocess.run([sys.executable, "run.py", "--all"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("BATCH PROCESSING ALL", res.stdout)
        self.assertIn("COMPLETED", res.stdout)

    def test_cli_query_category(self):
        """Verify python3 query.py --category Mathematics finds matching concepts."""
        res = subprocess.run([sys.executable, "query.py", "--category", "Mathematics"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("SEARCH RESULTS FOR CATEGORY: 'Mathematics'", res.stdout)
        self.assertIn("Linear Algebra", res.stdout)

    def test_cli_query_concept_with_details(self):
        """Verify python3 query.py --concept <name> --details outputs grounded evidence quotes."""
        res = subprocess.run([sys.executable, "query.py", "--concept", "GLRT", "--details"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("SEARCH RESULTS FOR CONCEPT: 'GLRT'", res.stdout)
        self.assertIn("Grounded Textual Evidence & Formulas:", res.stdout)

    def test_cli_query_cso_ontology(self):
        """Verify python3 query.py --cso <term> directly queries CSO.3.3.owl."""
        res = subprocess.run([sys.executable, "query.py", "--cso", "GLRT"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("CSO ONTOLOGY IDENTIFICATION & CLASSIFICATION", res.stdout)
        self.assertIn("Algorithms", res.stdout)
        self.assertIn("cso.kmi.open.ac.uk", res.stdout)

    def test_cli_batch_train(self):
        """Verify python3 batch_train.py runs parallel execution and compiles corpus_catalog.json."""
        res = subprocess.run([sys.executable, "batch_train.py", "--limit", "3"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("STARTING BATCH TRAINING PIPELINE", res.stdout)
        self.assertIn("BATCH RUN COMPLETED", res.stdout)

        catalog_file = Path("output/corpus_catalog.json")
        self.assertTrue(catalog_file.is_file())
        catalog_data = json.loads(catalog_file.read_text(encoding="utf-8"))
        self.assertIn("total_papers", catalog_data)
        self.assertIn("classifier_evaluation", catalog_data)
        self.assertIn("concept_index", catalog_data)


if __name__ == "__main__":
    unittest.main()
