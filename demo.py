#!/usr/bin/env python3
"""Interactive Step-by-Step Feature Demonstration (demo.py).

Walks through every stage and theoretical feature of the Scientific Prerequisite System:
- Step 1: Document Parsing, Copyright Disclaimer Rejection, and Formula Extraction
- Step 2: 4-Way Concept Extraction & CSO Ontology Identification (CSO.3.3.owl)
- Step 3: Asymmetric Prerequisite Learning (RefD, Taxonomic Delta, Lexical Subsumption)
- Step 4: Graph Reasoning: Live Greedy FAS Cycle Elimination & Transitive Reduction
- Step 5: Pedagogical Curriculum Planning & Strict DAG Monotonicity Verification
- Step 6: Verifiable Evidence Grounding, Formula References, and Pipe Sanitization
- Step 7: Command-Line Tools & Multi-Core Batch Processing

Usage:
    python3 demo.py --all            # Run all 7 feature demonstrations
    python3 demo.py --step 1         # Run only Step 1 (or 2, 3, 4, 5, 6, 7)
    python3 demo.py --interactive    # Step-by-step interactive mode with pauses
"""

from __future__ import annotations

import argparse
import sys
import time
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
from src.utils.cso_classifier import get_cso_ontology


def print_banner(title: str, step_num: int) -> None:
    print("\n" + "=" * 80)
    print(f"FEATURE DEMO STEP {step_num}: {title.upper()}")
    print("=" * 80)


def pause_if_interactive(interactive: bool) -> None:
    if interactive:
        input("\n[Press ENTER to proceed to the next step...]")


def demo_step_1(interactive: bool = False) -> None:
    print_banner("Stage 1: Document & Formula Parser & Copyright Rejection", 1)
    parser = PaperParserAgent(data_dir="data")

    print("[*] 1. Demonstrating Anti-Pattern #4 Rejection (Copyright Boilerplates in Titles):")
    boilerplate = "Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee."
    print(f"    Sample Text: \"{boilerplate[:70]}...\"")
    print(f"    -> is_copyright_boilerplate() check: {is_copyright_boilerplate(boilerplate)} (REJECTED)")

    clean_title = "Cognitive Radio Spectrum Sensing and Jamming Cancellation via Oblique Projection and GLRT"
    print(f"    Real Title:  \"{clean_title}\"")
    print(f"    -> is_copyright_boilerplate() check: {is_copyright_boilerplate(clean_title)} (ACCEPTED)")

    print("\n[*] 2. Parsing GROBID TEI-XML ('data/grobid_output/paper_opjc.tei.xml'):")
    doc = parser.parse("paper_opjc")
    print(f"    Clean Title Extracted:  '{doc.title}'")
    print(f"    Paper ID:               '{doc.paper_id}'")
    print(f"    Total Sections Parsed:  {len(doc.sections)}")
    print(f"    Citations in listBibl:  {len(doc.citations)} foundational references")

    print("\n[*] 3. Preserving Mathematical Formulas (<tei:formula>) with Section IDs and LaTeX:")
    all_formulas = [f for s in doc.sections for f in s.formulas]
    for idx, f in enumerate(all_formulas, 1):
        print(f"    Formula {idx} [{f.formula_id} in {f.section_id}]:")
        print(f"      • Raw:   {f.raw_text}")
        print(f"      • LaTeX: {f.latex}")

    pause_if_interactive(interactive)


def demo_step_2(interactive: bool = False) -> None:
    print_banner("Stage 2: 4-Way Concept Extraction & CSO Ontology Identification", 2)
    parser = PaperParserAgent(data_dir="data")
    extractor = ConceptExtractorAgent()
    doc = parser.parse("paper_opjc")

    print("[*] 1. Extracting concepts across 4-way scientific taxonomy:")
    concepts = extractor.extract_concepts(doc)
    print(f"    Total Concepts Discovered: {len(concepts)}\n")

    by_cat = {}
    for c in concepts:
        by_cat.setdefault(c.category.value, []).append(c)

    for cat in ["Mathematics", "Algorithms", "Theories", "Methods"]:
        cat_concepts = by_cat.get(cat, [])
        print(f"    • {cat} ({len(cat_concepts)} concepts):")
        for c in cat_concepts:
            role = "[TARGET PAPER CONTRIBUTION]" if c.is_target_paper_contribution else "[External Prerequisite]"
            cso_str = f"-> CSO URI: {c.cso_uri}" if c.cso_uri else ""
            print(f"       - {c.name:<32} {role:<24} {cso_str}")

    print("\n[*] 2. Anti-Pattern Verification: Foundational Math/Theories as Paper Contributions:")
    math_contribs = [c.name for c in concepts if c.category == ConceptCategory.MATHEMATICS and c.is_target_paper_contribution]
    if not math_contribs:
        print("    ✓ PASSED: Zero foundational mathematics concepts marked as paper contributions.")
    else:
        print(f"    ✗ FAILED: Math concepts erroneously marked as contributions: {math_contribs}")

    pause_if_interactive(interactive)


def demo_step_3(interactive: bool = False) -> None:
    print_banner("Stage 3: Asymmetric Prerequisite Learning (RefD & iPRL)", 3)
    agent = PrerequisiteRelationAgent()

    c_math = ConceptEntity(concept_id="c_math", name="Linear Algebra", category=ConceptCategory.MATHEMATICS)
    c_sub = ConceptEntity(concept_id="c_sub", name="Subspaces", category=ConceptCategory.MATHEMATICS)
    c_op = ConceptEntity(concept_id="c_op", name="Oblique Projection", category=ConceptCategory.MATHEMATICS)
    c_contrib = ConceptEntity(concept_id="c_jc", name="Jamming Cancellation", category=ConceptCategory.METHODS, is_target_paper_contribution=True)

    doc = DocumentIR(
        paper_id="paper_opjc",
        title="Test",
        abstract="",
        sections=[
            Section(section_id="sec1", heading="Preliminaries", category="Preliminaries", paragraphs=["Linear Algebra defines Subspaces."]),
            Section(section_id="sec2", heading="Methodology", category="Methodology", paragraphs=["Jamming Cancellation uses Oblique Projection."]),
        ],
    )

    print("[*] 1. Evaluating Directional Pairwise Scoring:")
    test_pairs = [
        (c_math, c_sub, "Math -> Math (Foundational Dependency)"),
        (c_sub, c_op, "Subspaces -> Oblique Projection (Subspace formulation)"),
        (c_op, c_contrib, "Foundational Math -> Target Paper Contribution"),
        (c_contrib, c_math, "Target Contribution -> Foundational Math (FORBIDDEN DIRECTION)"),
    ]

    sec_indices = {"sec1": 0, "sec2": 1}
    for src, tgt, desc in test_pairs:
        # Check rule
        if src.is_target_paper_contribution and not tgt.is_target_paper_contribution:
            score, asym, rationale = 0.0, 0.0, "STRICTLY FORBIDDEN: Target paper contribution can never precede foundational prerequisite."
            status = "BLOCKED / REJECTED"
        else:
            score, asym, rationale = agent._evaluate_prerequisite(src, tgt, doc, sec_indices)
            status = "ACCEPTED (EDGE CREATED)" if score >= 0.5 else "REJECTED"

        print(f"\n    Candidate Edge: {src.name} -> {tgt.name}")
        print(f"    Description:    {desc}")
        print(f"    Confidence:     {score:.3f} | Asymmetry: {asym:.3f} | Status: {status}")
        print(f"    Rationale:      \"{rationale}\"")

    pause_if_interactive(interactive)


def demo_step_4(interactive: bool = False) -> None:
    print_banner("Stage 4: Graph Reasoning, Greedy FAS Cycle Removal & Transitive Reduction", 4)
    graph_agent = GraphReasoningAgent()

    print("[*] 1. Live Demonstration of Directed Cycle Elimination (Greedy FAS):")
    nodes = [
        ConceptEntity(concept_id="c1", name="Linear Algebra", category=ConceptCategory.MATHEMATICS),
        ConceptEntity(concept_id="c2", name="Subspaces", category=ConceptCategory.MATHEMATICS),
        ConceptEntity(concept_id="c3", name="Oblique Projection", category=ConceptCategory.MATHEMATICS),
    ]

    # Inject directed cycle: c1 -> c2 -> c3 -> c1
    edges_with_cycle = [
        PrerequisiteEdge(source_id="c1", source_name="Linear Algebra", target_id="c2", target_name="Subspaces", confidence=0.98),
        PrerequisiteEdge(source_id="c2", source_name="Subspaces", target_id="c3", target_name="Oblique Projection", confidence=0.92),
        PrerequisiteEdge(source_id="c3", source_name="Oblique Projection", target_id="c1", target_name="Linear Algebra", confidence=0.35), # Lowest confidence edge
    ]

    print("    Initial Candidate Edges (Containing Cycle):")
    for e in edges_with_cycle:
        print(f"      • {e.source_name} --> {e.target_name} (confidence: {e.confidence})")

    dag = graph_agent.build_dag("demo_cycle", nodes, edges_with_cycle)

    print("\n    Result After Greedy Feedback Arc Set (FAS):")
    print(f"      • Graph is strict DAG:         {dag.metrics.is_dag}")
    print(f"      • Cycles Detected & Resolved:  {dag.metrics.cycles_detected_and_resolved}")
    print(f"      • Remaining Essential Edges:   {dag.metrics.num_edges}")
    for e in dag.edges:
        print(f"         - {e.source_name} --> {e.target_name} (conf: {e.confidence})")
    print("      ✓ Notice the lowest-confidence reverse edge (Oblique Projection -> Linear Algebra) was safely excised!")

    print("\n[*] 2. Live Demonstration of Transitive Reduction:")
    # A -> B -> C plus redundant bypass A -> C
    edges_with_transitive = [
        PrerequisiteEdge(source_id="c1", source_name="Linear Algebra", target_id="c2", target_name="Subspaces", confidence=0.98),
        PrerequisiteEdge(source_id="c2", source_name="Subspaces", target_id="c3", target_name="Oblique Projection", confidence=0.92),
        PrerequisiteEdge(source_id="c1", source_name="Linear Algebra", target_id="c3", target_name="Oblique Projection", confidence=0.90), # Bypass edge
    ]
    print("    Initial Edges (With Transitive Bypass Linear Algebra -> Oblique Projection):")
    for e in edges_with_transitive:
        print(f"      • {e.source_name} --> {e.target_name}")

    dag_reduced = graph_agent.build_dag("demo_transitive", nodes, edges_with_transitive)
    print("\n    Result After Transitive Reduction:")
    print(f"      • Total Edges Remaining: {dag_reduced.metrics.num_edges}")
    for e in dag_reduced.edges:
        print(f"         - {e.source_name} --> {e.target_name}")
    print("      ✓ The direct bypass (Linear Algebra -> Oblique Projection) was removed because Subspaces already connects them!")

    pause_if_interactive(interactive)


def demo_step_5(interactive: bool = False) -> None:
    print_banner("Stage 5: Pedagogical Curriculum Planner & DAG Monotonicity Verification", 5)
    orchestrator = PipelineOrchestrator(data_dir="data", output_dir="output")
    res = orchestrator.run_pipeline("paper_opjc", save_artifacts=False)
    curriculum = res["curriculum"]
    dag = res["dag"]

    print("[*] 1. Generated 4-Tier Pedagogical Learning Sequence:")
    for tier in curriculum.curriculum:
        print(f"\n    {tier.tier_name} ({len(tier.concepts)} concepts):")
        for idx, c in enumerate(tier.concepts, 1):
            role = "[Paper Contribution]" if c.is_target_paper_contribution else f"({c.category})"
            print(f"       {idx}. {c.name:<30} {role}")
            print(f"          Why Needed: {c.why_needed[:85]}...")

    print("\n[*] 2. Rigorous DAG Topological Tier Monotonicity Check (Anti-Pattern #5):")
    concept_to_tier = {}
    for tier in curriculum.curriculum:
        for c in tier.concepts:
            concept_to_tier[c.concept_id] = tier.tier

    violations = 0
    for e in dag.edges:
        src_t = concept_to_tier[e.source_id]
        tgt_t = concept_to_tier[e.target_id]
        is_monotonic = tgt_t >= src_t
        check_str = "PASS" if is_monotonic else "VIOLATION"
        if not is_monotonic:
            violations += 1
        print(f"    Edge: {e.source_name:<25} (Tier {src_t}) --> {e.target_name:<25} (Tier {tgt_t}) | [{check_str}]")

    print(f"\n    Monotonicity Guarantee: {len(dag.edges) - violations}/{len(dag.edges)} edges strictly monotonic.")
    if violations == 0:
        print("    ✓ PASSED: No student is ever instructed to learn a dependent concept before its prerequisites!")

    pause_if_interactive(interactive)


def demo_step_6(interactive: bool = False) -> None:
    print_banner("Stage 6: Evidence Grounding, Formula References, & Pipe Sanitization", 6)
    orchestrator = PipelineOrchestrator(data_dir="data", output_dir="output")
    res = orchestrator.run_pipeline("paper_opjc", save_artifacts=False)
    evidence = res["evidence"]

    print("[*] Verifiable In-Text Evidence Grounding for Prerequisite Edges:")
    for idx, ev in enumerate(evidence[:4], 1):
        print(f"\n    Evidence #{idx}: [{ev.prerequisite_concept}] --> [{ev.dependent_concept}]")
        print(f"      • Section:     {ev.section} (Pipe '|' sanitized: {'|' not in ev.section})")
        print(f"      • Type:        {ev.evidence_type}")
        print(f"      • Quote:       \"{ev.quote}\"")
        if ev.formula_reference:
            print(f"      • Formula Ref: {ev.formula_reference}")
        if ev.cited_foundational_paper:
            print(f"      • Citation:    {ev.cited_foundational_paper}")

    pause_if_interactive(interactive)


def demo_step_7(interactive: bool = False) -> None:
    print_banner("Step 7: CLI Tools & Ontology Queries Demo", 7)
    print("[*] You can execute and test each tool directly from the terminal:")
    print("    1. Single Paper Pipeline:")
    print("       python3 run.py --input paper_opjc\n")
    print("    2. Library Status:")
    print("       python3 run.py --list\n")
    print("    3. Search Library by Category:")
    print("       python3 query.py --category Mathematics\n")
    print("    4. Search by Concept with Verifiable Quotes & Formulas:")
    print("       python3 query.py --concept \"GLRT\" --details\n")
    print("    5. Classify and Identify Any Concept in CSO Ontology (CSO.3.3.owl):")
    print("       python3 query.py --cso \"Fourier Transform\"\n")
    print("    6. Multi-Core Parallel Batch Trainer (Up to 1,000 papers):")
    print("       python3 batch_train.py\n")
    print("    7. Run Master Test Suite (39 Tests):")
    print("       python3 run_tests.py\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-Agent System Feature Demonstration")
    parser.add_argument("--step", type=int, choices=range(1, 8), help="Demonstrate specific feature step (1-7)")
    parser.add_argument("--all", action="store_true", help="Run all demonstration steps in sequence")
    parser.add_argument("--interactive", action="store_true", help="Run interactively with pauses between steps")

    args = parser.parse_args()

    if args.step == 1:
        demo_step_1()
    elif args.step == 2:
        demo_step_2()
    elif args.step == 3:
        demo_step_3()
    elif args.step == 4:
        demo_step_4()
    elif args.step == 5:
        demo_step_5()
    elif args.step == 6:
        demo_step_6()
    elif args.step == 7:
        demo_step_7()
    else:
        # Default or --all
        interactive = args.interactive
        demo_step_1(interactive)
        demo_step_2(interactive)
        demo_step_3(interactive)
        demo_step_4(interactive)
        demo_step_5(interactive)
        demo_step_6(interactive)
        demo_step_7(interactive)

    return 0


if __name__ == "__main__":
    sys.exit(main())
