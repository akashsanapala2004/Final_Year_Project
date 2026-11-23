#!/usr/bin/env python3
"""Master CLI Runner for Scientific Prerequisite Concept Extraction & Knowledge Graph System.

Usage:
    python3 run.py --input <paper_id_or_path>
    python3 run.py --list
    python3 run.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from src.agents.orchestrator import PipelineOrchestrator


def get_available_paper_ids(data_dir: Path) -> List[str]:
    """Find all paper IDs available in grobid_output or raw_pdfs."""
    grobid_dir = data_dir / "grobid_output"
    pdf_dir = data_dir / "raw_pdfs"
    paper_ids = set()

    if grobid_dir.is_dir():
        for f in grobid_dir.glob("*.xml"):
            name = f.stem.replace(".tei", "")
            paper_ids.add(name)

    if pdf_dir.is_dir():
        for f in pdf_dir.glob("*.pdf"):
            paper_ids.add(f.stem)

    return sorted(list(paper_ids))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scientific Prerequisite Extraction & Curriculum Planning CLI"
    )
    parser.add_argument(
        "--input", "-i", type=str, help="Paper ID or XML/PDF file path to process"
    )
    parser.add_argument(
        "--list", "-l", action="store_true", help="List all available papers in data directory"
    )
    parser.add_argument(
        "--all", "-a", action="store_true", help="Process all available papers in data directory"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data", help="Directory containing input data"
    )
    parser.add_argument(
        "--output-dir", type=str, default="output", help="Directory to save generated curricula and graphs"
    )

    args = parser.parse_args()
    data_path = Path(args.data_dir)
    output_path = Path(args.output_dir)

    orchestrator = PipelineOrchestrator(data_dir=data_path, output_dir=output_path)

    if args.list:
        papers = get_available_paper_ids(data_path)
        print("=" * 70)
        print(f"AVAILABLE PAPERS IN LIBRARY ({len(papers)} total):")
        print("=" * 70)
        if not papers:
            print("No papers found in data/grobid_output/ or data/raw_pdfs/.")
            print("Add XML files to data/grobid_output/ or PDFs to data/raw_pdfs/.")
            return 0

        for pid in papers:
            json_exists = (output_path / f"{pid}_prerequisite_curriculum.json").exists()
            status = "✓ Processed" if json_exists else "○ Pending"
            print(f"  • {pid:<45} [{status}]")
        print("=" * 70)
        return 0

    if args.all:
        papers = get_available_paper_ids(data_path)
        if not papers:
            print("No papers found to process. Please place TEI-XML files in data/grobid_output/.")
            return 1

        print("=" * 70)
        print(f"BATCH PROCESSING ALL {len(papers)} PAPERS...")
        print("=" * 70)
        success_count = 0
        for pid in papers:
            print(f"\n[+] Processing: {pid}")
            try:
                res = orchestrator.run_pipeline(pid, save_artifacts=True)
                print(f"    ✓ Success: {res['document'].title}")
                print(f"    - Concepts Extracted: {len(res['concepts'])}")
                print(f"    - Essential DAG Edges: {len(res['edges'])}")
                print(f"    - Total Prerequisites: {res['curriculum'].total_prerequisites}")
                print(f"    - Output: {res['paths'].get('markdown', '')}")
                success_count += 1
            except Exception as e:
                print(f"    ✗ Error processing {pid}: {e}")

        print("\n" + "=" * 70)
        print(f"COMPLETED: {success_count}/{len(papers)} papers processed successfully.")
        print("=" * 70)
        return 0

    if args.input:
        paper_input = args.input
        print("=" * 70)
        print(f"SCIENTIFIC PREREQUISITE CURRICULUM PIPELINE")
        print(f"Input: {paper_input}")
        print("=" * 70)
        try:
            res = orchestrator.run_pipeline(paper_input, save_artifacts=True)
            doc = res["document"]
            dag = res["dag"]
            curriculum = res["curriculum"]
            paths = res["paths"]

            print(f"\nTitle:    {doc.title}")
            print(f"Paper ID: {doc.paper_id}")
            print(f"Concepts: {len(res['concepts'])} extracted")
            print(f"DAG:      {dag.metrics.num_nodes} nodes, {dag.metrics.num_edges} edges (Density: {dag.metrics.density})")
            print(f"Acyclic:  Strict DAG = {dag.metrics.is_dag} (Cycles Resolved: {dag.metrics.cycles_detected_and_resolved})")
            print(f"Prereqs:  {curriculum.total_prerequisites} foundational prerequisites across 4 tiers\n")

            print("--- Pedagogical Learning Sequence ---")
            for tier in curriculum.curriculum:
                print(f"\n{tier.tier_name}:")
                if not tier.concepts:
                    print("  (None)")
                for idx, c in enumerate(tier.concepts, 1):
                    role = "[Paper Contribution]" if c.is_target_paper_contribution else f"({c.category})"
                    print(f"  {idx}. {c.name} {role}")
                    print(f"     -> Why Needed: {c.why_needed}")

            print("\n" + "=" * 70)
            print("ARTIFACTS SAVED:")
            for k, p in paths.items():
                print(f"  • {k.upper():<10}: {p}")
            print("=" * 70)
            return 0
        except Exception as e:
            print(f"\n[ERROR] Pipeline failed for {paper_input}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
